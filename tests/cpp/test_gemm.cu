// Correctness tests for GEMV and the whole GEMM ladder. No rung is timed until
// it passes here.
//
// On tolerances. A GEMM accumulates k products per output, and fp32 addition is
// not associative, so every variant sums in a different order and none of them
// will match a reference bit for bit. The reference is therefore computed in
// double, and the comparison is on relative error against the magnitude of the
// result. For k in the low thousands with operands in [-1, 1], fp32 error grows
// roughly with sqrt(k) times the machine epsilon of 1.2e-7, which is a few times
// 1e-5; 1e-4 relative leaves headroom for that without being loose enough to
// hide a real indexing bug, which shows up as an error of order 1, not 1e-4.

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include "kernels.hpp"
#include "test_helpers.hpp"

namespace {

using roofline_test::DeviceBuffer;
using roofline_test::random_vector;

// Row major reference GEMM, accumulated in double so the comparison measures the
// kernel's error and not the reference's.
std::vector<float> reference_gemm(const std::vector<float>& a,
                                  const std::vector<float>& b, int m, int n,
                                  int k) {
    std::vector<float> c(static_cast<size_t>(m) * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int p = 0; p < k; ++p) {
                acc += static_cast<double>(a[static_cast<size_t>(i) * k + p]) *
                       static_cast<double>(b[static_cast<size_t>(p) * n + j]);
            }
            c[static_cast<size_t>(i) * n + j] = static_cast<float>(acc);
        }
    }
    return c;
}

// Normwise relative error: the largest absolute deviation anywhere in the
// matrix, divided by the largest magnitude in the reference.
//
// The obvious alternative, a per element relative error, is the wrong tool here
// and I tried it first. With random operands in [-1, 1] a GEMM output is a sum
// of k signed products, so a good fraction of the entries land near zero through
// cancellation. Dividing a small absolute error by one of those near zero
// expected values produces an enormous ratio that says nothing about kernel
// quality. It failed every variant including cuBLAS by an identical margin,
// which is what gave the game away: independent implementations do not share a
// bug, but they do share a reference and a metric.
//
// Measuring against the magnitude of the result as a whole is the standard way
// to check a GEMM, and it still catches real defects: an indexing or tiling bug
// produces an error of order 1 relative to the norm, not of order 1e-6.
double max_relative_error(const std::vector<float>& got,
                          const std::vector<float>& expected) {
    double worst_diff = 0.0;
    double largest = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = static_cast<double>(expected[i]);
        const double d = std::abs(static_cast<double>(got[i]) - e);
        worst_diff = std::max(worst_diff, d);
        largest = std::max(largest, std::abs(e));
    }
    return worst_diff / std::max(largest, 1e-6);
}

constexpr double kGemmTolerance = 1e-4;

}  // namespace

// ---------------------------------------------------------------- GEMV

class GemvTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(GemvTest, MatchesCpuReference) {
    const auto [m, n] = GetParam();
    const auto a = random_vector(static_cast<size_t>(m) * n, 11);
    const auto x = random_vector(n, 22);

    std::vector<float> expected(m);
    for (int i = 0; i < m; ++i) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
            acc += static_cast<double>(a[static_cast<size_t>(i) * n + j]) *
                   static_cast<double>(x[j]);
        }
        expected[i] = static_cast<float>(acc);
    }

    DeviceBuffer<float> da(static_cast<size_t>(m) * n), dx(n), dy(m);
    da.upload(a);
    dx.upload(x);
    roofline::launch_gemv(da.get(), dx.get(), dy.get(), m, n, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(m);
    dy.download(got);
    EXPECT_LT(max_relative_error(got, expected), kGemmTolerance);
}

INSTANTIATE_TEST_SUITE_P(Shapes, GemvTest,
                         ::testing::Values(std::make_tuple(1, 1),
                                           std::make_tuple(17, 33),
                                           std::make_tuple(256, 256),
                                           std::make_tuple(1024, 512),
                                           std::make_tuple(512, 1024)));

// ---------------------------------------------------------------- GEMM

class GemmTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(GemmTest, AllVariantsMatchCpuReference) {
    const auto [m, n, k] = GetParam();
    const auto a = random_vector(static_cast<size_t>(m) * k, 31);
    const auto b = random_vector(static_cast<size_t>(k) * n, 41);
    const auto expected = reference_gemm(a, b, m, n, k);

    const size_t c_count = static_cast<size_t>(m) * n;
    DeviceBuffer<float> da(static_cast<size_t>(m) * k),
        db(static_cast<size_t>(k) * n), dc(c_count);
    da.upload(a);
    db.upload(b);
    std::vector<float> got(c_count);

    // Each variant writes every element of C, so a stale buffer between runs
    // cannot mask a gap; clearing anyway makes a partial write obvious.
    for (int tile : {16, 32}) {
        CUDA_CHECK(cudaMemset(dc.get(), 0, c_count * sizeof(float)));
        roofline::launch_gemm_naive(da.get(), db.get(), dc.get(), m, n, k, tile);
        CUDA_CHECK(cudaDeviceSynchronize());
        dc.download(got);
        EXPECT_LT(max_relative_error(got, expected), kGemmTolerance)
            << "naive, tile " << tile;

        CUDA_CHECK(cudaMemset(dc.get(), 0, c_count * sizeof(float)));
        roofline::launch_gemm_tiled(da.get(), db.get(), dc.get(), m, n, k, tile);
        CUDA_CHECK(cudaDeviceSynchronize());
        dc.download(got);
        EXPECT_LT(max_relative_error(got, expected), kGemmTolerance)
            << "tiled, tile " << tile;
    }

    CUDA_CHECK(cudaMemset(dc.get(), 0, c_count * sizeof(float)));
    roofline::launch_gemm_register_blocked(da.get(), db.get(), dc.get(), m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    dc.download(got);
    EXPECT_LT(max_relative_error(got, expected), kGemmTolerance)
        << "register blocked";

    CUDA_CHECK(cudaMemset(dc.get(), 0, c_count * sizeof(float)));
    const bool vectorized_ran = roofline::launch_gemm_vectorized(
        da.get(), db.get(), dc.get(), m, n, k);
    if (vectorized_ran) {
        CUDA_CHECK(cudaDeviceSynchronize());
        dc.download(got);
        EXPECT_LT(max_relative_error(got, expected), kGemmTolerance)
            << "vectorized";
    } else {
        // Refusing an unaligned shape is the documented contract, not a failure.
        EXPECT_TRUE((k % 4) != 0 || (n % 4) != 0)
            << "vectorized refused an aligned shape";
    }

    CUDA_CHECK(cudaMemset(dc.get(), 0, c_count * sizeof(float)));
    roofline::launch_gemm_cublas(da.get(), db.get(), dc.get(), m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    dc.download(got);
    EXPECT_LT(max_relative_error(got, expected), kGemmTolerance) << "cuBLAS";
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, GemmTest,
    ::testing::Values(
        // Tile aligned squares.
        std::make_tuple(64, 64, 64), std::make_tuple(128, 128, 128),
        std::make_tuple(256, 256, 256),
        // Rectangular, to catch an m/n/k mix up a square case hides.
        std::make_tuple(128, 64, 256), std::make_tuple(64, 256, 128),
        // Not multiples of the tile or the block, exercising every guard.
        std::make_tuple(100, 100, 100), std::make_tuple(65, 33, 17),
        std::make_tuple(1, 1, 1), std::make_tuple(3, 5, 7)));

// A larger case checked against cuBLAS rather than a CPU triple loop, which is
// the right reference at this size: the CPU reference is O(n^3) on the host and
// gets slow well before the sizes the sweep actually benchmarks.
TEST(GemmLarge, CustomKernelsAgreeWithCublas) {
    constexpr int kN = 512;
    const auto a = random_vector(static_cast<size_t>(kN) * kN, 51);
    const auto b = random_vector(static_cast<size_t>(kN) * kN, 61);
    const size_t count = static_cast<size_t>(kN) * kN;

    DeviceBuffer<float> da(count), db(count), dc(count), dref(count);
    da.upload(a);
    db.upload(b);

    roofline::launch_gemm_cublas(da.get(), db.get(), dref.get(), kN, kN, kN);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> reference(count);
    dref.download(reference);

    std::vector<float> got(count);

    roofline::launch_gemm_tiled(da.get(), db.get(), dc.get(), kN, kN, kN, 32);
    CUDA_CHECK(cudaDeviceSynchronize());
    dc.download(got);
    EXPECT_LT(max_relative_error(got, reference), kGemmTolerance) << "tiled";

    roofline::launch_gemm_register_blocked(da.get(), db.get(), dc.get(), kN, kN,
                                           kN);
    CUDA_CHECK(cudaDeviceSynchronize());
    dc.download(got);
    EXPECT_LT(max_relative_error(got, reference), kGemmTolerance)
        << "register blocked";

    ASSERT_TRUE(roofline::launch_gemm_vectorized(da.get(), db.get(), dc.get(),
                                                 kN, kN, kN));
    CUDA_CHECK(cudaDeviceSynchronize());
    dc.download(got);
    EXPECT_LT(max_relative_error(got, reference), kGemmTolerance) << "vectorized";
}
