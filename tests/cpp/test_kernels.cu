// GPU correctness tests. Every kernel passes its test here before it is ever
// timed; that ordering is a hard rule on this project, not a preference.
//
// On tolerances. A GPU sums in a different order than a sequential CPU loop, and
// fp32 addition is not associative, so demanding an exact match would be wrong
// rather than strict. Where a kernel only moves data (transpose) the comparison
// is exact, because there is no arithmetic to reorder. Where a kernel does
// elementwise arithmetic in the same order as the reference (SAXPY) the
// tolerance is tight. Where a kernel reorders a long accumulation (reduction)
// the reference is computed in double and the comparison is a relative error
// scaled to the length of the sum.

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels.hpp"

namespace {

// Small RAII helper so a failing assertion cannot leak device memory.
template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& host) const {
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, host.size() * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

   private:
    T* ptr_ = nullptr;
    size_t count_;
};

std::vector<float> random_vector(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& value : v) {
        value = dist(gen);
    }
    return v;
}

}  // namespace

// ---------------------------------------------------------------- SAXPY

class SaxpyTest : public ::testing::TestWithParam<int> {};

TEST_P(SaxpyTest, MatchesCpuReference) {
    const int n = GetParam();
    const float a = 2.5f;
    const auto x = random_vector(n, 1234);
    const auto y = random_vector(n, 5678);

    // Reference in the same order and precision the kernel uses, so any
    // discrepancy is a real bug and not an accumulation artifact.
    std::vector<float> expected(n);
    for (int i = 0; i < n; ++i) {
        expected[i] = a * x[i] + y[i];
    }

    DeviceBuffer<float> dx(n), dy(n);
    dx.upload(x);
    dy.upload(y);
    roofline::launch_saxpy(a, dx.get(), dy.get(), n, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(n);
    dy.download(got);
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(got[i], expected[i], 1e-5f) << "element " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(Sizes, SaxpyTest,
                         ::testing::Values(1, 31, 256, 1000, 65536, 1048576));

// ------------------------------------------------------------ Reduction

class ReductionTest : public ::testing::TestWithParam<int> {};

TEST_P(ReductionTest, MatchesCpuReferenceIncludingNonPowerOfTwo) {
    const int n = GetParam();
    const auto in = random_vector(n, 99);

    // Double precision reference: the GPU reorders this sum, so comparing
    // against a float sequential sum would measure the reference's error too.
    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        expected += static_cast<double>(in[i]);
    }

    DeviceBuffer<float> din(n), dout(1),
        dscratch(roofline::reduction_scratch_elements());
    din.upload(in);
    roofline::launch_reduction(din.get(), dout.get(), dscratch.get(), n, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(1);
    dout.download(got);

    // Relative tolerance scaled by the magnitude of the terms summed. fp32 has
    // about 1e-7 relative precision per operation and a tree reduction over n
    // terms accumulates roughly log2(n) of them, so this is generous but still
    // tight enough to catch a genuinely wrong sum.
    const double scale = std::max(1.0, static_cast<double>(n)) * 1e-6;
    EXPECT_NEAR(static_cast<double>(got[0]), expected, scale)
        << "n = " << n;
}

INSTANTIATE_TEST_SUITE_P(Sizes, ReductionTest,
                         ::testing::Values(1, 3, 31, 255, 1000, 1000003,
                                           1048576, 4194304));

// ------------------------------------------------------------ Transpose

class TransposeTest
    : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(TransposeTest, NaiveAndTiledBothMatchCpuReferenceExactly) {
    const auto [rows, cols, tile] = GetParam();
    const auto in = random_vector(static_cast<size_t>(rows) * cols, 7);

    // Pure data movement, so the comparison is exact. Any difference at all is
    // an indexing bug.
    std::vector<float> expected(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            expected[static_cast<size_t>(c) * rows + r] =
                in[static_cast<size_t>(r) * cols + c];
        }
    }

    const size_t count = static_cast<size_t>(rows) * cols;
    DeviceBuffer<float> din(count), dnaive(count), dtiled(count);
    din.upload(in);

    roofline::launch_transpose_naive(din.get(), dnaive.get(), rows, cols, tile);
    roofline::launch_transpose_tiled(din.get(), dtiled.get(), rows, cols, tile);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got_naive(count), got_tiled(count);
    dnaive.download(got_naive);
    dtiled.download(got_tiled);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_FLOAT_EQ(got_naive[i], expected[i]) << "naive, index " << i;
        ASSERT_FLOAT_EQ(got_tiled[i], expected[i]) << "tiled, index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, TransposeTest,
    ::testing::Values(
        // Square and exactly tile aligned.
        std::make_tuple(32, 32, 32), std::make_tuple(64, 64, 16),
        std::make_tuple(256, 256, 32),
        // Non square, to catch a rows/cols mix up that a square case hides.
        std::make_tuple(64, 128, 32), std::make_tuple(128, 64, 32),
        // Not a multiple of the tile, to exercise the boundary guards.
        std::make_tuple(100, 70, 32), std::make_tuple(33, 17, 16)));
