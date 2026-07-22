// The benchmark timing harness.
//
// Rules this enforces:
//
//   * CUDA events, never host wall clock, so what is measured is device time.
//   * Many launches between one pair of events, then divide, so per launch
//     overhead does not swamp the cheap kernels like SAXPY.
//   * Warmup iterations discarded first, so the boost clock has ramped and the
//     caches are warm before anything counts.
//   * At least ten timed batches, reporting mean, median, and standard
//     deviation. On a boost clocked consumer card that also drives a display,
//     the spread is real information and is not averaged away.

#ifndef ROOFLINE_TIMING_HPP
#define ROOFLINE_TIMING_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace roofline {

struct TimingResult {
    double mean_ms = 0.0;    // per launch
    double median_ms = 0.0;  // per launch
    double stddev_ms = 0.0;  // across batches, per launch
    double min_ms = 0.0;
    int batches = 0;
    int launches_per_batch = 0;
};

// Time a kernel launch expressed as a callable taking no arguments. The callable
// must only launch; it must not synchronize, or the batching is pointless.
template <typename LaunchFn>
TimingResult time_kernel(LaunchFn&& launch, int warmup_iterations,
                         int timed_batches, int launches_per_batch) {
    TimingResult result;
    result.batches = timed_batches;
    result.launches_per_batch = launches_per_batch;
    if (timed_batches <= 0 || launches_per_batch <= 0) {
        return result;
    }

    // Discarded: clock ramp and cache warming.
    for (int i = 0; i < warmup_iterations; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> per_launch_ms;
    per_launch_ms.reserve(timed_batches);

    for (int batch = 0; batch < timed_batches; ++batch) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < launches_per_batch; ++i) {
            launch();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        per_launch_ms.push_back(static_cast<double>(ms) / launches_per_batch);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double sum = 0.0;
    for (double v : per_launch_ms) {
        sum += v;
    }
    result.mean_ms = sum / per_launch_ms.size();

    std::vector<double> sorted = per_launch_ms;
    std::sort(sorted.begin(), sorted.end());
    const size_t mid = sorted.size() / 2;
    result.median_ms = (sorted.size() % 2 == 0)
                           ? 0.5 * (sorted[mid - 1] + sorted[mid])
                           : sorted[mid];
    result.min_ms = sorted.front();

    double variance = 0.0;
    for (double v : per_launch_ms) {
        const double d = v - result.mean_ms;
        variance += d * d;
    }
    // Sample standard deviation; with a single batch there is no spread to
    // report and it stays zero rather than dividing by zero.
    result.stddev_ms =
        per_launch_ms.size() > 1
            ? std::sqrt(variance / (per_launch_ms.size() - 1))
            : 0.0;

    return result;
}

// Pick how many launches to put between one pair of events.
//
// A single fixed count cannot serve this suite. SAXPY at a million elements
// takes tens of microseconds, so timing one launch at a time would measure
// mostly launch overhead and event resolution. A naive GEMM at 4096 takes
// hundreds of milliseconds, so fifty launches per batch would mean minutes per
// cell and hours for the sweep. So the harness measures one launch first and
// picks a count that makes a batch last roughly target_batch_ms, capped by the
// configured maximum.
template <typename LaunchFn>
int calibrate_launches_per_batch(LaunchFn&& launch, double target_batch_ms,
                                 int max_launches) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float single_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&single_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (single_ms <= 0.0f) {
        // Below the event timer's resolution, so batch as much as allowed.
        return max_launches;
    }
    const int wanted = static_cast<int>(target_batch_ms / single_ms);
    return std::max(1, std::min(wanted, max_launches));
}

// Time a kernel, choosing the batch size automatically.
template <typename LaunchFn>
TimingResult time_kernel_auto(LaunchFn&& launch, int warmup_iterations,
                              int timed_batches, int max_launches_per_batch,
                              double target_batch_ms = 50.0) {
    const int per_batch = calibrate_launches_per_batch(
        launch, target_batch_ms, max_launches_per_batch);
    return time_kernel(launch, warmup_iterations, timed_batches, per_batch);
}

// Achieved throughput in FLOP/s from a kernel's actual arithmetic and its mean
// per launch time.
inline double achieved_flops(double flop_count, double mean_ms) {
    return (mean_ms > 0.0) ? flop_count / (mean_ms * 1.0e-3) : 0.0;
}

// Achieved bandwidth in byte/s, used for the kernels whose story is traffic
// rather than arithmetic, such as the transpose pair.
inline double achieved_bandwidth(double byte_count, double mean_ms) {
    return (mean_ms > 0.0) ? byte_count / (mean_ms * 1.0e-3) : 0.0;
}

}  // namespace roofline

#endif  // ROOFLINE_TIMING_HPP
