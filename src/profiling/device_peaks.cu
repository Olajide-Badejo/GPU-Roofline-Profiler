// Implementation of the theoretical and empirical ceiling derivation.

#include "device_peaks.hpp"

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace roofline {
namespace {

// FP32 lanes per SM by compute capability. This is the one number that is not
// discoverable from the runtime, so it is a table, and an unknown architecture
// says so loudly rather than silently guessing a peak that would be wrong.
int fp32_lanes_for(int major, int minor) {
    switch (major) {
        case 7:  // Volta, Turing
            return (minor == 0) ? 64 : 64;
        case 8:  // Ampere, Ada
            return (minor == 0) ? 64 : 128;
        case 9:  // Hopper
            return 128;
        case 10:  // Blackwell datacentre
        case 12:  // Blackwell consumer, sm_120 (the RTX 5070)
            return 128;
        default:
            std::fprintf(stderr,
                         "unknown compute capability %d.%d; assuming 128 FP32 "
                         "lanes per SM, so the theoretical peak may be wrong\n",
                         major, minor);
            return 128;
    }
}

// Eight independent accumulators so the FMA pipeline is never waiting on a
// dependency, and an unrolled inner block so loop overhead is negligible next to
// the arithmetic.
constexpr int kFmaAccumulators = 8;
constexpr int kFmaUnroll = 32;

__global__ void fma_peak_kernel(float* out, int iters) {
    float b = 1.0000001f;
    float a = 0.9999999f;
    float acc[kFmaAccumulators];
#pragma unroll
    for (int i = 0; i < kFmaAccumulators; ++i) {
        acc[i] = static_cast<float>(threadIdx.x + i) * 1e-3f;
    }

    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int u = 0; u < kFmaUnroll; ++u) {
#pragma unroll
            for (int i = 0; i < kFmaAccumulators; ++i) {
                acc[i] = fmaf(acc[i], b, a);
            }
        }
    }

    // Consume the results so the compiler cannot delete the whole loop. The
    // branch is never taken at run time but the compiler cannot prove it.
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kFmaAccumulators; ++i) {
        sum += acc[i];
    }
    if (sum == 1.2345e30f) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    }
}

}  // namespace

DeviceInfo query_device_info(int device) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    DeviceInfo info;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.fp32_lanes_per_sm = fp32_lanes_for(prop.major, prop.minor);
    info.memory_bus_width_bits = prop.memoryBusWidth;
    info.total_global_mem = prop.totalGlobalMem;
    info.shared_mem_per_block = prop.sharedMemPerBlock;
    info.l2_cache_bytes = prop.l2CacheSize;
    info.warp_size = prop.warpSize;

    // CUDA 13 removed clockRate and memoryClockRate from cudaDeviceProp. The
    // attribute query is the supported replacement; both are checked so a future
    // removal fails loudly instead of yielding a peak of zero.
    int clock_khz = 0;
    int mem_clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate,
                                      device));
    info.sm_clock_hz = static_cast<double>(clock_khz) * 1.0e3;
    info.memory_clock_hz = static_cast<double>(mem_clock_khz) * 1.0e3;
    return info;
}

double theoretical_fp32_flops(const DeviceInfo& info) {
    return static_cast<double>(info.sm_count) * info.fp32_lanes_per_sm * 2.0 *
           info.sm_clock_hz;
}

double theoretical_bandwidth(const DeviceInfo& info) {
    // Double data rate: the reported memory clock is the command clock, and the
    // bus transfers on both edges. For this card 192 bits at 14.001 GHz gives
    // 672 GB/s, matching the vendor figure, which is the check that this
    // convention is the right one.
    return (info.memory_bus_width_bits / 8.0) * info.memory_clock_hz * 2.0;
}

double measure_fma_peak_flops(const DeviceInfo& info) {
    // Enough blocks to fill every SM several times over so the measurement is
    // throughput limited rather than occupancy limited.
    const int block_size = 256;
    const int blocks = info.sm_count * 8;
    const int iters = 4096;

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(blocks) * block_size *
                                      sizeof(float)));

    // Warmup, so the clock has ramped before the timed run.
    fma_peak_kernel<<<blocks, block_size>>>(d_out, 64);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double best_flops = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        fma_peak_kernel<<<blocks, block_size>>>(d_out, iters);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        const double threads = static_cast<double>(blocks) * block_size;
        const double fmas =
            threads * iters * kFmaUnroll * kFmaAccumulators;
        const double flops = fmas * 2.0 / (ms * 1.0e-3);
        best_flops = std::max(best_flops, flops);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
    return best_flops;
}

double measure_copy_bandwidth(size_t bytes) {
    // Keep well clear of the display's memory: shrink the buffers rather than
    // fail if the card is busy.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = free_bytes > (1024u * 1024u * 1024u)
                              ? (free_bytes - (1024u * 1024u * 1024u)) / 2
                              : free_bytes / 4;
    bytes = std::min(bytes, budget);
    bytes &= ~static_cast<size_t>(15);  // keep it a multiple of 16
    if (bytes == 0) {
        return 0.0;
    }

    void* src = nullptr;
    void* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemset(src, 1, bytes));

    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double best_bandwidth = 0.0;
    for (int trial = 0; trial < 10; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        // A copy both reads and writes, so it moves twice its size.
        const double moved = static_cast<double>(bytes) * 2.0;
        best_bandwidth = std::max(best_bandwidth, moved / (ms * 1.0e-3));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    return best_bandwidth;
}

Peaks measure_all_peaks(const DeviceInfo& info) {
    Peaks peaks;
    peaks.theoretical_flops = theoretical_fp32_flops(info);
    peaks.theoretical_bandwidth = theoretical_bandwidth(info);
    peaks.measured_flops = measure_fma_peak_flops(info);
    peaks.measured_bandwidth = measure_copy_bandwidth();
    return peaks;
}

}  // namespace roofline
