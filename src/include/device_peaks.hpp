// Deriving and measuring this machine's two roofline ceilings.
//
// Theoretical peaks come from what the device reports about itself. Empirical
// peaks come from pushing the hardware: a register resident FMA loop for
// compute, a large device to device copy for bandwidth. Both go on every
// roofline plot, and the distance between them is one of the things the report
// is actually about, so they are kept separate rather than blended.

#ifndef ROOFLINE_DEVICE_PEAKS_HPP
#define ROOFLINE_DEVICE_PEAKS_HPP

#include <string>

namespace roofline {

struct DeviceInfo {
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    int sm_count = 0;
    int fp32_lanes_per_sm = 0;
    double sm_clock_hz = 0.0;
    double memory_clock_hz = 0.0;
    int memory_bus_width_bits = 0;
    size_t total_global_mem = 0;
    size_t shared_mem_per_block = 0;
    int l2_cache_bytes = 0;
    int warp_size = 0;
};

struct Peaks {
    // FLOP/s and byte/s, SI throughout. Converted to the friendlier GFLOP/s and
    // GB/s only at the reporting edge.
    double theoretical_flops = 0.0;
    double theoretical_bandwidth = 0.0;
    double measured_flops = 0.0;
    double measured_bandwidth = 0.0;
};

// Query the device. Reads the SM and memory clocks through
// cudaDeviceGetAttribute rather than cudaDeviceProp, because CUDA 13 removed
// clockRate and memoryClockRate from that struct.
DeviceInfo query_device_info(int device = 0);

// cores * 2 * clock, counting a fused multiply add as two operations.
double theoretical_fp32_flops(const DeviceInfo& info);

// (bus width / 8) * memory clock * 2 for the double data rate.
double theoretical_bandwidth(const DeviceInfo& info);

// Saturate the FP32 pipes with dependent-free fused multiply adds held entirely
// in registers, so nothing but arithmetic throughput is being measured.
double measure_fma_peak_flops(const DeviceInfo& info);

// Large device to device copy, counting both the read and the write.
double measure_copy_bandwidth(size_t bytes = 256u * 1024u * 1024u);

// Everything above, in one call.
Peaks measure_all_peaks(const DeviceInfo& info);

}  // namespace roofline

#endif  // ROOFLINE_DEVICE_PEAKS_HPP
