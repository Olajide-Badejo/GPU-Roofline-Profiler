// Sanity tests for the ceiling derivation, plus a report of the actual numbers
// this card produces.
//
// These are guard rails, not precise expectations. The point is to catch a
// derivation that is wrong by a factor (a missing multiply add factor of two, a
// missing double data rate, a clock read as kHz instead of Hz), because those
// are the failure modes that would silently move every point on the roofline.

#include <gtest/gtest.h>

#include <cstdio>

#include "device_peaks.hpp"

namespace {

constexpr double kGiga = 1.0e9;
constexpr double kTera = 1.0e12;

}  // namespace

TEST(DevicePeaks, ReportsPlausibleDeviceInfo) {
    const auto info = roofline::query_device_info();
    std::printf("\ndevice              : %s\n", info.name.c_str());
    std::printf("compute capability  : %d.%d\n", info.compute_major,
                info.compute_minor);
    std::printf("SMs                 : %d\n", info.sm_count);
    std::printf("FP32 lanes per SM   : %d\n", info.fp32_lanes_per_sm);
    std::printf("total FP32 lanes    : %d\n",
                info.sm_count * info.fp32_lanes_per_sm);
    std::printf("SM clock            : %.3f GHz\n", info.sm_clock_hz / kGiga);
    std::printf("memory clock        : %.3f GHz\n",
                info.memory_clock_hz / kGiga);
    std::printf("bus width           : %d bits\n", info.memory_bus_width_bits);

    EXPECT_GT(info.sm_count, 0);
    EXPECT_GT(info.fp32_lanes_per_sm, 0);
    // A clock read with the wrong unit scale is off by 1e3 or 1e6, which these
    // bounds catch while staying true for any modern card.
    EXPECT_GT(info.sm_clock_hz, 0.2e9);
    EXPECT_LT(info.sm_clock_hz, 6.0e9);
    EXPECT_GT(info.memory_clock_hz, 0.2e9);
    EXPECT_GT(info.memory_bus_width_bits, 0);
}

TEST(DevicePeaks, TheoreticalCeilingsAreSelfConsistent) {
    const auto info = roofline::query_device_info();
    const double flops = roofline::theoretical_fp32_flops(info);
    const double bandwidth = roofline::theoretical_bandwidth(info);

    std::printf("\ntheoretical FP32    : %.2f TFLOP/s\n", flops / kTera);
    std::printf("theoretical bandwidth: %.1f GB/s\n", bandwidth / kGiga);
    std::printf("ridge point         : %.2f FLOP/byte\n", flops / bandwidth);

    // The derivation must equal its own definition; this catches a typo in the
    // implementation rather than a misunderstanding of the hardware.
    EXPECT_DOUBLE_EQ(flops, static_cast<double>(info.sm_count) *
                                info.fp32_lanes_per_sm * 2.0 * info.sm_clock_hz);

    // Order of magnitude guards. Any consumer or datacentre GPU of the last
    // several generations lands inside these.
    EXPECT_GT(flops, 0.5 * kTera);
    EXPECT_LT(flops, 500.0 * kTera);
    EXPECT_GT(bandwidth, 50.0 * kGiga);
    EXPECT_LT(bandwidth, 10000.0 * kGiga);
}

TEST(DevicePeaks, MeasuredCeilingsAreBelowTheoreticalAndNotAbsurd) {
    const auto info = roofline::query_device_info();
    const auto peaks = roofline::measure_all_peaks(info);

    std::printf("\nmeasured FP32       : %.2f TFLOP/s (%.1f%% of theoretical)\n",
                peaks.measured_flops / kTera,
                100.0 * peaks.measured_flops / peaks.theoretical_flops);
    std::printf("measured bandwidth  : %.1f GB/s (%.1f%% of theoretical)\n",
                peaks.measured_bandwidth / kGiga,
                100.0 * peaks.measured_bandwidth / peaks.theoretical_bandwidth);

    EXPECT_GT(peaks.measured_flops, 0.0);
    EXPECT_GT(peaks.measured_bandwidth, 0.0);

    // Real hardware does not beat its own theoretical peak. A little headroom is
    // allowed because the boost clock can briefly exceed the reported base, but
    // a large overshoot means the FLOP or byte counting is wrong.
    EXPECT_LT(peaks.measured_flops, peaks.theoretical_flops * 1.10);
    EXPECT_LT(peaks.measured_bandwidth, peaks.theoretical_bandwidth * 1.10);

    // And a microbenchmark built to saturate its unit should get reasonably
    // close, or it is not measuring what it claims to.
    EXPECT_GT(peaks.measured_flops, peaks.theoretical_flops * 0.30);
    EXPECT_GT(peaks.measured_bandwidth, peaks.theoretical_bandwidth * 0.30);
}
