"""Unit tests for the roofline math against hand computed cases.

A wrong formula here silently invalidates every figure downstream, so these
check exact numbers, not just that a function runs.
"""

from __future__ import annotations

import math

import pytest

from roofline import model
from roofline.model import Ceilings, IntensitySource


def test_ridge_point_is_flops_over_bandwidth():
    # 30 TFLOP/s over 600 GB/s = 50 FLOP/byte, by hand.
    assert model.ridge_point(30.0e12, 600.0e9) == pytest.approx(50.0)


def test_attainable_is_memory_bound_below_ridge():
    # AI 10 FLOP/byte, well below the ridge at 50: bound is AI * bandwidth.
    got = model.attainable_flops(10.0, 30.0e12, 600.0e9)
    assert got == pytest.approx(10.0 * 600.0e9)


def test_attainable_is_compute_bound_above_ridge():
    # AI 100 FLOP/byte, above the ridge: bound saturates at peak flops.
    got = model.attainable_flops(100.0, 30.0e12, 600.0e9)
    assert got == pytest.approx(30.0e12)


def test_attainable_exactly_at_ridge_equals_peak():
    peak_flops, peak_bw = 30.0e12, 600.0e9
    ridge = model.ridge_point(peak_flops, peak_bw)
    got = model.attainable_flops(ridge, peak_flops, peak_bw)
    assert got == pytest.approx(peak_flops)


def test_arithmetic_intensity_basic():
    # GEMM 2*M*N*K flops over moved bytes.
    assert model.arithmetic_intensity(2.0 * 1024**3, 1024.0) == pytest.approx(
        2.0 * 1024**3 / 1024.0
    )


def test_arithmetic_intensity_rejects_zero_bytes():
    with pytest.raises(ValueError):
        model.arithmetic_intensity(1.0, 0.0)


def test_achieved_flops():
    # 2e9 flops in 1 ms is 2e12 FLOP/s.
    assert model.achieved_flops(2.0e9, 1.0e-3) == pytest.approx(2.0e12)


def test_resolve_intensity_prefers_measured():
    ai, source = model.resolve_intensity(
        flops=1000.0, theoretical_bytes=100.0, measured_bytes=200.0
    )
    assert source is IntensitySource.MEASURED
    assert ai == pytest.approx(5.0)


def test_resolve_intensity_falls_back_to_theoretical():
    ai, source = model.resolve_intensity(
        flops=1000.0, theoretical_bytes=100.0, measured_bytes=None
    )
    assert source is IntensitySource.THEORETICAL
    assert ai == pytest.approx(10.0)


def test_resolve_intensity_ignores_nonpositive_measured():
    _, source = model.resolve_intensity(
        flops=1000.0, theoretical_bytes=100.0, measured_bytes=0.0
    )
    assert source is IntensitySource.THEORETICAL


def test_ceilings_validate_positive():
    with pytest.raises(ValueError):
        Ceilings(peak_flops=-1.0, peak_bandwidth=1.0, label="bad")
    with pytest.raises(ValueError):
        Ceilings(peak_flops=1.0, peak_bandwidth=0.0, label="bad")


def test_ceilings_helpers_agree_with_free_functions():
    ceil = Ceilings(peak_flops=31.0e12, peak_bandwidth=672.0e9, label="theoretical")
    assert ceil.ridge_point == pytest.approx(model.ridge_point(31.0e12, 672.0e9))
    assert ceil.attainable(5.0) == pytest.approx(
        model.attainable_flops(5.0, 31.0e12, 672.0e9)
    )


def test_is_memory_bound():
    ceil = Ceilings(peak_flops=30.0e12, peak_bandwidth=600.0e9, label="t")
    assert model.is_memory_bound(10.0, ceil) is True
    assert model.is_memory_bound(90.0, ceil) is False


def test_utilization_flags_impossible_values():
    # Over unity is a counting bug, and the function reports it faithfully
    # rather than clamping.
    assert model.utilization(60.0e12, 30.0e12) == pytest.approx(2.0)


def test_theoretical_fp32_peak_matches_reference_order():
    # RTX 5070 reference: 6144 FP32 lanes near 2.51 GHz should land close to the
    # vendor quoted ~31 TFLOP/s. Model as 48 SMs * 128 lanes.
    peak = model.theoretical_fp32_peak_flops(
        num_sm=48, fp32_cores_per_sm=128, boost_clock_hz=2.51e9
    )
    assert peak == pytest.approx(48 * 128 * 2 * 2.51e9)
    assert 28.0e12 < peak < 34.0e12


def test_theoretical_bandwidth_gddr7_effective_rate():
    # 192 bit bus at an effective 28 Gbps per pin, factor 1 (rate already
    # effective), should land near the vendor quoted ~672 GB/s.
    bw = model.theoretical_bandwidth_bytes_per_s(
        bus_width_bits=192, memory_clock_hz=28.0e9, data_rate_factor=1.0
    )
    assert bw == pytest.approx(192 / 8 * 28.0e9)
    assert 650.0e9 < bw < 700.0e9


def test_peak_derivations_reject_bad_input():
    with pytest.raises(ValueError):
        model.theoretical_fp32_peak_flops(0, 128, 2.5e9)
    with pytest.raises(ValueError):
        model.theoretical_bandwidth_bytes_per_s(0, 1.0e9)


def test_ridge_point_rejects_nonpositive():
    with pytest.raises(ValueError):
        model.ridge_point(0.0, 1.0)


def test_finiteness_of_derived_numbers():
    peak = model.theoretical_fp32_peak_flops(48, 128, 2.51e9)
    assert math.isfinite(peak)
