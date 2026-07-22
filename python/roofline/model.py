"""The roofline model itself: ceilings, ridge point, arithmetic intensity, and
attainable performance.

Everything here is pure arithmetic on floats, no I/O and no plotting, so it is
cheap to unit test against hand computed cases. The roofline bound
(Williams, Waterman, Patterson, 2009) is

    attainable_flops(ai) = min(peak_flops, ai * peak_bandwidth)

with arithmetic intensity ai in FLOP per byte, peak_flops in FLOP/s, and
peak_bandwidth in byte/s. The ridge point is the intensity where the sloped
memory ceiling meets the flat compute ceiling; left of it a kernel is memory
bound, right of it compute bound.

Units are SI throughout (FLOP/s and byte/s), converted to the friendlier
GFLOP/s and GB/s only at the plotting and table edges. Keeping one unit system
in the math is what stops a stray factor of 1e9 from quietly wrecking a plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GIGA = 1.0e9


class IntensitySource(str, Enum):
    """Where an arithmetic intensity's byte count came from.

    The distinction is not cosmetic. Measured DRAM bytes from Nsight Compute
    include the traffic a kernel actually generated (cache misses, spills,
    replayed loads); the theoretical count is the minimum a perfect
    implementation would move. The gap between the two intensities is most of
    the interesting story, so every point carries its source and plots label it.
    """

    MEASURED = "measured"
    THEORETICAL = "theoretical"


@dataclass(frozen=True)
class Ceilings:
    """The two roofline ceilings for one hardware configuration.

    Attributes:
        peak_flops: Compute ceiling in FLOP/s (the flat roof).
        peak_bandwidth: Memory ceiling slope in byte/s (the sloped roof).
        label: Human name for this pair, for example "theoretical" or
            "measured (FMA + copy)".
    """

    peak_flops: float
    peak_bandwidth: float
    label: str

    def __post_init__(self) -> None:
        if self.peak_flops <= 0.0:
            raise ValueError(f"peak_flops must be positive, got {self.peak_flops}")
        if self.peak_bandwidth <= 0.0:
            raise ValueError(
                f"peak_bandwidth must be positive, got {self.peak_bandwidth}"
            )

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) where memory and compute roofs meet."""
        return ridge_point(self.peak_flops, self.peak_bandwidth)

    def attainable(self, arithmetic_intensity: float) -> float:
        """Roofline bound in FLOP/s at the given intensity."""
        return attainable_flops(
            arithmetic_intensity, self.peak_flops, self.peak_bandwidth
        )


def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    """Return the ridge point intensity in FLOP/byte.

    This is peak_flops / peak_bandwidth: the intensity below which a kernel
    cannot reach the compute ceiling no matter how efficient it is, because the
    memory system cannot feed it fast enough.
    """
    _require_positive(peak_flops, "peak_flops")
    _require_positive(peak_bandwidth, "peak_bandwidth")
    return peak_flops / peak_bandwidth


def attainable_flops(
    arithmetic_intensity: float, peak_flops: float, peak_bandwidth: float
) -> float:
    """Return the roofline attainable performance in FLOP/s.

    min(peak_flops, arithmetic_intensity * peak_bandwidth). A kernel can never
    beat this bound; how close it gets is what the whole suite measures.
    """
    _require_nonnegative(arithmetic_intensity, "arithmetic_intensity")
    _require_positive(peak_flops, "peak_flops")
    _require_positive(peak_bandwidth, "peak_bandwidth")
    return min(peak_flops, arithmetic_intensity * peak_bandwidth)


def arithmetic_intensity(flops: float, num_bytes: float) -> float:
    """Return arithmetic intensity in FLOP/byte for a kernel run.

    flops is the kernel's actual floating point work (for example 2*M*N*K for a
    GEMM); num_bytes is the bytes moved to and from DRAM. A transpose does zero
    floating point work, so its intensity is zero and callers must be ready for
    that rather than dividing blindly.
    """
    _require_nonnegative(flops, "flops")
    _require_positive(num_bytes, "num_bytes")
    return flops / num_bytes


def achieved_flops(flops: float, seconds: float) -> float:
    """Return achieved performance in FLOP/s from work done and wall time.

    seconds is the mean per launch device time from cudaEvent timing, not host
    wall clock.
    """
    _require_nonnegative(flops, "flops")
    _require_positive(seconds, "seconds")
    return flops / seconds


def resolve_intensity(
    flops: float,
    theoretical_bytes: float,
    measured_bytes: float | None = None,
) -> tuple[float, IntensitySource]:
    """Pick the best available arithmetic intensity and say where it came from.

    Prefers measured DRAM bytes from Nsight Compute when present and positive,
    and falls back to the theoretical byte count otherwise. The returned source
    is what the plot legend and tables use to label the point honestly, so a
    reader always knows whether a point sits on measured or modelled traffic.
    """
    if measured_bytes is not None and measured_bytes > 0.0:
        return arithmetic_intensity(flops, measured_bytes), IntensitySource.MEASURED
    return arithmetic_intensity(flops, theoretical_bytes), IntensitySource.THEORETICAL


def utilization(achieved: float, ceiling: float) -> float:
    """Return achieved performance as a fraction of a ceiling, in [0, inf).

    Used both for the compute ceiling (achieved GFLOP/s over peak) and the
    memory ceiling (achieved GB/s over peak bandwidth). Values above 1.0 signal
    a bug in the FLOP or byte counting rather than a physical miracle, so
    callers should surface them loudly.
    """
    _require_nonnegative(achieved, "achieved")
    _require_positive(ceiling, "ceiling")
    return achieved / ceiling


def is_memory_bound(arithmetic_intensity_value: float, ceilings: Ceilings) -> bool:
    """True if the kernel's intensity places it left of the ridge point."""
    _require_nonnegative(arithmetic_intensity_value, "arithmetic_intensity_value")
    return arithmetic_intensity_value < ceilings.ridge_point


def theoretical_fp32_peak_flops(
    num_sm: int, fp32_cores_per_sm: int, boost_clock_hz: float
) -> float:
    """Derive the theoretical FP32 compute ceiling in FLOP/s.

    cores * 2 * clock, where the factor of 2 counts a fused multiply add as two
    floating point operations. The C++ side derives the same number from
    cudaGetDeviceProperties; this mirror exists so the Python tests can check the
    formula against hand computed cases. A result far from the vendor reference
    is a bug in the derivation, never a hardware surprise.
    """
    if num_sm <= 0:
        raise ValueError(f"num_sm must be positive, got {num_sm}")
    if fp32_cores_per_sm <= 0:
        raise ValueError(
            f"fp32_cores_per_sm must be positive, got {fp32_cores_per_sm}"
        )
    _require_positive(boost_clock_hz, "boost_clock_hz")
    return float(num_sm) * float(fp32_cores_per_sm) * 2.0 * boost_clock_hz


def theoretical_bandwidth_bytes_per_s(
    bus_width_bits: int, memory_clock_hz: float, data_rate_factor: float = 2.0
) -> float:
    """Derive theoretical DRAM bandwidth in byte/s.

    (bus_width_bits / 8) * memory_clock_hz * data_rate_factor. The data rate
    factor is 2 for classic double data rate; GDDR7 quotes an effective transfer
    rate that already folds in its signalling, so callers pass the effective
    rate with a factor of 1 to avoid double counting. Which convention was used
    is recorded next to the call site.
    """
    if bus_width_bits <= 0:
        raise ValueError(f"bus_width_bits must be positive, got {bus_width_bits}")
    _require_positive(memory_clock_hz, "memory_clock_hz")
    _require_positive(data_rate_factor, "data_rate_factor")
    return (bus_width_bits / 8.0) * memory_clock_hz * data_rate_factor


def _require_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non negative, got {value}")
