"""Roofline figures: log-log plots with both ceilings and every kernel as a
labeled point.

Matplotlib runs on the non interactive Agg backend so figures render on a
headless CI box and on the target machine identically. Every figure is written
as a vector PDF for the report and, optionally, a raster PNG for the README.
The plotting code stays thin: all the arithmetic it draws comes from
:mod:`roofline.model`, so a figure can never disagree with the tested math.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from roofline.model import Ceilings, IntensitySource  # noqa: E402

GIGA = 1.0e9


@dataclass(frozen=True)
class RooflinePoint:
    """One kernel configuration placed on the roofline.

    Attributes:
        label: Short kernel and size label, for example "gemm_tiled 4096".
        arithmetic_intensity: FLOP/byte for this run.
        achieved_gflops: Measured throughput in GFLOP/s.
        intensity_source: Whether the intensity used measured or theoretical
            bytes, which controls the marker style so the two are never confused.
    """

    label: str
    arithmetic_intensity: float
    achieved_gflops: float
    intensity_source: IntensitySource = IntensitySource.THEORETICAL


def plot_roofline(
    points: list[RooflinePoint],
    ceilings: list[Ceilings],
    out_pdf: str | Path,
    out_png: str | Path | None = None,
    title: str = "Roofline",
    ai_range: tuple[float, float] | None = None,
) -> Path:
    """Draw a log-log roofline and save it. Returns the PDF path written.

    Each entry in ceilings draws its own two segment roof (sloped memory bound
    then flat compute bound), visually distinct, so the theoretical and measured
    ceilings sit on the same axes for honest comparison. Points from measured
    byte counts and points from theoretical byte counts get different markers.
    """
    if not ceilings:
        raise ValueError("at least one Ceilings is required to draw a roofline")

    lo, hi = _intensity_axis_range(points, ceilings, ai_range)
    ai_grid = np.logspace(np.log10(lo), np.log10(hi), num=256)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for index, ceiling in enumerate(ceilings):
        attainable_gflops = np.array(
            [ceiling.attainable(ai) for ai in ai_grid]
        ) / GIGA
        ax.plot(
            ai_grid,
            attainable_gflops,
            linewidth=2.0,
            linestyle="-" if index == 0 else "--",
            label=f"{ceiling.label} ceiling",
            zorder=2,
        )
        ridge = ceiling.ridge_point
        if lo <= ridge <= hi:
            ax.axvline(ridge, color="0.7", linewidth=0.8, zorder=1)

    _scatter_points(ax, points)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("performance (GFLOP/s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_pdf


def _scatter_points(ax: plt.Axes, points: list[RooflinePoint]) -> None:
    for point in points:
        marker = "o" if point.intensity_source is IntensitySource.MEASURED else "s"
        ax.scatter(
            point.arithmetic_intensity,
            point.achieved_gflops,
            marker=marker,
            s=45,
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            point.label,
            (point.arithmetic_intensity, point.achieved_gflops),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
        )


def _intensity_axis_range(
    points: list[RooflinePoint],
    ceilings: list[Ceilings],
    ai_range: tuple[float, float] | None,
) -> tuple[float, float]:
    if ai_range is not None:
        lo, hi = ai_range
        if lo <= 0.0 or hi <= lo:
            raise ValueError(f"invalid ai_range {ai_range}")
        return lo, hi
    intensities = [p.arithmetic_intensity for p in points if p.arithmetic_intensity > 0]
    intensities.extend(c.ridge_point for c in ceilings)
    if not intensities:
        return 0.1, 100.0
    lo = min(intensities) / 4.0
    hi = max(intensities) * 4.0
    return max(lo, 1.0e-3), hi
