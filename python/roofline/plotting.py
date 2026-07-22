"""Roofline figures: log-log plots with both ceilings and the kernel ladder.

Matplotlib runs on the non interactive Agg backend so figures render identically
on a headless CI box and on the target machine. Every figure is written as a
vector PDF for the report and, optionally, a raster PNG for the README. The
plotting stays thin: all the arithmetic it draws comes from
:mod:`roofline.model`, so a figure can never disagree with the tested math.

On colour. The GEMM ladder colours (red, yellow, pink, brown) and the lemon
measured ceiling are chosen by the repo owner rather than derived from a
validated palette. Two consequences are worth stating plainly.

The ladder's ordering is no longer carried by shade. An earlier version drew the
five rungs as one hue from light to dark, so "more optimised" read as "darker"
without needing the legend. With categorical hues that cue is gone and the
reading rests on the legend order and the direct labels.

Lemon on a near white surface is a low contrast combination, and yellow for the
tiled rung sits close to it in hue. The mitigations are that the ceilings are
lines while the kernels are markers, so the two never compete for the same
shape, that the measured ceiling is drawn thicker than it otherwise would be,
and that every series still carries a distinct marker. Identity is never carried
by colour alone, which is what keeps the figure readable in print and for a
colourblind reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from roofline.model import Ceilings, IntensitySource  # noqa: E402

GIGA = 1.0e9

# Chart chrome, from the reference palette. Text and axes wear ink colours, never
# a series colour.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# GEMM ladder colours, chosen by the repo owner. These are categorical rather
# than an ordinal ramp, so the ladder's ordering is no longer carried by shade
# and rests entirely on the legend order and the labels.
_RED = "#d62728"
_YELLOW = "#e8c400"
_PINK = "#e377c2"
_BROWN = "#8c564b"
_NAVY = "#0d366b"

# Categorical slots for the non GEMM families.
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_VIOLET = "#4a3aa7"
_MAGENTA = "#e87ba4"

# Ceiling colours. Theoretical stays black; measured is lemon. Lemon is pale
# against a near white surface, so that line is drawn thicker than it otherwise
# would be to keep it legible.
CEILING_THEORETICAL = "#0b0b0b"
CEILING_MEASURED = "#e1e11a"

# Fixed series styling. Assigned by identity and never cycled, so adding or
# removing a kernel from a sweep cannot repaint the others.
SERIES_STYLE: dict[str, tuple[str, str]] = {
    "gemm_naive": (_RED, "o"),
    "gemm_tiled": (_YELLOW, "o"),
    "gemm_register_blocked": (_PINK, "o"),
    "gemm_vectorized": (_BROWN, "o"),
    "gemm_cublas": (_NAVY, "o"),
    "saxpy": (_ORANGE, "s"),
    "reduction": (_AQUA, "^"),
    "gemv": (_VIOLET, "D"),
    "transpose": (_MAGENTA, "v"),
}

# Anything unrecognised is drawn in muted ink rather than inventing a new hue.
_FALLBACK_STYLE = (INK_MUTED, "X")

# Where each family's single direct label sits relative to its point, in points.
# The memory bound families cluster tightly at the left of the axis and their
# labels would otherwise sit on top of each other, so they are fanned out. The
# cuBLAS label is pushed left because its point is near the right edge and the
# text would run off the figure.
SERIES_LABEL_OFFSET: dict[str, tuple[int, int]] = {
    "saxpy": (-10, -20),
    "reduction": (-10, 10),
    "gemv": (12, 10),
    "gemm_cublas": (-14, 14),
    "transpose": (12, -20),
}
_DEFAULT_LABEL_OFFSET = (9, 5)

# Families whose label is right aligned. The three memory bound families sit on
# top of one another at the left of the axis, so two of them put their text to
# the left of the marker and one to the right, which is what stops the boxes
# from landing on each other or on a neighbouring marker.
_RIGHT_ALIGNED = {"gemm_cublas", "saxpy", "reduction"}

# Human readable series names for the legend.
SERIES_LABEL: dict[str, str] = {
    "gemm_naive": "GEMM naive",
    "gemm_tiled": "GEMM tiled",
    "gemm_register_blocked": "GEMM register blocked",
    "gemm_vectorized": "GEMM vectorized",
    "gemm_cublas": "cuBLAS SGEMM (library)",
    "saxpy": "SAXPY",
    "reduction": "reduction",
    "gemv": "GEMV",
    "transpose": "transpose",
}


@dataclass(frozen=True)
class RooflinePoint:
    """One kernel configuration placed on the roofline.

    Attributes:
        series: Identity key used for colour and marker, for example
            "gemm_tiled". Styling is looked up by this, never by position.
        label: Short human label for a direct annotation, for example "4096".
        arithmetic_intensity: FLOP/byte for this run.
        achieved_gflops: Measured throughput in GFLOP/s.
        intensity_source: Whether the intensity used measured or theoretical
            bytes. Recorded per point so the caption can state honestly which
            byte count produced the position.
        annotate: Whether this point carries a direct label. Only a few do; a
            number on every point is unreadable and hides the shape of the data.
    """

    series: str
    label: str
    arithmetic_intensity: float
    achieved_gflops: float
    intensity_source: IntensitySource = IntensitySource.THEORETICAL
    annotate: bool = field(default=False)


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
    then flat compute bound), so the theoretical and measured ceilings sit on the
    same axes for honest comparison.
    """
    if not ceilings:
        raise ValueError("at least one Ceilings is required to draw a roofline")

    lo, hi = _intensity_axis_range(points, ceilings, ai_range)
    ai_grid = np.logspace(np.log10(lo), np.log10(hi), num=256)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    _draw_ceilings(ax, ceilings, ai_grid, lo, hi)
    _draw_points(ax, points)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)", color=INK_SECONDARY)
    ax.set_ylabel("performance (GFLOP/s)", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12)

    # Recessive grid and axes: the data should be the most visible thing here.
    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.grid(True, which="minor", color=GRIDLINE, linewidth=0.3, alpha=0.6,
            zorder=0)
    ax.tick_params(colors=INK_MUTED, which="both", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)

    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95,
              facecolor=SURFACE, edgecolor=AXIS, labelcolor=INK_SECONDARY)
    fig.tight_layout()

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, facecolor=SURFACE)
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_pdf


def _draw_ceilings(
    ax: plt.Axes,
    ceilings: list[Ceilings],
    ai_grid: np.ndarray,
    lo: float,
    hi: float,
) -> None:
    # Theoretical is black and solid, measured is lemon and dashed. The lemon
    # line is drawn thicker because it is pale against the surface, and it is
    # given a dark outline so it stays visible where it overlaps the black one.
    styles = [
        ("-", CEILING_THEORETICAL, 2.0),
        ("--", CEILING_MEASURED, 3.0),
    ]
    for index, ceiling in enumerate(ceilings):
        style, colour, width = styles[index % len(styles)]
        attainable = np.array([ceiling.attainable(ai) for ai in ai_grid]) / GIGA
        path_effects = None
        if colour == CEILING_MEASURED:
            path_effects = [
                pe.Stroke(linewidth=width + 1.2, foreground=INK_SECONDARY),
                pe.Normal(),
            ]
        ax.plot(
            ai_grid,
            attainable,
            linewidth=width,
            linestyle=style,
            color=colour,
            label=f"{ceiling.label} ceiling",
            zorder=2,
            path_effects=path_effects,
        )
        ridge = ceiling.ridge_point
        if lo <= ridge <= hi:
            ax.axvline(ridge, color=AXIS, linewidth=0.8, linestyle=":", zorder=1)


def _draw_points(ax: plt.Axes, points: list[RooflinePoint]) -> None:
    # Group by series so each contributes exactly one legend entry, and so the
    # drawing order is stable rather than dependent on row order.
    by_series: dict[str, list[RooflinePoint]] = {}
    for point in points:
        by_series.setdefault(point.series, []).append(point)

    for series in [s for s in SERIES_STYLE if s in by_series] + [
        s for s in by_series if s not in SERIES_STYLE
    ]:
        group = by_series[series]
        colour, marker = SERIES_STYLE.get(series, _FALLBACK_STYLE)
        ax.scatter(
            [p.arithmetic_intensity for p in group],
            [p.achieved_gflops for p in group],
            marker=marker,
            s=58,
            color=colour,
            # A surface coloured ring keeps overlapping marks readable.
            edgecolors=SURFACE,
            linewidths=1.0,
            zorder=3,
            label=SERIES_LABEL.get(series, series),
        )
        for point in group:
            if point.annotate:
                # Only one point per kernel family is labelled, and the label
                # names the largest problem size that family was run at, with
                # its unit. A label on every point produced the same text four
                # or five times over and hid the shape of the data.
                offset = SERIES_LABEL_OFFSET.get(series, _DEFAULT_LABEL_OFFSET)
                ax.annotate(
                    point.label,
                    (point.arithmetic_intensity, point.achieved_gflops),
                    textcoords="offset points",
                    xytext=offset,
                    fontsize=7.5,
                    color=INK_PRIMARY,
                    horizontalalignment=(
                        "right" if series in _RIGHT_ALIGNED else "left"
                    ),
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": SURFACE,
                        "edgecolor": AXIS,
                        "linewidth": 0.5,
                        "alpha": 0.9,
                    },
                    # A leader line, because SAXPY and the reduction both top
                    # out at the same element count, so two labels legitimately
                    # read the same and the line is what says which is which.
                    arrowprops={
                        "arrowstyle": "-",
                        "color": INK_MUTED,
                        "linewidth": 0.6,
                        "shrinkA": 1,
                        "shrinkB": 3,
                    },
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
