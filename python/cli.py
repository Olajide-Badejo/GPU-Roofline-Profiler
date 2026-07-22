"""One entry point that regenerates every figure and table from raw results.

Run after any benchmark or profiling pass to rebuild the report inputs with no
hand editing:

    python cli.py --results ../results/sample_run --report ../report

It reads the timing CSV (and the Nsight Compute and NVML CSVs when present),
computes the ceilings, and writes the roofline figures into report/figures and
the booktabs fragments into report/tables. Every configuration becomes a labeled
point; intensity uses measured DRAM bytes where Nsight data exists and falls
back to theoretical byte counts with an explicit label otherwise.

This module stays runnable before any real data exists: with nothing to plot it
says so plainly and exits cleanly, rather than inventing points. No performance
number is ever synthesized here; unmeasured means unplotted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from roofline import loaders, tables
from roofline.model import Ceilings, resolve_intensity
from roofline.plotting import RooflinePoint, plot_roofline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="results directory holding timing.csv and optional ncu/nvml CSVs",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="report directory; figures/ and tables/ are written under it",
    )
    parser.add_argument(
        "--peaks",
        type=Path,
        default=None,
        help="optional peaks CSV with theoretical and measured ceilings",
    )
    return parser


def load_ceilings(peaks_path: Path | None) -> list[Ceilings]:
    """Load ceilings from the peaks CSV, or return an empty list if absent.

    The C++ peaks utility writes theoretical and measured ceilings on this
    machine. Until that file exists the CLI simply has no roof to draw and says
    so, rather than falling back to spec sheet numbers.
    """
    if peaks_path is None or not peaks_path.exists():
        return []
    import pandas as pd

    frame = pd.read_csv(peaks_path)
    ceilings: list[Ceilings] = []
    for _, row in frame.iterrows():
        ceilings.append(
            Ceilings(
                peak_flops=float(row["peak_flops"]),
                peak_bandwidth=float(row["peak_bandwidth"]),
                label=str(row["label"]),
            )
        )
    return ceilings


def points_from_timing(results_dir: Path) -> list[RooflinePoint]:
    """Turn each timing row into a labeled roofline point.

    Uses measured bytes and flops columns when the driver and Nsight passes have
    populated them; otherwise the point carries a theoretical intensity label.
    """
    timing_path = results_dir / "timing.csv"
    if not timing_path.exists():
        return []
    frame = loaders.load_timing_csv(timing_path)

    # Only the largest configuration of each series carries a direct label. A
    # label on every point turns the figure into a wall of overlapping text and
    # hides the very shape it is meant to show.
    largest: dict[str, float] = {}
    for _, row in frame.iterrows():
        series = _series_key(row)
        size = float(row.get("problem_size", 0.0) or 0.0)
        largest[series] = max(largest.get(series, 0.0), size)

    points: list[RooflinePoint] = []
    for _, row in frame.iterrows():
        flops = float(row.get("flops", 0.0) or 0.0)
        theo_bytes = float(row.get("theoretical_bytes", 0.0) or 0.0)
        meas_bytes = row.get("measured_bytes", None)
        meas_bytes = None if meas_bytes is None or meas_bytes != meas_bytes else float(
            meas_bytes
        )
        if flops <= 0.0 or theo_bytes <= 0.0:
            # A transpose does no floating point work, so it has no place on an
            # arithmetic intensity axis at all. It is reported on the bandwidth
            # figure instead rather than being forced onto this one.
            continue
        ai, source = resolve_intensity(flops, theo_bytes, meas_bytes)
        series = _series_key(row)
        size = float(row.get("problem_size", 0.0) or 0.0)
        points.append(
            RooflinePoint(
                series=series,
                label=str(row["problem_size"]),
                arithmetic_intensity=ai,
                achieved_gflops=float(row["achieved_gflops"]),
                intensity_source=source,
                annotate=(size == largest.get(series)),
            )
        )
    return points


def _series_key(row: object) -> str:
    """Identity used for colour and marker: kernel, plus variant for GEMM.

    The GEMM variants are the whole point of the ladder, so they are separate
    series; every other kernel has a single implementation and does not need the
    suffix.
    """
    kernel = str(row["kernel"])  # type: ignore[index]
    variant = str(row.get("variant", "") or "")  # type: ignore[union-attr]
    if kernel == "gemm" and variant:
        return f"gemm_{variant}"
    return kernel


def regenerate(results_dir: Path, report_dir: Path, peaks_path: Path | None) -> int:
    """Regenerate all figures and tables. Returns a process exit code."""
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    ceilings = load_ceilings(peaks_path)
    points = points_from_timing(results_dir)

    if not ceilings or not points:
        print(
            "nothing to plot yet: "
            f"{len(ceilings)} ceiling(s), {len(points)} point(s). "
            "Run the benchmark and peaks passes first."
        )
        # Still (re)write any tables we can, then exit cleanly.
        _write_timing_table(results_dir, tables_dir)
        return 0

    steps = ["main roofline", "timing table"]
    for _ in tqdm(steps, desc="regenerating report inputs", unit="artifact"):
        pass
    plot_roofline(
        points,
        ceilings,
        figures_dir / "roofline_main.pdf",
        figures_dir / "roofline_main.png",
        title="RTX 5070 roofline",
    )
    _write_timing_table(results_dir, tables_dir)
    print(f"wrote figures to {figures_dir} and tables to {tables_dir}")
    return 0


def _write_timing_table(results_dir: Path, tables_dir: Path) -> None:
    timing_path = results_dir / "timing.csv"
    if not timing_path.exists():
        return
    frame = loaders.load_timing_csv(timing_path)
    columns = [
        c
        for c in ("kernel", "problem_size", "mean_ms", "stddev_ms", "achieved_gflops")
        if c in frame.columns
    ]
    tables.write_table(
        frame[columns],
        tables_dir / "timing_summary.tex",
        float_format="{:.3f}",
        headers={
            "kernel": "kernel",
            "problem_size": "size",
            "mean_ms": "mean (ms)",
            "stddev_ms": "sd (ms)",
            "achieved_gflops": "GFLOP/s",
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return regenerate(args.results, args.report, args.peaks)


if __name__ == "__main__":
    sys.exit(main())
