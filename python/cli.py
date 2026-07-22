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
    parser.add_argument(
        "--ncu",
        type=Path,
        default=None,
        help="optional Nsight Compute output directory, for measured DRAM bytes",
    )
    return parser


# Maps an ncu cell name onto the (kernel, variant) it profiled, so measured
# bytes can be attached to the right timing rows. The ncu subset deliberately
# profiles one size per cell, so the join is on identity rather than on size.
NCU_CELL_TO_SERIES = {
    "saxpy": ("saxpy", ""),
    "transpose_naive": ("transpose", "naive"),
    "transpose_tiled": ("transpose", "tiled"),
    "gemm_naive": ("gemm", "naive"),
    "gemm_tiled": ("gemm", "tiled"),
    "gemm_register_blocked": ("gemm", "register_blocked"),
    "gemm_vectorized": ("gemm", "vectorized"),
}


def load_measured_bytes(ncu_dir: Path | None) -> dict[tuple[str, str], float]:
    """Return measured DRAM bytes keyed by (kernel, variant).

    Absent or unreadable Nsight output is not an error: the analysis simply falls
    back to theoretical byte counts and labels every affected point as such.
    """
    if ncu_dir is None or not ncu_dir.exists():
        return {}
    from roofline import ncu as ncu_mod
    from roofline.loaders import DataValidationError

    try:
        summary = ncu_mod.summarize_cells(ncu_mod.parse_ncu_directory(ncu_dir))
    except (DataValidationError, FileNotFoundError) as exc:
        print(f"note: no usable Nsight Compute data ({exc}); using theoretical bytes")
        return {}

    measured: dict[tuple[str, str], float] = {}
    for _, row in summary.iterrows():
        key = NCU_CELL_TO_SERIES.get(str(row["cell"]))
        if key is not None and float(row["measured_bytes"]) > 0.0:
            measured[key] = float(row["measured_bytes"])
    return measured


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


def points_from_timing(
    results_dir: Path, measured: dict[tuple[str, str], float] | None = None
) -> list[RooflinePoint]:
    """Turn each timing row into a labeled roofline point.

    Uses measured bytes and flops columns when the driver and Nsight passes have
    populated them; otherwise the point carries a theoretical intensity label.
    """
    timing_path = results_dir / "timing.csv"
    if not timing_path.exists():
        return []
    frame = loaders.load_timing_csv(timing_path)

    # Exactly one point per kernel FAMILY carries a direct label, not one per
    # series. Labelling every series meant the five GEMM variants all printed
    # the same "4096", and SAXPY, reduction, and GEMV all printed the same
    # "67108864", because those families happen to top out at the same size. The
    # legend already says which series is which; the label only needs to say how
    # big the largest run was, once, per family.
    #
    # The labelled point is the fastest of the family's largest configurations,
    # so the text sits at the top of its cluster where there is room for it.
    best: dict[str, tuple[float, float, int]] = {}
    for index, row in frame.iterrows():
        family = str(row["kernel"])
        size = float(row.get("problem_size", 0.0) or 0.0)
        gflops = float(row.get("achieved_gflops", 0.0) or 0.0)
        current = best.get(family)
        if current is None or (size, gflops) > (current[0], current[1]):
            best[family] = (size, gflops, int(index))
    labelled_rows = {entry[2] for entry in best.values()}

    points: list[RooflinePoint] = []
    for index, row in frame.iterrows():
        flops = float(row.get("flops", 0.0) or 0.0)
        theo_bytes = float(row.get("theoretical_bytes", 0.0) or 0.0)
        # Measured DRAM bytes come from the Nsight pass, which profiles one size
        # per kernel. They are applied only to the row whose size was actually
        # profiled, so no point ever claims a measurement taken at another size.
        meas_bytes = None
        if measured:
            key = (str(row["kernel"]), str(row.get("variant", "") or ""))
            if key not in measured:
                key = (str(row["kernel"]), "")
            candidate = measured.get(key)
            if candidate is not None and float(
                row.get("problem_size", 0.0) or 0.0
            ) == _profiled_size(str(row["kernel"])):
                meas_bytes = candidate
        if flops <= 0.0 or theo_bytes <= 0.0:
            # A transpose does no floating point work, so it has no place on an
            # arithmetic intensity axis at all. It is reported on the bandwidth
            # figure instead rather than being forced onto this one.
            continue
        ai, source = resolve_intensity(flops, theo_bytes, meas_bytes)
        series = _series_key(row)
        points.append(
            RooflinePoint(
                series=series,
                label=_size_label(row),
                arithmetic_intensity=ai,
                achieved_gflops=float(row["achieved_gflops"]),
                intensity_source=source,
                annotate=(int(index) in labelled_rows),
            )
        )
    return points


def _size_label(row: object) -> str:
    """Human readable problem size, with its unit.

    The raw figure is a count, not a rate: 67108864 is 2^26 elements, which is
    easy to misread as a throughput when it appears bare on a performance plot.
    Vector kernels are labelled in elements and matrix kernels by their
    dimensions, so the number can only be read as what it is.
    """
    kernel = str(row["kernel"])  # type: ignore[index]
    if kernel in {"gemm", "transpose"}:
        m = int(row.get("m", 0) or 0)  # type: ignore[union-attr]
        n = int(row.get("n", 0) or 0)  # type: ignore[union-attr]
        if m > 0 and n > 0:
            return f"{m} x {n}"
        size = int(float(row.get("problem_size", 0) or 0))  # type: ignore[union-attr]
        return f"{size} x {size}"
    if kernel == "gemv":
        m = int(row.get("m", 0) or 0)  # type: ignore[union-attr]
        n = int(row.get("n", 0) or 0)  # type: ignore[union-attr]
        return f"{m} x {n} matrix"
    count = float(row.get("problem_size", 0.0) or 0.0)  # type: ignore[union-attr]
    return f"{_compact_count(count)} elements"


def _compact_count(value: float) -> str:
    """Render an element count compactly.

    These sweeps use powers of two, so 67108864 is exactly 64 mebi. Binary units
    render that as a round "64Mi" instead of the misleading "67.1089M" that
    decimal units produce, and they are the honest unit for a power of two size.
    Sizes that are not whole binary multiples fall back to decimal with three
    significant figures.
    """
    count = int(value)
    for unit, scale in (("Gi", 1 << 30), ("Mi", 1 << 20), ("ki", 1 << 10)):
        if count >= scale and count % scale == 0:
            return f"{count // scale}{unit}"
    if value >= 1.0e6:
        return f"{value / 1.0e6:.3g}M"
    if value >= 1.0e3:
        return f"{value / 1.0e3:.3g}k"
    return f"{count}"


# The size each kernel was profiled at, matching configs/sweep.yaml's ncu_subset
# and the reduced sweep the ncu wrapper generates.
_PROFILED_SIZES = {"saxpy": 16777216.0, "transpose": 2048.0, "gemm": 2048.0}


def _profiled_size(kernel: str) -> float:
    """Problem size the Nsight pass profiled for this kernel, or NaN if none."""
    return _PROFILED_SIZES.get(kernel, float("nan"))


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


def regenerate(
    results_dir: Path,
    report_dir: Path,
    peaks_path: Path | None,
    ncu_dir: Path | None = None,
) -> int:
    """Regenerate all figures and tables. Returns a process exit code."""
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    ceilings = load_ceilings(peaks_path)
    measured = load_measured_bytes(ncu_dir)
    if measured:
        print(f"using measured DRAM bytes for {len(measured)} profiled kernels")
    points = points_from_timing(results_dir, measured)

    if not ceilings or not points:
        print(
            "nothing to plot yet: "
            f"{len(ceilings)} ceiling(s), {len(points)} point(s). "
            "Run the benchmark and peaks passes first."
        )
        # Still (re)write any tables we can, then exit cleanly.
        _write_timing_table(results_dir, tables_dir)
        _write_counter_table(ncu_dir, tables_dir)
        return 0

    steps = ["main roofline", "timing table"]
    for _ in tqdm(steps, desc="regenerating report inputs", unit="artifact"):
        pass
    plot_roofline(
        points,
        ceilings,
        figures_dir / "roofline_main.pdf",
        figures_dir / "roofline_main.png",
        title="RTX 5070 roofline plot",
    )
    _write_timing_table(results_dir, tables_dir)
    _write_counter_table(ncu_dir, tables_dir)
    _write_ceilings_table(ceilings, tables_dir)
    _write_environment_table(results_dir, tables_dir)
    _write_appendix_tables(results_dir, ncu_dir, tables_dir)
    print(f"wrote figures to {figures_dir} and tables to {tables_dir}")
    return 0


def _write_environment_table(results_dir: Path, tables_dir: Path) -> None:
    """Write the machine environment table from the run manifest.

    Generated at report build time from what the run actually recorded, never
    typed from memory, so the appendix always describes the machine that produced
    the numbers in the document.
    """
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return
    import json

    import pandas as pd

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = manifest.get("device", {})
    cuda = manifest.get("cuda", {})
    total_lanes = int(device.get("sm_count", 0)) * int(
        device.get("fp32_lanes_per_sm", 0)
    )

    rows = [
        ("GPU", str(device.get("name", "unknown"))),
        ("compute capability", str(device.get("compute_capability", ""))),
        ("SMs", f"{device.get('sm_count', 0)}"),
        ("FP32 lanes per SM", f"{device.get('fp32_lanes_per_sm', 0)}"),
        ("total FP32 lanes", f"{total_lanes}"),
        ("SM clock (GHz)", f"{float(device.get('sm_clock_hz', 0)) / 1e9:.3f}"),
        (
            "memory clock (GHz)",
            f"{float(device.get('memory_clock_hz', 0)) / 1e9:.3f}",
        ),
        ("memory bus (bits)", f"{device.get('memory_bus_width_bits', 0)}"),
        (
            "total VRAM (MiB)",
            f"{int(device.get('total_global_mem', 0)) // (1024 * 1024)}",
        ),
        ("L2 cache (MiB)", f"{int(device.get('l2_cache_bytes', 0)) // (1024 * 1024)}"),
        ("CUDA driver version", str(cuda.get("driver_version", ""))),
        ("CUDA runtime version", str(cuda.get("runtime_version", ""))),
        ("run timestamp (UTC)", str(manifest.get("timestamp_utc", ""))),
    ]
    frame = pd.DataFrame(rows, columns=["property", "value"])
    tables.write_table(
        frame,
        tables_dir / "environment_summary.tex",
        headers={"property": "property", "value": "value"},
    )


def _write_ceilings_table(ceilings: list[Ceilings], tables_dir: Path) -> None:
    """Write the ceilings table, in the friendlier units the report reads in."""
    if not ceilings:
        return
    import pandas as pd

    frame = pd.DataFrame(
        {
            "ceiling": [c.label for c in ceilings],
            "compute_tflops": [c.peak_flops / 1.0e12 for c in ceilings],
            "bandwidth_gbps": [c.peak_bandwidth / 1.0e9 for c in ceilings],
            "ridge_flop_per_byte": [c.ridge_point for c in ceilings],
        }
    )
    tables.write_table(
        frame,
        tables_dir / "ceilings.tex",
        float_format="{:.2f}",
        headers={
            "ceiling": "ceiling",
            "compute_tflops": "compute (TFLOP/s)",
            "bandwidth_gbps": "bandwidth (GB/s)",
            "ridge_flop_per_byte": "ridge (FLOP/byte)",
        },
    )


def _write_counter_table(ncu_dir: Path | None, tables_dir: Path) -> None:
    """Write the Nsight Compute counter table the discussion section cites."""
    if ncu_dir is None or not ncu_dir.exists():
        return
    from roofline import ncu as ncu_mod
    from roofline.loaders import DataValidationError

    try:
        summary = ncu_mod.summarize_cells(ncu_mod.parse_ncu_directory(ncu_dir))
    except (DataValidationError, FileNotFoundError):
        return

    # A percentage above 100 cannot be real. Report it as measured, but say so
    # rather than letting the report assert it with a straight face.
    for cell, column, value in ncu_mod.implausible_percentages(summary):
        print(
            f"note: {cell} reports {column} = {value:.2f}, which exceeds 100 "
            f"and is a counter artifact; reported as measured"
        )

    table = summary[
        [
            "cell",
            "l2_hit_pct",
            "occupancy_pct",
            "bank_conflicts",
            "ld_sectors_per_request",
            "st_sectors_per_request",
            "dram_gbps",
        ]
    ].copy()
    # Bank conflict counts run to hundreds of millions. Printed in full they
    # push the table off the right of the page, so they are shown in millions,
    # which is also the only precision anyone reads them at.
    table["bank_conflicts"] = table["bank_conflicts"] / 1.0e6
    # Headers are kept short for the same reason: this table has seven columns
    # and the page is only so wide.
    tables.write_table(
        table,
        tables_dir / "ncu_counters.tex",
        float_format="{:.2f}",
        # Headers are plain text: the writer escapes LaTeX specials itself, so
        # pre-escaping a percent sign here would put a literal backslash in the
        # rendered table.
        headers={
            "cell": "kernel",
            "l2_hit_pct": "L2 hit %",
            "occupancy_pct": "occupancy %",
            "bank_conflicts": "conflicts (M)",
            "ld_sectors_per_request": "ld sec/req",
            "st_sectors_per_request": "st sec/req",
            "dram_gbps": "DRAM GB/s",
        },
    )


def _write_appendix_tables(
    results_dir: Path, ncu_dir: Path | None, tables_dir: Path
) -> None:
    """Write the three long appendix tables, all page breaking longtables.

    These are the raw records behind the summaries in the body: every timing row
    rather than a selection, and every Nsight metric rather than the derived
    handful. They are long by nature, so a plain tabular would run off the page.
    """
    import pandas as pd

    # A.1 full timing table: every row the sweep produced.
    timing_path = results_dir / "timing.csv"
    if timing_path.exists():
        frame = loaders.load_timing_csv(timing_path)
        columns = [
            c
            for c in (
                "kernel",
                "variant",
                "problem_size",
                "block_size",
                "tile_dim",
                "mean_ms",
                "stddev_ms",
                "achieved_gflops",
                "achieved_gbps",
            )
            if c in frame.columns
        ]
        tables.write_longtable(
            frame[columns],
            tables_dir / "timing_full.tex",
            float_format="{:.3f}",
            headers={
                "kernel": "kernel",
                "variant": "variant",
                "problem_size": "size",
                "block_size": "block",
                "tile_dim": "tile",
                "mean_ms": "mean (ms)",
                "stddev_ms": "sd (ms)",
                "achieved_gflops": "GFLOP/s",
                "achieved_gbps": "GB/s",
            },
        )

    # A: full environment, from the manifest plus the recorded peaks.
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows: list[tuple[str, str]] = []

        def flatten(prefix: str, node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    flatten(f"{prefix}.{key}" if prefix else str(key), value)
            else:
                rows.append((prefix, str(node)))

        flatten("", manifest)
        tables.write_longtable(
            pd.DataFrame(rows, columns=["field", "value"]),
            tables_dir / "environment_full.tex",
            column_format="ll",
            headers={"field": "field", "value": "value"},
        )

    # A.2 full Nsight Compute metrics: every metric for every profiled cell.
    if ncu_dir is not None and ncu_dir.exists():
        from roofline import ncu as ncu_mod
        from roofline.loaders import DataValidationError

        try:
            tidy = ncu_mod.parse_ncu_directory(ncu_dir)
        except (DataValidationError, FileNotFoundError):
            return
        tidy = tidy[["cell", "metric_name", "metric_value"]].sort_values(
            ["cell", "metric_name"]
        )
        tables.write_longtable(
            tidy,
            tables_dir / "ncu_full.tex",
            column_format="llr",
            float_format="{:.4g}",
            headers={
                "cell": "kernel",
                "metric_name": "metric",
                "metric_value": "value",
            },
        )


def _write_timing_table(results_dir: Path, tables_dir: Path) -> None:
    timing_path = results_dir / "timing.csv"
    if not timing_path.exists():
        return
    frame = loaders.load_timing_csv(timing_path)
    # The variant column is not optional here. Without it the five GEMM rows at
    # each size are indistinguishable, and the whole point of the ladder is
    # telling those five apart. The bandwidth column is included because the
    # transpose rows do zero floating point work and would otherwise be a column
    # of zeroes with nothing to say.
    columns = [
        c
        for c in (
            "kernel",
            "variant",
            "problem_size",
            "mean_ms",
            "stddev_ms",
            "achieved_gflops",
            "achieved_gbps",
        )
        if c in frame.columns
    ]
    # 62 rows does not fit on one page, and a plain tabular would simply run off
    # the bottom and be clipped, so this one breaks across pages with a repeated
    # header.
    tables.write_longtable(
        frame[columns],
        tables_dir / "timing_summary.tex",
        float_format="{:.3f}",
        headers={
            "kernel": "kernel",
            "variant": "variant",
            "problem_size": "size",
            "mean_ms": "mean (ms)",
            "stddev_ms": "sd (ms)",
            "achieved_gflops": "GFLOP/s",
            "achieved_gbps": "GB/s",
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return regenerate(args.results, args.report, args.peaks, args.ncu)


if __name__ == "__main__":
    sys.exit(main())
