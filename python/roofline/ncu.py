"""Parsing Nsight Compute output into something the analysis can join.

The wrapper script writes one CSV per profiled cell, and each file has the
driver's own stdout in front of the CSV header, because ncu forwards the child
process output. So the parser finds the header line rather than assuming the
file starts with one.

Two conversions happen here. Raw ncu rows become a tidy (cell, metric, value)
frame, and that frame is folded into a per cell summary with the derived
quantities the report actually talks about: total DRAM bytes, achieved DRAM
bandwidth, and sectors per request for loads and stores.

On sectors per request. A memory request from a warp is split into 32 byte
sectors, so the ideal ratio is not a constant: it is
(32 threads * bytes per thread) / 32. For a 4 byte scalar access that is 4, and
for a 16 byte float4 access it is 16. Reporting the ratio without its ideal
would make a correctly vectorized kernel look four times worse than a scalar
one, so the ideal is carried alongside.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from roofline.loaders import DataValidationError

# The first field of the ncu CSV header, used to find where the CSV starts.
_HEADER_PREFIX = '"ID","Process ID"'

_BYTES_PER_SECTOR = 32
_THREADS_PER_WARP = 32


def parse_ncu_csv(path: str | Path) -> pd.DataFrame:
    """Parse one ncu CSV into tidy (kernel, metric_name, metric_value) rows.

    Skips the driver stdout that precedes the header, strips thousands
    separators, and rejects a file whose metrics all came back as "n/a", which is
    what an unsupported metric name or a missing privilege produces.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    start = text.find(_HEADER_PREFIX)
    if start < 0:
        raise DataValidationError(f"{path}: no ncu CSV header found")

    frame = pd.read_csv(io.StringIO(text[start:]))
    required = {"Kernel Name", "Metric Name", "Metric Value"}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"{path}: missing columns {sorted(missing)}")

    tidy = pd.DataFrame(
        {
            "kernel": frame["Kernel Name"].astype(str),
            "metric_name": frame["Metric Name"].astype(str),
            "raw_value": frame["Metric Value"].astype(str),
        }
    )
    # ncu writes thousands separators, and "n/a" for anything it could not
    # collect. The latter becomes NaN so it can be counted and reported rather
    # than silently treated as zero.
    tidy["metric_value"] = pd.to_numeric(
        tidy["raw_value"].str.replace(",", "", regex=False), errors="coerce"
    )
    if tidy["metric_value"].notna().sum() == 0:
        raise DataValidationError(
            f"{path}: every metric is n/a. Either the metric names are not "
            f"supported on this device or ncu lacked the privileges to read "
            f"counters."
        )
    return tidy.drop(columns=["raw_value"])


def parse_ncu_directory(directory: str | Path) -> pd.DataFrame:
    """Parse every ncu CSV in a directory, tagged by cell name (the file stem)."""
    directory = Path(directory)
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        try:
            tidy = parse_ncu_csv(path)
        except DataValidationError:
            # A cell that collected nothing is skipped rather than poisoning the
            # whole load; the summary reports which cells are present.
            continue
        tidy.insert(0, "cell", path.stem)
        frames.append(tidy)
    if not frames:
        raise DataValidationError(f"{directory}: no usable ncu CSVs found")
    return pd.concat(frames, ignore_index=True)


def _is_intensive(metric_name: str) -> bool:
    """True for metrics that must be averaged rather than added across launches.

    A counter such as dram__bytes_op_read.sum is extensive: profiling three
    launches and adding them gives the total traffic, which is what we want. A
    percentage or a ratio is intensive: adding an L2 hit rate of 97 percent to
    another of 97 percent does not give 194 percent of anything. Getting this
    wrong produced a reported hit rate of 100.39 percent, which is how the bug
    was caught.
    """
    return metric_name.endswith((".pct", ".ratio")) or "pct_of_peak" in metric_name


def summarize_cells(tidy: pd.DataFrame) -> pd.DataFrame:
    """Fold tidy metric rows into one row per cell with derived quantities.

    Counters are summed across profiled launches; percentages and ratios are
    averaged. See :func:`_is_intensive`.
    """
    tidy = tidy.copy()
    tidy["intensive"] = tidy["metric_name"].map(_is_intensive)

    extensive = tidy[~tidy["intensive"]].pivot_table(
        index="cell", columns="metric_name", values="metric_value", aggfunc="sum"
    )
    intensive = tidy[tidy["intensive"]].pivot_table(
        index="cell", columns="metric_name", values="metric_value", aggfunc="mean"
    )
    wide = extensive.join(intensive, how="outer").reset_index()

    def col(name: str) -> pd.Series:
        if name in wide.columns:
            return wide[name].fillna(0.0)
        return pd.Series([0.0] * len(wide), index=wide.index)

    out = pd.DataFrame({"cell": wide["cell"]})
    out["dram_read_bytes"] = col("dram__bytes_op_read.sum")
    out["dram_write_bytes"] = col("dram__bytes_op_write.sum")
    out["measured_bytes"] = out["dram_read_bytes"] + out["dram_write_bytes"]

    duration_ns = col("gpu__time_duration.sum")
    out["duration_ms"] = duration_ns / 1.0e6
    # byte/ns is GB/s, so no further scaling is needed.
    out["dram_gbps"] = (out["measured_bytes"] / duration_ns).where(duration_ns > 0, 0.0)

    out["l2_hit_pct"] = col("lts__t_sector_hit_rate.pct")
    out["occupancy_pct"] = col("sm__warps_active.avg.pct_of_peak_sustained_active")
    out["sm_throughput_pct"] = col(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    )
    out["bank_conflicts"] = col("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
    out["warp_efficiency"] = col(
        "smsp__thread_inst_executed_per_inst_executed.ratio"
    )

    ld_sectors = col("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    ld_requests = col("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    st_sectors = col("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
    st_requests = col("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum")
    out["ld_sectors_per_request"] = (ld_sectors / ld_requests).where(
        ld_requests > 0, 0.0
    )
    out["st_sectors_per_request"] = (st_sectors / st_requests).where(
        st_requests > 0, 0.0
    )
    return out


def implausible_percentages(
    summary: pd.DataFrame, tolerance_pct: float = 0.5
) -> list[tuple[str, str, float]]:
    """Find reported percentages above 100, returned as (cell, column, value).

    Hardware counters are not arithmetic identities. This card reports an L2
    sector hit rate of 100.39 percent for the naive GEMM, which cannot be true:
    a hit rate is hits over accesses and is bounded by 1. Re-running gave 97.02
    for the same kernel, so it is measurement noise in how sector hits are
    attributed, not a parsing error.

    The value is kept verbatim rather than clamped, because silently rewriting a
    measurement to look plausible is exactly the kind of quiet dishonesty this
    project is meant to avoid. Instead it is surfaced here so the caller can flag
    it, and the report reads such a value as "essentially all hits" while saying
    what the counter actually said.
    """
    flagged: list[tuple[str, str, float]] = []
    percent_columns = [
        "l2_hit_pct",
        "occupancy_pct",
        "sm_throughput_pct",
    ]
    for _, row in summary.iterrows():
        for column in percent_columns:
            if column in summary.columns:
                value = float(row[column])
                if value > 100.0 + tolerance_pct:
                    flagged.append((str(row["cell"]), column, value))
    return flagged


def ideal_sectors_per_request(bytes_per_thread: int) -> float:
    """The sectors-per-request a perfectly coalesced warp would achieve.

    Comparing a measured ratio against this rather than against a fixed 4 is
    what keeps a float4 kernel from looking four times worse than a scalar one
    when it is in fact moving four times as much per instruction.
    """
    if bytes_per_thread <= 0:
        raise ValueError(f"bytes_per_thread must be positive, got {bytes_per_thread}")
    return (_THREADS_PER_WARP * bytes_per_thread) / _BYTES_PER_SECTOR
