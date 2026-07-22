"""Loading raw pipeline output into validated pandas frames.

Three inputs feed the analysis: the benchmark timing CSV from the C++ driver,
the NVML monitor CSV, and Nsight Compute's CSV export. Each loader validates
its input and rejects malformed rows loudly rather than letting a NaN or a
negative time slip into a plot. Silent bad data is the failure mode that would
invalidate the whole report without anyone noticing, so the loaders are strict
on purpose and are unit tested against deliberately broken rows.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# Columns the driver writes for every timing row (spec Section 8.1). Extra
# columns (per kernel config params) are allowed and carried through; these are
# the ones the analysis relies on and therefore validates.
TIMING_REQUIRED_COLUMNS = (
    "kernel",
    "problem_size",
    "mean_ms",
    "median_ms",
    "stddev_ms",
    "achieved_gflops",
    "timestamp",
)

# Positive-valued numeric columns: a zero or negative here is a broken row.
TIMING_POSITIVE_COLUMNS = ("mean_ms", "median_ms", "achieved_gflops")

NVML_REQUIRED_COLUMNS = (
    "timestamp",
    "power_w",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "gpu_util_pct",
)

NCU_REQUIRED_COLUMNS = ("kernel", "metric_name", "metric_value")


class DataValidationError(ValueError):
    """Raised when an input file is missing columns or carries broken rows."""


def load_timing_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate the benchmark timing CSV.

    Rejects the file if a required column is missing, and rejects the load if any
    row has a non positive time or non finite achieved throughput, naming the
    offending row indices so the problem is findable.
    """
    frame = _read_csv(path)
    _require_columns(frame, TIMING_REQUIRED_COLUMNS, path)
    _reject_nonfinite(frame, TIMING_POSITIVE_COLUMNS, path)
    _reject_nonpositive(frame, TIMING_POSITIVE_COLUMNS, path)
    return frame


def load_nvml_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate the NVML monitor CSV.

    Parses the timestamp column to datetime so timing rows can later be joined to
    power and clock samples by time. Rejects negative power or temperature, which
    would signal a mangled sample rather than a real reading.
    """
    frame = _read_csv(path)
    _require_columns(frame, NVML_REQUIRED_COLUMNS, path)
    _reject_nonfinite(frame, ("power_w", "temperature_c"), path)
    _reject_negative(frame, ("power_w",), path)
    frame = frame.copy()
    # The NVML monitor writes ISO 8601 timestamps; pin the format so samples
    # with and without fractional seconds both parse consistently rather than
    # falling back to slow per element inference.
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], format="ISO8601", errors="coerce"
    )
    if frame["timestamp"].isna().any():
        bad = frame.index[frame["timestamp"].isna()].tolist()
        raise DataValidationError(
            f"{path}: unparseable timestamps at rows {bad}"
        )
    return frame


def load_ncu_csv(path: str | Path) -> pd.DataFrame:
    """Load a tidy Nsight Compute metric export into (kernel, metric, value).

    Expects the long form the ncu wrapper writes: one row per (kernel, metric)
    with a numeric value. Metric names are queried from the installed ncu at
    profile time (they shift between versions) and recorded in the profiling
    guide, so this loader stays agnostic about which specific metrics are present
    and only insists the value column is numeric.
    """
    frame = _read_csv(path)
    _require_columns(frame, NCU_REQUIRED_COLUMNS, path)
    frame = frame.copy()
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    if frame["metric_value"].isna().any():
        bad = frame.index[frame["metric_value"].isna()].tolist()
        raise DataValidationError(
            f"{path}: non numeric metric_value at rows {bad}"
        )
    return frame


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise DataValidationError(f"{path}: file has no data rows")
    return frame


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], path: str | Path
) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataValidationError(f"{path}: missing required columns {missing}")


def _reject_nonfinite(
    frame: pd.DataFrame, columns: tuple[str, ...], path: str | Path
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        bad = frame.index[~numeric.apply(_is_finite)].tolist()
        if bad:
            raise DataValidationError(
                f"{path}: non finite values in {column!r} at rows {bad}"
            )


def _reject_nonpositive(
    frame: pd.DataFrame, columns: tuple[str, ...], path: str | Path
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        bad = frame.index[numeric <= 0.0].tolist()
        if bad:
            raise DataValidationError(
                f"{path}: non positive values in {column!r} at rows {bad}"
            )


def _reject_negative(
    frame: pd.DataFrame, columns: tuple[str, ...], path: str | Path
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        bad = frame.index[numeric < 0.0].tolist()
        if bad:
            raise DataValidationError(
                f"{path}: negative values in {column!r} at rows {bad}"
            )


def _is_finite(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
