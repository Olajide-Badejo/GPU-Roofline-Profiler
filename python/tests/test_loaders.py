"""Unit tests for the loaders, focused on rejecting malformed rows loudly.

The loaders are the last line of defence before bad data reaches a plot, so the
tests deliberately feed them broken input and assert they raise rather than
return quietly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roofline import loaders
from roofline.loaders import DataValidationError

GOOD_TIMING = """\
kernel,problem_size,mean_ms,median_ms,stddev_ms,achieved_gflops,timestamp
saxpy,1048576,0.12,0.11,0.01,120.5,2026-07-22T10:00:00Z
gemm_tiled,4096,3.40,3.38,0.05,9800.0,2026-07-22T10:00:01Z
"""

GOOD_NVML = """\
timestamp,power_w,temperature_c,sm_clock_mhz,memory_clock_mhz,gpu_util_pct
2026-07-22T10:00:00Z,120.0,52.0,2500,14000,98
2026-07-22T10:00:00.1Z,122.0,52.5,2505,14000,99
"""

GOOD_NCU = """kernel,metric_name,metric_value
gemm_tiled,dram__bytes_read.sum,1234567
gemm_tiled,sm__warps_active.avg.pct_of_peak_sustained_active,71.2
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_timing_csv_ok(tmp_path):
    frame = loaders.load_timing_csv(_write(tmp_path, "t.csv", GOOD_TIMING))
    assert len(frame) == 2
    assert set(loaders.TIMING_REQUIRED_COLUMNS).issubset(frame.columns)


def test_load_timing_missing_column(tmp_path):
    bad = GOOD_TIMING.replace(",achieved_gflops", ",wrong_name")
    with pytest.raises(DataValidationError, match="missing required columns"):
        loaders.load_timing_csv(_write(tmp_path, "t.csv", bad))


def test_load_timing_rejects_negative_time(tmp_path):
    bad = GOOD_TIMING.replace("0.12,0.11", "-0.12,0.11")
    with pytest.raises(DataValidationError, match="non positive"):
        loaders.load_timing_csv(_write(tmp_path, "t.csv", bad))


def test_load_timing_rejects_nan_throughput(tmp_path):
    bad = GOOD_TIMING.replace(",120.5,", ",,")
    with pytest.raises(DataValidationError, match="non finite"):
        loaders.load_timing_csv(_write(tmp_path, "t.csv", bad))


def test_load_timing_rejects_empty(tmp_path):
    header = GOOD_TIMING.splitlines()[0] + "\n"
    with pytest.raises(DataValidationError, match="no data rows"):
        loaders.load_timing_csv(_write(tmp_path, "t.csv", header))


def test_load_timing_missing_file():
    with pytest.raises(FileNotFoundError):
        loaders.load_timing_csv("does_not_exist_12345.csv")


def test_load_nvml_ok(tmp_path):
    frame = loaders.load_nvml_csv(_write(tmp_path, "n.csv", GOOD_NVML))
    assert len(frame) == 2
    assert str(frame["timestamp"].dtype).startswith("datetime64")


def test_load_nvml_rejects_bad_timestamp(tmp_path):
    bad = GOOD_NVML.replace("2026-07-22T10:00:00Z", "not-a-time")
    with pytest.raises(DataValidationError, match="unparseable timestamps"):
        loaders.load_nvml_csv(_write(tmp_path, "n.csv", bad))


def test_load_nvml_rejects_negative_power(tmp_path):
    bad = GOOD_NVML.replace("120.0,52.0", "-120.0,52.0")
    with pytest.raises(DataValidationError, match="negative"):
        loaders.load_nvml_csv(_write(tmp_path, "n.csv", bad))


def test_load_ncu_ok(tmp_path):
    frame = loaders.load_ncu_csv(_write(tmp_path, "c.csv", GOOD_NCU))
    assert len(frame) == 2
    assert frame["metric_value"].dtype.kind == "f"


def test_load_ncu_rejects_nonnumeric_value(tmp_path):
    bad = GOOD_NCU.replace("1234567", "not_a_number")
    with pytest.raises(DataValidationError, match="non numeric"):
        loaders.load_ncu_csv(_write(tmp_path, "c.csv", bad))
