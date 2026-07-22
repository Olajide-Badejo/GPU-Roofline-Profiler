"""Tests for the Nsight Compute parser.

The parser has to cope with two real properties of ncu output: the profiled
program's own stdout appears before the CSV header, and unsupported metrics come
back as the string "n/a" rather than as an error. Both are covered here, because
both have already cost a run on this project.
"""

from __future__ import annotations

import pytest

from roofline import ncu
from roofline.loaders import DataValidationError

# Real shape of an ncu export: driver stdout first, then the CSV.
NCU_SAMPLE = '''device: NVIDIA GeForce RTX 5070 (sm_120), 48 SMs
measuring ceilings ...
wrote 7 rows to somewhere/timing.csv
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)","0","12.0","s","dram__bytes_op_read.sum","","53,261,312"
"0","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)","0","12.0","s","dram__bytes_op_write.sum","","3,377,920"
"0","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)","0","12.0","s","gpu__time_duration.sum","ns","10,196,000"
"0","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)","0","12.0","s","lts__t_sector_hit_rate.pct","%","97.03"
"0","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)","0","12.0","s","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum","","400"
"0","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)","0","12.0","s","l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum","","100"
'''

ALL_NA = '''some driver output
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","p.exe","h","k","1","7","(1,1,1)","(1,1,1)","0","12.0","s","dram__bytes_read.sum","","n/a"
"0","1","p.exe","h","k","1","7","(1,1,1)","(1,1,1)","0","12.0","s","dram__bytes_write.sum","","n/a"
'''


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parses_past_leading_driver_output(tmp_path):
    tidy = ncu.parse_ncu_csv(_write(tmp_path, "gemm_tiled.csv", NCU_SAMPLE))
    assert len(tidy) == 6
    assert set(tidy.columns) == {"kernel", "metric_name", "metric_value"}


def test_strips_thousands_separators(tmp_path):
    tidy = ncu.parse_ncu_csv(_write(tmp_path, "c.csv", NCU_SAMPLE))
    read = tidy.loc[tidy["metric_name"] == "dram__bytes_op_read.sum", "metric_value"]
    assert float(read.iloc[0]) == pytest.approx(53_261_312)


def test_all_na_raises_rather_than_returning_zeros(tmp_path):
    # This is the exact failure that a wrong metric name or a missing privilege
    # produces, and it must never look like a successful load of zeros.
    with pytest.raises(DataValidationError, match="n/a"):
        ncu.parse_ncu_csv(_write(tmp_path, "c.csv", ALL_NA))


def test_missing_header_raises(tmp_path):
    with pytest.raises(DataValidationError, match="no ncu CSV header"):
        ncu.parse_ncu_csv(_write(tmp_path, "c.csv", "just driver noise\n"))


def test_summarize_derives_bytes_bandwidth_and_ratios(tmp_path):
    tidy = ncu.parse_ncu_csv(_write(tmp_path, "gemm_tiled.csv", NCU_SAMPLE))
    tidy.insert(0, "cell", "gemm_tiled")
    summary = ncu.summarize_cells(tidy)

    row = summary.iloc[0]
    assert row["measured_bytes"] == pytest.approx(53_261_312 + 3_377_920)
    assert row["duration_ms"] == pytest.approx(10.196)
    # bytes per nanosecond is GB/s directly.
    assert row["dram_gbps"] == pytest.approx(
        (53_261_312 + 3_377_920) / 10_196_000, rel=1e-6
    )
    assert row["l2_hit_pct"] == pytest.approx(97.03)
    assert row["ld_sectors_per_request"] == pytest.approx(4.0)


def test_parse_directory_tags_cells(tmp_path):
    _write(tmp_path, "gemm_tiled.csv", NCU_SAMPLE)
    _write(tmp_path, "gemm_naive.csv", NCU_SAMPLE)
    tidy = ncu.parse_ncu_directory(tmp_path)
    assert set(tidy["cell"].unique()) == {"gemm_tiled", "gemm_naive"}


def test_parse_directory_skips_unusable_but_keeps_good(tmp_path):
    _write(tmp_path, "good.csv", NCU_SAMPLE)
    _write(tmp_path, "bad.csv", ALL_NA)
    tidy = ncu.parse_ncu_directory(tmp_path)
    assert set(tidy["cell"].unique()) == {"good"}


def test_percentages_are_averaged_not_summed(tmp_path):
    # Two profiled launches of the same kernel, each with a 90 percent L2 hit
    # rate and 100 bytes read. The bytes must add to 200; the hit rate must stay
    # 90, not become 180. Summing percentages once produced a reported hit rate
    # above 100, which is impossible and is what this guards against.
    doubled = NCU_SAMPLE + (
        '"1","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)",'
        '"0","12.0","s","lts__t_sector_hit_rate.pct","%","97.03"\n'
        '"1","1","p.exe","h","gemm_tiled_kernel","1","7","(32,32,1)","(64,64,1)",'
        '"0","12.0","s","dram__bytes_op_read.sum","","53,261,312"\n'
    )
    tidy = ncu.parse_ncu_csv(_write(tmp_path, "gemm_tiled.csv", doubled))
    tidy.insert(0, "cell", "gemm_tiled")
    row = ncu.summarize_cells(tidy).iloc[0]

    assert row["l2_hit_pct"] == pytest.approx(97.03)
    assert row["dram_read_bytes"] == pytest.approx(2 * 53_261_312)


def test_implausible_percentages_are_reported_not_clamped(tmp_path):
    # A hit rate above 100 is physically impossible but this card really does
    # report one. It must be surfaced, and it must not be quietly rewritten.
    impossible = NCU_SAMPLE.replace(
        '"lts__t_sector_hit_rate.pct","%","97.03"',
        '"lts__t_sector_hit_rate.pct","%","100.39"',
    )
    tidy = ncu.parse_ncu_csv(_write(tmp_path, "gemm_naive.csv", impossible))
    tidy.insert(0, "cell", "gemm_naive")
    summary = ncu.summarize_cells(tidy)

    flagged = ncu.implausible_percentages(summary)
    assert ("gemm_naive", "l2_hit_pct", pytest.approx(100.39)) in [
        (c, col, v) for c, col, v in flagged
    ]
    # The value survives unchanged; nothing clamped it to look respectable.
    assert summary.iloc[0]["l2_hit_pct"] == pytest.approx(100.39)


def test_plausible_percentages_are_not_flagged(tmp_path):
    tidy = ncu.parse_ncu_csv(_write(tmp_path, "gemm_tiled.csv", NCU_SAMPLE))
    tidy.insert(0, "cell", "gemm_tiled")
    assert ncu.implausible_percentages(ncu.summarize_cells(tidy)) == []


def test_intensive_classification():
    assert ncu._is_intensive("lts__t_sector_hit_rate.pct")
    assert ncu._is_intensive("smsp__thread_inst_executed_per_inst_executed.ratio")
    assert ncu._is_intensive("sm__warps_active.avg.pct_of_peak_sustained_active")
    assert not ncu._is_intensive("dram__bytes_op_read.sum")
    assert not ncu._is_intensive("gpu__time_duration.sum")


def test_ideal_sectors_per_request_depends_on_access_width():
    # 4 byte scalar access, and 16 byte float4 access.
    assert ncu.ideal_sectors_per_request(4) == pytest.approx(4.0)
    assert ncu.ideal_sectors_per_request(16) == pytest.approx(16.0)
    with pytest.raises(ValueError):
        ncu.ideal_sectors_per_request(0)
