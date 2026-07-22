# Profiling guide

How to reproduce the Nsight Systems and Nsight Compute passes step by step,
including the metric name mapping, which shifts between tool versions.

## Prerequisites

- CUDA Toolkit installed (brings `nsys` and `ncu`).
- On WSL2 only: GPU performance counter access must be enabled in the Windows
  NVIDIA control panel, and some counters are still unavailable. Verify `ncu` can
  actually collect counters before relying on it, and record that verification.
- The benchmark driver built in Release.

## Nsight Systems (timeline)

`scripts/run_nsys_profile.ps1` wraps the driver under `nsys profile`. The kernels
carry NVTX ranges around each kernel and configuration, so the timeline is
labeled rather than a wall of anonymous launches. Reports land in a predictable
folder under the run directory. Nsight Systems is light enough to run over the
full sweep.

## Nsight Compute (counters)

`scripts/run_ncu_profile.ps1` runs `ncu` over the curated subset in
`configs/sweep.yaml` under `ncu_subset`, not the full sweep, because `ncu`
replays each kernel many times per metric set and profiling everything would take
hours to no benefit. The wrapper prints a `[cell k/n]` banner per cell with
elapsed time, since these cells are slow and few, and it skips cells already
marked `.done` so an interrupted pass resumes.

### Querying metric names

Metric names move between `ncu` versions, so query them from the installed tool
rather than hardcoding:

```powershell
ncu --query-metrics | Select-String -Pattern "dram|occupancy|bank|warp"
```

Record the exact names used into the table below when the pass is first run on
this machine.

### Metric mapping (verified on this machine)

Verified against Nsight Compute **2026.2.1.0** on the RTX 5070 (sm_120, compute
capability 12.0), by reading `ncu --query-metrics` rather than assuming.

| concept | metric name | status |
| --- | --- | --- |
| DRAM read bytes | `dram__bytes_op_read.sum` | corrected, see below |
| DRAM write bytes | `dram__bytes_op_write.sum` | corrected, see below |
| achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | confirmed |
| shared bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | confirmed |
| warp execution efficiency | `smsp__thread_inst_executed_per_inst_executed.ratio` | confirmed |
| L2 hit rate | `lts__t_sector_hit_rate.pct` | confirmed |
| SM throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | confirmed |
| kernel duration | `gpu__time_duration.sum` | confirmed |
| global load sectors | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | confirmed present |
| global load requests | `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | confirmed present |
| global store sectors | `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | confirmed present |
| global store requests | `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` | confirmed present |

**The DRAM name trap.** Most published examples use `dram__bytes_read.sum` and
`dram__bytes_write.sum`. Those names do not exist on this chip, where the
counters carry an `_op_` infix. Critically, `ncu` does **not** reject an
unsupported metric name: it accepts it, profiles the kernel, and reports the
value as `n/a`. The resulting CSV is well formed and completely empty of the
thing you asked for. This is why the wrapper script counts `n/a` values after
every pass and warns, and why the metric list is derived from `--query-metrics`
on the machine rather than from documentation.

**Coalescing.** Sectors divided by requests is the coalescing measure. A fully
coalesced warp reading 4 byte values touches 128 contiguous bytes, which is 4
sectors of 32 bytes, so **4 sectors per request is ideal** and a larger ratio
means the warp is scattering across more sectors than the data requires.

### Permissions

Nsight Compute needs administrator rights on Windows. Without them it attaches,
runs the kernel, and returns `n/a` for every counter without any error. Either
run the pass from an elevated shell (what this project does) or set
`HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak\RmProfilingAdminOnly`
to 0 and reboot, which permanently allows non-admin profiling. Nsight Systems
does not need elevation for CUDA and NVTX tracing.

## Output parsing

The wrapper exports `ncu` results as CSV in the long form the analysis expects:
one row per (kernel, metric, value). `roofline.loaders.load_ncu_csv` validates
that the value column is numeric and rejects anything else loudly.
