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

```
ncu --query-metrics | Select-String -Pattern "dram|occupancy|bank|warp"
```

Record the exact names used into the table below when the pass is first run on
this machine.

### Metric mapping (to fill in from the installed ncu)

| concept | metric name (record from `ncu --query-metrics`) |
|---|---|
| achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` (verify) |
| DRAM read bytes | `dram__bytes_read.sum` (verify) |
| DRAM write bytes | `dram__bytes_write.sum` (verify) |
| shared bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` (verify) |
| warp execution efficiency | `smsp__thread_inst_executed_per_inst_executed.ratio` (verify) |

The names above are the usual ones for recent Nsight Compute, marked "verify"
because they must be confirmed against the installed version before use. Once
confirmed, drop the "verify" note and record the tool version alongside.

## Output parsing

The wrapper exports `ncu` results as CSV in the long form the analysis expects:
one row per (kernel, metric, value). `roofline.loaders.load_ncu_csv` validates
that the value column is numeric and rejects anything else loudly.
