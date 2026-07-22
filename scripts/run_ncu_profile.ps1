# Nsight Compute pass over the curated kernel subset.
#
# MUST BE RUN FROM AN ELEVATED POWERSHELL. On Windows the NVIDIA driver restricts
# GPU performance counters to administrators unless RmProfilingAdminOnly is set
# to 0 in the registry. Without elevation ncu still attaches and still runs the
# kernels, but every metric comes back "n/a", which is a silent and very
# convincing way to collect nothing. This script checks for elevation up front
# and refuses rather than producing a file full of "n/a".
#
# ncu replays each kernel many times per metric set, so this deliberately covers
# a curated subset of sizes rather than the full timing sweep.

param(
    [string]$OutDir = "",
    [switch]$Force,
    [int]$Size = 2048
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

# --- elevation gate --------------------------------------------------------
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error @"
Nsight Compute needs administrator rights to read GPU performance counters on
Windows. Re-run this script from an elevated PowerShell, or set
HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak\RmProfilingAdminOnly
to 0 and reboot to allow non-admin profiling.
"@
    exit 1
}

. (Join-Path $PSScriptRoot "dev_env.ps1") -Quiet

if ([string]::IsNullOrEmpty($OutDir)) {
    $stamp = (Get-Date -Format "yyyyMMddTHHmmssZ")
    $OutDir = Join-Path $repo "results\raw\ncu_$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$exe = Join-Path $repo "build\roofline_profiler.exe"
if (-not (Test-Path $exe)) { Write-Error "driver not built: $exe"; exit 1 }

# Metric names shift between Nsight Compute versions, so the exact list this run
# used is recorded next to the results rather than trusted from memory.
$metrics = @(
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "lts__t_sector_hit_rate.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum"
)
$metricArg = ($metrics -join ",")
Set-Content -Path (Join-Path $OutDir "metrics_used.txt") -Value $metrics -Encoding utf8

# Record the tool version alongside, since the mapping is version dependent.
(ncu --version) | Set-Content -Path (Join-Path $OutDir "ncu_version.txt") -Encoding utf8

# The kernels worth the replay cost: the transpose pair (the coalescing and bank
# conflict story) and the GEMM ladder (the reuse story).
$cells = @(
    @{ name = "transpose_naive";        regex = "transpose_naive_kernel" },
    @{ name = "transpose_tiled";        regex = "transpose_tiled_kernel" },
    @{ name = "gemm_naive";             regex = "gemm_naive_kernel" },
    @{ name = "gemm_tiled";             regex = "gemm_tiled_kernel" },
    @{ name = "gemm_register_blocked";  regex = "gemm_register_blocked_kernel" },
    @{ name = "gemm_vectorized";        regex = "gemm_vectorized_kernel" },
    @{ name = "saxpy";                  regex = "saxpy_kernel" }
)

# A reduced sweep so ncu is not replaying the whole timing matrix.
$ncuConfig = Join-Path $OutDir "sweep_ncu.yaml"
@"
harness:
  warmup_iterations: 0
  timed_batches: 1
  launches_per_batch: 1
  vram_headroom_mib: 1536
saxpy:
  sizes: [16777216]
  block_sizes: [256]
reduction:
  sizes: []
  block_sizes: []
transpose:
  sizes: [$Size]
  tile_dims: [32]
gemv:
  sizes: []
  block_sizes: []
gemm:
  sizes: [$Size]
  variants: [naive, tiled, register_blocked, vectorized]
  tile_dim: 32
"@ | Set-Content -Path $ncuConfig -Encoding utf8

$total = $cells.Count
$index = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($cell in $cells) {
    $index++
    $csv = Join-Path $OutDir ("{0}.csv" -f $cell.name)
    $done = "$csv.done"
    if ((Test-Path $done) -and (-not $Force)) {
        Write-Host ("[cell {0}/{1}] {2}: already done, skipping" -f $index, $total, $cell.name)
        continue
    }

    Write-Host ("[cell {0}/{1}] {2} (elapsed {3})" -f `
        $index, $total, $cell.name, $sw.Elapsed.ToString("hh\:mm\:ss"))

    & ncu --metrics $metricArg `
          --kernel-name ("regex:" + $cell.regex) `
          --launch-count 1 `
          --csv `
          --target-processes all `
          $exe --config $ncuConfig --out (Join-Path $OutDir "driver_scratch") |
        Set-Content -Path $csv -Encoding utf8

    if ($LASTEXITCODE -eq 0) {
        New-Item -ItemType File -Path $done -Force | Out-Null
    }
    else {
        Write-Warning ("cell {0} failed with exit {1}" -f $cell.name, $LASTEXITCODE)
    }
}

$sw.Stop()
Write-Host ("[done] {0} cells in {1}" -f $total, $sw.Elapsed.ToString("hh\:mm\:ss"))
Write-Host "output: $OutDir"
