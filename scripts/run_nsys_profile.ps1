# Nsight Systems pass: a labeled timeline of the whole sweep.
#
# Unlike Nsight Compute, this does not need administrator rights. It traces CUDA
# API and kernel activity, which is available to any user; only CPU sampling and
# context switch tracing are admin gated, and neither is needed here. nsys prints
# a warning about disabling them, which is expected and harmless.
#
# The driver pushes an NVTX range per (kernel, configuration) cell, so the
# timeline reads as labeled work instead of thousands of anonymous launches.

param(
    [string]$OutDir = "",
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

. (Join-Path $PSScriptRoot "dev_env.ps1") -Quiet

if ([string]::IsNullOrEmpty($OutDir)) {
    $stamp = (Get-Date -Format "yyyyMMddTHHmmssZ")
    $OutDir = Join-Path $repo "results\raw\nsys_$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ([string]::IsNullOrEmpty($Config)) {
    $Config = Join-Path $repo "configs\sweep.yaml"
}

$exe = Join-Path $repo "build\roofline_profiler.exe"
if (-not (Test-Path $exe)) { Write-Error "driver not built: $exe"; exit 1 }

(nsys --version) | Set-Content -Path (Join-Path $OutDir "nsys_version.txt") -Encoding utf8

$report = Join-Path $OutDir "timeline"
$log = Join-Path $OutDir "nsys_log.txt"
$driverOut = Join-Path $OutDir "driver"

Write-Host "[nsys] tracing the sweep, output to $OutDir"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

# Run through cmd so that nsys writing to stderr does not trip PowerShell's
# native command error handling, which turns ordinary warnings into failures.
$cmd = "nsys profile --force-overwrite true -o `"$report`" --trace cuda,nvtx " +
       "`"$exe`" --config `"$Config`" --out `"$driverOut`" > `"$log`" 2>&1"
cmd /c $cmd
$code = $LASTEXITCODE
$sw.Stop()

Write-Host ("[nsys] finished in {0} (exit {1})" -f $sw.Elapsed.ToString("hh\:mm\:ss"), $code)

if (Test-Path "$report.nsys-rep") {
    $size = (Get-Item "$report.nsys-rep").Length
    Write-Host "[ok] $report.nsys-rep ($size bytes)"

    # Export the summary statistics as CSV so the analysis and the report can use
    # real numbers rather than a screenshot of a GUI.
    foreach ($rep in @("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum")) {
        $target = Join-Path $OutDir "$rep.csv"
        cmd /c "nsys stats --report $rep --format csv --force-export true `"$report.nsys-rep`" > `"$target`" 2>&1"
        if (Test-Path $target) {
            Write-Host "[ok] $rep.csv"
        }
    }
}
else {
    Write-Warning "no nsys report produced; see $log"
    exit 1
}
