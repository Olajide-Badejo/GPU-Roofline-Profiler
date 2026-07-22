# One command from a built tree to compiled reports.
#
#   .\scripts\run_full_pipeline.ps1
#
# Runs, in order: build, GPU correctness tests, the timing sweep, the Nsight
# Systems pass, the analysis, and the report build. The Nsight Compute pass is
# included only when this shell is elevated, because ncu cannot read GPU
# counters otherwise and would silently produce a directory of "n/a" values.
# When it is skipped the analysis still runs and simply falls back to
# theoretical byte counts, with every affected point labelled as such.
#
# Correctness gates the timing: if a kernel test fails the pipeline stops rather
# than benchmarking code that computes the wrong answer.

param(
    [switch]$SkipNsys,
    [switch]$SkipNcu,
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "dev_env.ps1") -Quiet

if ([string]::IsNullOrEmpty($Config)) {
    $Config = Join-Path $repo "configs\sweep.yaml"
}

$stamp = (Get-Date -Format "yyyyMMddTHHmmssZ")
Push-Location $repo
try {
    $sha = (git rev-parse --short HEAD 2>$null)
    if (-not $sha) { $sha = "nogit" }
}
finally {
    Pop-Location
}
$runDir = Join-Path $repo "results\raw\${stamp}_${sha}"
$ncuDir = Join-Path $repo "results\raw\ncu_${stamp}"

$overall = [System.Diagnostics.Stopwatch]::StartNew()

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host ("=== {0} ===" -f $Name)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Body
    $sw.Stop()
    Write-Host ("--- {0} took {1}" -f $Name, $sw.Elapsed.ToString("hh\:mm\:ss"))
}

# Native tools are invoked through cmd because PowerShell turns any write to
# stderr by a native executable into an error record, and with
# ErrorActionPreference set to Stop an ordinary CMake deprecation warning would
# abort the whole pipeline. Exit codes are checked explicitly instead.
Invoke-Step "build" {
    $build = Join-Path $repo "build"
    $log = Join-Path $env:TEMP "roofline_build.log"
    cmd /c "cmake -S `"$repo`" -B `"$build`" -G Ninja > `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 20; throw "configure failed" }
    cmd /c "cmake --build `"$build`" >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 20; throw "build failed" }
}

Invoke-Step "GPU correctness tests" {
    $tests = Join-Path $repo "build\tests\cpp\kernel_tests.exe"
    $log = Join-Path $env:TEMP "roofline_tests.log"
    cmd /c "`"$tests`" --gtest_brief=1 > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    Get-Content $log -Tail 3
    if ($code -ne 0) {
        throw "kernel correctness tests failed; refusing to benchmark wrong answers"
    }
}

Invoke-Step "timing sweep" {
    $exe = Join-Path $repo "build\roofline_profiler.exe"
    $log = Join-Path $env:TEMP "roofline_sweep.log"
    cmd /c "`"$exe`" --config `"$Config`" --out `"$runDir`" > `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 20; throw "sweep failed" }
    Get-Content $log -Tail 4
}

if (-not $SkipNsys) {
    Invoke-Step "Nsight Systems" {
        & (Join-Path $PSScriptRoot "run_nsys_profile.ps1") -Config $Config
    }
}

# ncu needs administrator rights; without them every metric is "n/a".
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $SkipNcu -and $elevated) {
    Invoke-Step "Nsight Compute" {
        & (Join-Path $PSScriptRoot "run_ncu_profile.ps1") -OutDir $ncuDir -Force
    }
}
elseif (-not $SkipNcu) {
    Write-Host ""
    Write-Warning @"
Skipping the Nsight Compute pass: this shell is not elevated, and ncu returns
n/a for every counter without administrator rights. Re-run this script from an
elevated PowerShell for counter-backed analysis. The pipeline continues and the
roofline will use theoretical byte counts, labelled as such.
"@
    $ncuDir = ""
}

Invoke-Step "analysis" {
    $py = Join-Path $repo ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = "python" }
    # Not $args: that is a PowerShell automatic variable and assigning to it has
    # side effects inside the enclosing scope.
    $cliArgs = @(
        (Join-Path $repo "python\cli.py"),
        "--results", $runDir,
        "--report", (Join-Path $repo "report"),
        "--peaks", (Join-Path $runDir "peaks.csv")
    )
    if ($ncuDir -and (Test-Path $ncuDir)) { $cliArgs += @("--ncu", $ncuDir) }
    # Through cmd again: tqdm draws its progress bar on stderr, which PowerShell
    # would otherwise treat as a failure of the whole step.
    $quoted = ($cliArgs | ForEach-Object { "`"$_`"" }) -join " "
    $log = Join-Path $env:TEMP "roofline_analysis.log"
    cmd /c "`"$py`" $quoted > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    Get-Content $log | Where-Object { $_ -notmatch "artifact/s\]" }
    if ($code -ne 0) { throw "analysis failed" }
}

Invoke-Step "reports" {
    & (Join-Path $PSScriptRoot "build_reports.ps1")
}

$overall.Stop()
Write-Host ""
Write-Host ("[done] full pipeline in {0}" -f $overall.Elapsed.ToString("hh\:mm\:ss"))
Write-Host "results: $runDir"
Write-Host "report:  $(Join-Path $repo 'report\main.pdf')"
