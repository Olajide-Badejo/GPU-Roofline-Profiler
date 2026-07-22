# Set up a full CUDA + MSVC build environment in the current PowerShell session.
#
# Dot source it:   . .\scripts\dev_env.ps1
#
# Three things make this necessary on Windows. A shell started before the
# toolchain was installed carries a stale PATH, so the machine and user PATH are
# re-read from the registry. nvcc shells out to cl.exe as its host compiler, and
# cl.exe only exists inside a Visual Studio developer environment, so vcvars64 is
# imported by running it in cmd and copying the resulting variables across.
# Nsight Systems installs outside the toolkit and is not added to PATH by its
# installer, so it is located and appended.
#
# Pass -Quiet to skip the version banner.

param([switch]$Quiet)

$ErrorActionPreference = "Stop"

# 1. Refresh PATH from the registry so a stale session sees new installs.
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"

# 2. Import the MSVC developer environment so nvcc can find cl.exe.
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere not found; Visual Studio Build Tools are not installed"
}
$vsRoot = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsRoot) { throw "no Visual Studio install with the C++ tools found" }

$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

# Run vcvars64 in cmd, dump the resulting environment, and copy it into this
# session. This is the standard way to get a developer environment into
# PowerShell, since vcvars64 is a batch script and cannot set our variables.
cmd /c "call `"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
    }
}

# 3. Nsight Systems is installed outside the toolkit and not put on PATH.
$nsysDir = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Directory `
    -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "Nsight Systems*" } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if ($nsysDir) {
    $nsysTarget = Join-Path $nsysDir.FullName "target-windows-x64"
    if (Test-Path (Join-Path $nsysTarget "nsys.exe")) {
        $env:Path = "$nsysTarget;$env:Path"
    }
}

if (-not $Quiet) {
    Write-Host "=== build environment ==="
    $tools = @(
        @{ Name = "nvcc";  Args = @("--version") },
        @{ Name = "cl";    Args = @() },
        @{ Name = "ninja"; Args = @("--version") },
        @{ Name = "cmake"; Args = @("--version") },
        @{ Name = "nsys";  Args = @("--version") },
        @{ Name = "ncu";   Args = @("--version") }
    )
    foreach ($t in $tools) {
        $cmd = Get-Command $t.Name -ErrorAction SilentlyContinue
        if ($cmd) {
            Write-Host ("  {0,-6} {1}" -f $t.Name, $cmd.Source)
        }
        else {
            Write-Host ("  {0,-6} MISSING" -f $t.Name)
        }
    }
}
