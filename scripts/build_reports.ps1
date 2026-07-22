# Build every report to PDF without needing make or a Perl engine.
#
# Compiles the main report, the debug report, and the personal report with a
# direct pdflatex plus bibtex sequence. This is the reliable path on a fresh
# Windows box where MiKTeX's latexmk cannot run for want of Perl. Once Perl is
# installed the report Makefiles drive latexmk instead; this script stays as the
# no-dependency fallback the full pipeline calls.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

function Build-Report {
    param(
        [string]$Directory,
        [string]$Main,
        [bool]$UseBibtex
    )
    Write-Host "[build] $Main in $Directory"
    Push-Location $Directory
    try {
        $env:max_print_line = 1000
        & pdflatex -interaction=nonstopmode -halt-on-error "$Main.tex" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed for $Main" }
        if ($UseBibtex) {
            & bibtex $Main | Out-Null
        }
        & pdflatex -interaction=nonstopmode -halt-on-error "$Main.tex" | Out-Null
        & pdflatex -interaction=nonstopmode -halt-on-error "$Main.tex" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "pdflatex final pass failed for $Main" }
        if (-not (Test-Path "$Main.pdf")) { throw "no PDF produced for $Main" }
        $size = (Get-Item "$Main.pdf").Length
        Write-Host "[ok] $Main.pdf ($size bytes)"
    }
    finally {
        Pop-Location
    }
}

Build-Report -Directory (Join-Path $repo "report") -Main "main" -UseBibtex $true
Build-Report -Directory (Join-Path $repo "report_debug") -Main "debug_report" -UseBibtex $false
Write-Host "[done] all reports built"
