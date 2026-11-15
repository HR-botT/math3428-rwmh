# build.ps1 — source: report.ipynb → report.md → report.pdf
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
    Write-Error "xelatex not found in PATH. Install MiKTeX or add its bin folder to PATH."
    exit 1
}

if (-not (Test-Path "report.ipynb")) {
    Write-Error "report.ipynb not found in $PSScriptRoot"
    exit 1
}

Write-Host "Extracting report.md from report.ipynb ..." -ForegroundColor Cyan
python extract_report.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "extract_report.py failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Rendering report.pdf via pandoc + xelatex ..." -ForegroundColor Cyan
pandoc report.md `
    -o report.pdf `
    --pdf-engine=xelatex `
    -V geometry:margin=1.5cm `
    -V fontsize=10pt `
    -V linestretch=1.05 `
    --include-in-header=header.tex

if ($LASTEXITCODE -ne 0) {
    Write-Error "pandoc failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Done. Generated report.md and report.pdf." -ForegroundColor Green