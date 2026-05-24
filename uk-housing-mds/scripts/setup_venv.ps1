# Windows venv + Jupyter kernel setup for portfolio projects.
# Run once per project from the project root (the folder that contains requirements.txt):
#   .\scripts\setup_venv.ps1
#
# Idempotent: skips steps that are already done.

$ErrorActionPreference = "Stop"

# Project name = current directory name (used as kernel name)
$ProjectName = (Get-Item .).Name
$KernelName = "$ProjectName-venv"
$DisplayName = "$ProjectName (.venv)"

if (-not (Test-Path "requirements.txt")) {
    Write-Error "requirements.txt not found - run this from the project root."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating venv..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "Venv already exists; skipping creation." -ForegroundColor DarkGray
}

Write-Host "Installing requirements..." -ForegroundColor Cyan
& ".venv\Scripts\pip" install --upgrade pip --quiet
& ".venv\Scripts\pip" install -r requirements.txt

Write-Host "Installing ipykernel + registering Jupyter kernel..." -ForegroundColor Cyan
& ".venv\Scripts\pip" install ipykernel --quiet
& ".venv\Scripts\python" -m ipykernel install --user --name $KernelName --display-name $DisplayName | Out-Null

Write-Host ""
Write-Host "  Done. Activate the venv with:" -ForegroundColor Green
Write-Host "    .venv\Scripts\activate" -ForegroundColor Green
Write-Host ""
Write-Host "  In VS Code, select the Jupyter kernel: '$DisplayName'" -ForegroundColor Green
