# Start the demo on Windows (PowerShell): FastAPI backend on :8000 + React (Vite) frontend on :5173.
#   powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
#   powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -ModelDir models\effv2s384   # single model
param([string]$ModelDir = "")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if ($ModelDir) { $env:MODEL_DIR = $ModelDir; Write-Host "model dir: $ModelDir" } else { Write-Host "classifier: configs\inference.yaml" }
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv found - follow the Quick start in README.md first." }
$backend = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "demo.backend.app:app", "--port", "8000" -PassThru -NoNewWindow
try {
    Push-Location demo\frontend
    if (-not (Test-Path node_modules)) { npm install --no-audit --no-fund }
    npx vite --open
} finally {
    Pop-Location
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
}
