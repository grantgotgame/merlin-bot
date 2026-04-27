# Launches Merlin and opens "The Tower" web UI.
# Activates the local venv, ensures the web deps are present, then runs merlin.py.
$dir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $dir

$venv = Join-Path $dir "venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
$venvPip    = Join-Path $venv "Scripts\pip.exe"
$activate   = Join-Path $venv "Scripts\Activate.ps1"

if (-not (Test-Path $venvPython)) {
    Write-Host "  No venv found at $venv -- creating..." -ForegroundColor Yellow
    python -m venv $venv
}

if (Test-Path $activate) { & $activate }

# Self-heal: if the web deps somehow vanished, install them silently before boot.
$probe = & $venvPython -c "import fastapi, uvicorn" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing missing web dependencies (fastapi, uvicorn)..." -ForegroundColor Yellow
    & $venvPip install fastapi "uvicorn[standard]" --quiet
}

Write-Host ""
Write-Host "  Starting Merlin..." -ForegroundColor Cyan
Write-Host "  The Tower will open at http://localhost:8800" -ForegroundColor DarkGray
Write-Host ""

& $venvPython merlin.py
