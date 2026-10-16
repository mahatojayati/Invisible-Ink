# PowerShell startup script for Invisible-Ink on Windows
Write-Host "Starting InvisibleInk..." -ForegroundColor Cyan

$venvPath = $null
if (Test-Path "venv\Scripts\Activate.ps1") {
    $venvPath = "venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv_app\Scripts\Activate.ps1") {
    $venvPath = "venv_app\Scripts\Activate.ps1"
} else {
    Write-Host "Creating virtual environment 'venv_app'..." -ForegroundColor Yellow
    $pyCmd = "python"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pyCmd = "py -3.12"
    }
    Invoke-Expression "$pyCmd -m venv venv_app"
    $venvPath = "venv_app\Scripts\Activate.ps1"
}

if ($venvPath -and (Test-Path $venvPath)) {
    Write-Host "Activating virtual environment: $venvPath" -ForegroundColor Green
    & $venvPath
}

Write-Host "Verifying dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "Starting Flask Server at http://127.0.0.1:5001 ..." -ForegroundColor Green
python app.py
