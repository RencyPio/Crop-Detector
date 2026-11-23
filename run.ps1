param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-PortFree([int]$Port) {
    -not (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}
function Get-FreePort([int]$Start) {
    $p = $Start
    while (-not (Test-PortFree $p)) { $p++ }
    return $p
}

$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Setup needed: no venv found. Run these once:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\python.exe -m pip install -r requirements-dev.txt"
    exit 1
}
if (-not ((Test-Path "models\crop_disease_cnn.tflite") -or (Test-Path "models\crop_disease_cnn_ft_final.keras"))) {
    Write-Host "The trained model is missing. Copy crop_disease_cnn.tflite into the models folder." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path "data\chroma_db")) {
    Write-Host "Building the treatment knowledge base (first run only, about a minute)..." -ForegroundColor Cyan
    & $py -m src.rag.ingest
    & $py -m src.rag.vector_store
}
if (-not ((Test-Path ".env") -and (Select-String -Path ".env" -Pattern "^GROQ_API_KEY=.+" -Quiet))) {
    Write-Host "Note: no GROQ_API_KEY in .env, so advice will show the source passages instead of a written answer." -ForegroundColor Yellow
}

$apiPort = Get-FreePort 8000
$appPort = Get-FreePort 8501
if ($apiPort -ne 8000) { Write-Host "Port 8000 is busy, using $apiPort for the API." -ForegroundColor Yellow }

New-Item -ItemType Directory -Force logs | Out-Null
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:API_URL = "http://127.0.0.1:$apiPort"

Write-Host "Starting the diagnosis service (loading the model takes ~20 seconds)..." -ForegroundColor Cyan
$api = Start-Process -FilePath $py -PassThru -WindowStyle Hidden `
    -ArgumentList "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "$apiPort" `
    -RedirectStandardOutput "logs\api.out.log" -RedirectStandardError "logs\api.err.log"

try {
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        if ($api.HasExited) { break }
        try { Invoke-RestMethod "$env:API_URL/health" -TimeoutSec 2 | Out-Null; $ready = $true; break } catch { Start-Sleep 2 }
    }
    if (-not $ready) {
        Write-Host "The diagnosis service did not start. Last lines of logs\api.err.log:" -ForegroundColor Red
        Get-Content "logs\api.err.log" -Tail 15
        exit 1
    }
    Write-Host "Ready! Opening http://localhost:$appPort in your browser. Close this window to stop." -ForegroundColor Green
    & $py -m streamlit run frontend/app.py --server.port $appPort --browser.gatherUsageStats false `
        --server.headless $(if ($NoBrowser) { "true" } else { "false" })
}
finally {
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Stopped."
}
