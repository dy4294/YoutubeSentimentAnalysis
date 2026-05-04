# run.ps1 — Launch the YouTube Sentiment Analysis app
# Usage: .\run.ps1

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py   = Join-Path $Root "venv\Scripts\python.exe"
$App  = Join-Path $Root "app.py"

if (-not (Test-Path $Py)) {
    Write-Error "Virtual environment not found at $Py"
    Write-Host "Create it with:  py -m venv venv  and then:  .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Write-Warning ".env file not found — API keys will not be loaded."
    Write-Host "Create a .env file with:"
    Write-Host "  YOUTUBE_API_KEY=your_key_here"
    Write-Host "  OPENAI_API_KEY=your_key_here"
}

Write-Host "Starting YouTube Sentiment Analysis..."
Write-Host "Open http://localhost:8501 in your browser"
Write-Host ""

& $Py -m streamlit run $App --server.port 8501
