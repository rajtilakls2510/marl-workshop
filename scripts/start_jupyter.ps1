# Start JupyterLab from the uv environment (Windows).
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
if (-not (Test-Path .\.venv\Scripts\jupyter.exe)) { Write-Host "No .venv found: run scripts\setup_uv.bat first." -ForegroundColor Red; exit 1 }
$env:SDL_VIDEODRIVER = "dummy"; $env:PYGAME_HIDE_SUPPORT_PROMPT = "1"; $env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\jupyter.exe lab notebooks\00_check_setup.ipynb
