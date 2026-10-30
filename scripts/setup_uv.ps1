# Docker-free setup for Windows (PowerShell): creates .\.venv with Python 3.10 and installs everything.
#   powershell -ExecutionPolicy Bypass -File scripts\setup_uv.ps1            (auto-detect GPU)
#   ... -Cpu                    force the small CPU-only torch (~200 MB instead of ~3 GB)
#   ... -Wheelhouse D:\wheels   offline install from a folder of wheels (USB kit)
# NOTE: written carefully but not yet tested on a real Windows machine.
param([switch]$Cpu, [string]$Wheelhouse = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$Backend = if ($Cpu) { "cpu" } else { "auto" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> uv not found: installing it (no admin rights needed)"
    if ($Wheelhouse -and (Test-Path "$Wheelhouse\uv.exe")) {
        New-Item -ItemType Directory -Force "$HOME\.local\bin" | Out-Null
        Copy-Item "$Wheelhouse\uv.exe" "$HOME\.local\bin\uv.exe"
    } else {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: uv installation failed; see docs\INSTALL_UV.md (pip fallback)." -ForegroundColor Red; exit 1
}

Write-Host "==> Creating .venv (Python 3.10)"
uv venv --python 3.10 --allow-existing .venv
$Py = ".\.venv\Scripts\python.exe"

Write-Host "==> Installing packages (torch backend: $Backend) -- this takes a few minutes"
if ($Wheelhouse) { uv pip install --python $Py --no-index --find-links $Wheelhouse -r requirements.txt }
else             { uv pip install --python $Py --torch-backend $Backend -r requirements.txt }
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: install failed" -ForegroundColor Red; exit 1 }

Write-Host "==> Verifying"
& $Py verify_env.py

Write-Host "`nSetup done. Start the notebooks with:  scripts\start_jupyter.bat"
