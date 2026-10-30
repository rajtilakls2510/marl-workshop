@echo off
REM Double-click friendly wrapper for setup_uv.ps1 (bypasses the PowerShell script-execution policy).
REM Installs the small CPU-only PyTorch by default. For an NVIDIA GPU build, run from PowerShell:
REM     powershell -ExecutionPolicy Bypass -File scripts\setup_uv.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_uv.ps1" -Cpu %*
pause
