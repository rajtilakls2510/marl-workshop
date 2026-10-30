@echo off
REM Double-click friendly wrapper for start_jupyter.ps1 (bypasses the PowerShell script-execution policy).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_jupyter.ps1" %*
pause
