@echo off
REM Double-click friendly wrapper for run.ps1 (bypasses the PowerShell script-execution policy).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
pause
