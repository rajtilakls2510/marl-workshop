# Attendee (Windows, offline path): load the image from the tarball on the USB stick / LAN server.
#   powershell -ExecutionPolicy Bypass -File docker\load.ps1 D:\marl-workshop_latest.tar.gz
# docker load reads .tar.gz directly, so no gunzip is needed.
param([Parameter(Mandatory=$true)][string]$File)
if (-not (Test-Path $File)) { Write-Host "File not found: $File" -ForegroundColor Red; exit 1 }
Write-Host "==> Loading $File into Docker (a few minutes, no internet needed)"
docker load -i $File
Write-Host "==> Done. Now run: docker\run.bat"
