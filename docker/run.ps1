# Attendee (Windows PowerShell): start the workshop environment (JupyterLab in your browser).
#   powershell -ExecutionPolicy Bypass -File docker\run.ps1
#   ... -Cpu       never request the GPU
#   ... -Verify    run the environment self-check instead of Jupyter
# NOTE: written carefully but not yet tested on a real Windows machine.
param([switch]$Cpu, [switch]$Verify)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$GithubUser = if ($env:GITHUB_USER) { $env:GITHUB_USER } else { "rajtilakls2510" }  # keep in sync with config.sh
$LocalImage = "marl-workshop"
$RegistryImage = "ghcr.io/$GithubUser/marl-workshop"
$Port = if ($env:JUPYTER_PORT) { $env:JUPYTER_PORT } else { "8888" }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker is not installed. See docs\INSTALL_DOCKER.md, or use the Docker-free route: scripts\setup_uv.bat" -ForegroundColor Red; exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Start Docker Desktop, wait for the whale icon to settle, and retry." -ForegroundColor Red; exit 1
}

$Image = $env:IMAGE
if (-not $Image) {
    docker image inspect "${LocalImage}:latest" *> $null
    if ($LASTEXITCODE -eq 0) { $Image = "${LocalImage}:latest" } else { $Image = "${RegistryImage}:latest" }
}
docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Image $Image not found locally: pulling (several GB, this can take a while)"
    docker pull $Image
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pull failed. Ask the instructor for the USB/LAN copy and use docker\load.ps1" -ForegroundColor Red; exit 1 }
}

$GpuArgs = @()
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    docker run --rm --gpus all --entrypoint true $Image *> $null
    if ($LASTEXITCODE -eq 0) { $GpuArgs = @("--gpus", "all"); Write-Host "==> NVIDIA GPU available to Docker: enabling --gpus all" }
    else { Write-Host "==> NVIDIA GPU found but Docker can't use it: running on CPU" }
}

$Cmd = @()
if ($Verify) { $Cmd = @("python", "verify_env.py") }

docker run -it --rm `
    -p "127.0.0.1:${Port}:8888" `
    -v "${RepoRoot}:/workspace" `
    @GpuArgs `
    --name marl-workshop `
    $Image @Cmd
