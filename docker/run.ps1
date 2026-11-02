# Attendee (Windows PowerShell): start the workshop environment (JupyterLab in your browser).
#   powershell -ExecutionPolicy Bypass -File docker\run.ps1
#   ... -Cpu       never request the GPU
#   ... -Verify    run the environment self-check instead of Jupyter
# It uses the image you pulled: ghcr.io/.../marl-workshop:gpu if you pulled it and Docker can use your NVIDIA GPU,
# otherwise :cpu (or :gpu on the CPU). If no image is there yet, it pulls :cpu (~2 GB).
# NOTE: written carefully but not yet tested on a real Windows machine.
param([switch]$Cpu, [switch]$Verify)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$GithubUser = if ($env:GITHUB_USER) { $env:GITHUB_USER } else { "rajtilakls2510" }  # keep in sync with config.sh
$LocalImage = "marl-workshop"
$RegistryCpu = "ghcr.io/$GithubUser/marl-workshop:cpu"
$RegistryGpu = "ghcr.io/$GithubUser/marl-workshop:gpu"
$Port = if ($env:JUPYTER_PORT) { $env:JUPYTER_PORT } else { "8888" }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker is not installed. See docs\INSTALL_DOCKER.md, or use the Codespaces route in your browser (docs\INSTALL_CODESPACES.md)" -ForegroundColor Red; exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Start Docker Desktop, wait for the whale icon to settle, and retry." -ForegroundColor Red; exit 1
}

# Already running (e.g. from an earlier window)? Don't start a second copy.
if (docker ps -q -f "name=^marl-workshop$") {
    Write-Host "==> The workshop container is already running: open http://localhost:${Port}/lab?token=marl"
    Write-Host "    (to restart it: docker stop marl-workshop, then run this script again)"
    exit 0
}
docker rm marl-workshop *> $null

function Have($img) { docker image inspect $img *> $null; return ($LASTEXITCODE -eq 0) }
# Can Docker hand the NVIDIA GPU to a container of this image? (cheap probe: start and stop a no-op container)
function GpuOk($img) {
    if ($Cpu -or -not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $false }
    docker run --rm --gpus all --entrypoint true $img *> $null
    return ($LASTEXITCODE -eq 0)
}

# Pick the image: $env:IMAGE > pulled :gpu (if the GPU works) > pulled :cpu > pulled :gpu (on CPU) > local build > pull :cpu
$Image = $env:IMAGE
if (-not $Image) {
    if ((Have $RegistryGpu) -and (GpuOk $RegistryGpu)) { $Image = $RegistryGpu }
    elseif (Have $RegistryCpu) { $Image = $RegistryCpu }
    elseif (Have $RegistryGpu) { $Image = $RegistryGpu }
    elseif (Have "${LocalImage}:latest") { $Image = "${LocalImage}:latest" }
    elseif (Have "${LocalImage}:cpu") { $Image = "${LocalImage}:cpu" }
    else { $Image = $RegistryCpu }
}
if (-not (Have $Image)) {
    Write-Host "==> Image $Image not found locally: pulling it (about 2 GB for :cpu, 8 GB for :gpu; this can take a while)"
    docker pull $Image
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pull failed. Check your internet connection and docs\TROUBLESHOOTING.md." -ForegroundColor Red; exit 1 }
}
Write-Host "==> Using image $Image"

# GPU only for a CUDA image and only if the machine really supports it.
$GpuArgs = @()
if (-not $Image.EndsWith(":cpu")) {
    if (GpuOk $Image) { $GpuArgs = @("--gpus", "all"); Write-Host "==> NVIDIA GPU available to Docker: enabling --gpus all" }
    elseif (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Write-Host "==> NVIDIA GPU found but Docker can't use it: running on CPU" }
}

$Cmd = @()
if ($Verify) { $Cmd = @("python", "verify_env.py") }

docker run -it --rm `
    -p "127.0.0.1:${Port}:8888" `
    -v "${RepoRoot}:/workspace" `
    @GpuArgs `
    --name marl-workshop `
    $Image @Cmd
