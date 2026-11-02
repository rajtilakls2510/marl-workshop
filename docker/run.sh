#!/usr/bin/env bash
# Attendee: start the workshop environment (JupyterLab in your browser).
#
#   ./docker/run.sh            run from the repo you cloned (your edits are saved in it)
#   IMAGE=... ./docker/run.sh  use a specific image
# It uses the image you pulled: ghcr.io/.../marl-workshop:gpu if you pulled it and Docker can use your NVIDIA GPU,
# otherwise :cpu (or :gpu on the CPU). If no image is there yet, it pulls :cpu (~2 GB).
#   ./docker/run.sh --cpu      never request the GPU
#   ./docker/run.sh verify     run the environment self-check instead of Jupyter
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

USE_GPU=1; CMD=()
for arg in "$@"; do
  case "$arg" in
    --cpu) USE_GPU=0 ;;
    verify) CMD=(python verify_env.py) ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed. Either install it (docs/INSTALL_DOCKER.md) or use the" >&2
  echo "       Codespaces route in your browser (docs/INSTALL_CODESPACES.md)." >&2; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is installed but not running (or you lack permission)." >&2
  echo "       Start Docker Desktop / the docker service; on Linux see docs/TROUBLESHOOTING.md." >&2; exit 1
fi

# Already running (e.g. from an earlier terminal)? Don't start a second copy.
if [ -n "$(docker ps -q -f name=^marl-workshop$)" ]; then
  echo "==> The workshop container is already running: open http://localhost:${JUPYTER_PORT}/lab?token=${JUPYTER_TOKEN}"
  echo "    (to restart it: docker stop marl-workshop, then run this script again)"
  exit 0
fi
docker rm marl-workshop >/dev/null 2>&1 || true   # leftover stopped container with the same name

have() { docker image inspect "$1" >/dev/null 2>&1; }
# Can Docker hand the NVIDIA GPU to a container of image $1? (cheap probe: start and stop a no-op container)
gpu_ok() { [ "$USE_GPU" = "1" ] && command -v nvidia-smi >/dev/null 2>&1 \
           && docker run --rm --gpus all --entrypoint true "$1" >/dev/null 2>&1; }

# Pick the image: $IMAGE > pulled :gpu (if the GPU works) > pulled :cpu > pulled :gpu (on CPU) > local build > pull :cpu
if [ -z "${IMAGE:-}" ]; then
  if have "$REGISTRY_GPU" && gpu_ok "$REGISTRY_GPU"; then IMAGE="$REGISTRY_GPU"
  elif have "$REGISTRY_CPU"; then IMAGE="$REGISTRY_CPU"
  elif have "$REGISTRY_GPU"; then IMAGE="$REGISTRY_GPU"
  elif have "${LOCAL_IMAGE}:latest"; then IMAGE="${LOCAL_IMAGE}:latest"
  elif have "${LOCAL_IMAGE}:cpu"; then IMAGE="${LOCAL_IMAGE}:cpu"
  else IMAGE="$REGISTRY_CPU"; fi
fi
if ! have "$IMAGE"; then
  echo "==> Image $IMAGE not found locally: pulling it (about 2 GB for :cpu, 8 GB for :gpu; this can take a while)"
  docker pull "$IMAGE" || { echo "ERROR: pull failed. Check your internet connection and docs/TROUBLESHOOTING.md." >&2; exit 1; }
fi
echo "==> Using image $IMAGE"

# GPU only for a CUDA image and only if the machine really supports it.
GPU_ARGS=()
case "$IMAGE" in
  *:cpu) ;;
  *) if gpu_ok "$IMAGE"; then GPU_ARGS=(--gpus all); echo "==> NVIDIA GPU available to Docker: enabling --gpus all"
     elif [ "$USE_GPU" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
       echo "==> NVIDIA GPU found but Docker can't use it (nvidia-container-toolkit missing?): running on CPU"; fi ;;
esac

USER_ARGS=()
[ "$(uname -s)" = "Linux" ] && USER_ARGS=(--user "$(id -u):$(id -g)")
TTY_ARGS=(-i); [ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-it)

# Bind to localhost only: the notebook token is public knowledge, so don't expose it to the room's network.
# ${arr[@]+"${arr[@]}"} keeps empty arrays working under `set -u` on macOS's bash 3.2.
docker run "${TTY_ARGS[@]}" --rm \
  -p "127.0.0.1:${JUPYTER_PORT}:8888" \
  -v "$REPO_ROOT":/workspace \
  ${USER_ARGS[@]+"${USER_ARGS[@]}"} ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  --name marl-workshop \
  "$IMAGE" ${CMD[@]+"${CMD[@]}"}
