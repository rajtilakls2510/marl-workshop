#!/usr/bin/env bash
# Attendee: start the workshop environment (JupyterLab in your browser).
#
#   ./docker/run.sh            run from the repo you cloned (your edits are saved in it)
#   IMAGE=... ./docker/run.sh  use a specific image (e.g. LAN registry 192.168.0.5:5000/marl-workshop)
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
  echo "       Docker-free route: scripts/setup_uv.sh  (docs/INSTALL_UV.md)." >&2; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is installed but not running (or you lack permission)." >&2
  echo "       Start Docker Desktop / the docker service; on Linux see docs/TROUBLESHOOTING.md." >&2; exit 1
fi

# Pick the image: $IMAGE > local build/load > public registry.
if [ -z "${IMAGE:-}" ]; then
  if docker image inspect "${LOCAL_IMAGE}:latest" >/dev/null 2>&1; then IMAGE="${LOCAL_IMAGE}:latest"
  else IMAGE="${REGISTRY_IMAGE}:latest"; fi
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> Image $IMAGE not found locally: pulling (several GB, this can take a while)"
  docker pull "$IMAGE" || { echo "ERROR: pull failed. Ask the instructor for the USB/LAN copy and use ./docker/load.sh" >&2; exit 1; }
fi

# GPU only if the machine really supports it (cheap probe: start and stop a no-op container).
GPU_ARGS=()
if [ "$USE_GPU" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if docker run --rm --gpus all --entrypoint true "$IMAGE" >/dev/null 2>&1; then
    GPU_ARGS=(--gpus all); echo "==> NVIDIA GPU available to Docker: enabling --gpus all"
  else
    echo "==> NVIDIA GPU found but Docker can't use it (nvidia-container-toolkit missing?): running on CPU"
  fi
fi

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
