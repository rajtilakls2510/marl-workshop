#!/usr/bin/env bash
# Maintainer: publish the two locally built images to GHCR (linux/amd64).
#   marl-workshop:latest (CUDA PyTorch, ./docker/build.sh)          -> ghcr.io/<user>/marl-workshop:gpu
#   marl-workshop:cpu    (CPU PyTorch, FLAVOR=cpu ./docker/build.sh cpu) -> ghcr.io/<user>/marl-workshop:cpu
#
# One-time login:  echo $GITHUB_PAT | docker login ghcr.io -u <GITHUB_USER> --password-stdin
#   (classic PAT with the write:packages scope). Afterwards make the package PUBLIC in the GitHub UI
#   (Packages -> marl-workshop -> Package settings -> Change visibility): Codespaces and attendees pull anonymously.
#
# Usage: ./docker/push.sh            push both
#        ./docker/push.sh cpu        push only one (cpu | gpu)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

push() {   # push <local image> <registry image>
  if ! docker image inspect "$1" >/dev/null 2>&1; then
    echo "ERROR: $1 not found - build it first (see the header of this script)." >&2; exit 1
  fi
  docker tag "$1" "$2"
  echo "==> Pushing $2"
  docker push "$2"
}
case "${1:-both}" in
  cpu)  push "${LOCAL_IMAGE}:cpu" "$REGISTRY_CPU" ;;
  gpu)  push "${LOCAL_IMAGE}:latest" "$REGISTRY_GPU" ;;
  both) push "${LOCAL_IMAGE}:cpu" "$REGISTRY_CPU"; push "${LOCAL_IMAGE}:latest" "$REGISTRY_GPU" ;;
  *) echo "Usage: $0 [cpu|gpu|both]" >&2; exit 1 ;;
esac
