#!/usr/bin/env bash
# Maintainer: publish the image to GHCR (linux/amd64 only; Apple Silicon Macs run it under emulation).
#
# One-time login:  echo $GITHUB_PAT | docker login ghcr.io -u <GITHUB_USER> --password-stdin
#   (PAT needs the write:packages scope). Afterwards make the package PUBLIC in the GitHub UI
#   (Packages -> marl-workshop -> Package settings -> Change visibility).
#
# Usage: GITHUB_USER=yourname ./docker/push.sh [tag]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

if [ -z "$GITHUB_USER" ]; then
  echo "ERROR: GITHUB_USER is empty (set it in docker/config.sh)." >&2; exit 1
fi
TAG="${1:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64}"

docker buildx inspect marl-builder >/dev/null 2>&1 || docker buildx create --name marl-builder --use
docker buildx use marl-builder
echo "==> Building + pushing ${REGISTRY_IMAGE}:${TAG} for ${PLATFORMS}"
docker buildx build \
  -f "$SCRIPT_DIR/Dockerfile" \
  --platform "$PLATFORMS" \
  -t "${REGISTRY_IMAGE}:${TAG}" \
  --push \
  "$REPO_ROOT"
