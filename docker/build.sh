#!/usr/bin/env bash
# Maintainer: build the workshop image locally (needs internet: pulls ~4 GB of wheels).
#
# Usage: ./docker/build.sh [tag]            tag defaults to "latest"
#        FLAVOR=cpu ./docker/build.sh cpu   slim CPU-only variant (~2 GB), tagged :cpu
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

TAG="${1:-latest}"
FLAVOR="${FLAVOR:-cu129}"

echo "==> Building ${LOCAL_IMAGE}:${TAG} (torch flavor: ${FLAVOR})"
DOCKER_BUILDKIT=1 docker build \
  -f "$SCRIPT_DIR/Dockerfile" \
  --build-arg TORCH_FLAVOR="$FLAVOR" \
  -t "${LOCAL_IMAGE}:${TAG}" \
  "$REPO_ROOT"

echo
docker images "${LOCAL_IMAGE}:${TAG}"
