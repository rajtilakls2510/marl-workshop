#!/usr/bin/env bash
# Maintainer: build the workshop image locally (needs internet: pulls ~4 GB of wheels).
#
# Usage: ./docker/build.sh [tag]            tag defaults to "latest"
# Full build output is also written to docker/build.log (watch it with: tail -f docker/build.log)
#        FLAVOR=cpu ./docker/build.sh cpu   slim CPU-only variant (~2 GB), tagged :cpu
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

TAG="${1:-latest}"
FLAVOR="${FLAVOR:-cu129}"

LOG="$SCRIPT_DIR/build.log"
echo "==> Building ${LOCAL_IMAGE}:${TAG} (torch flavor: ${FLAVOR}); log: $LOG"
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f "$SCRIPT_DIR/Dockerfile" \
  --build-arg TORCH_FLAVOR="$FLAVOR" \
  -t "${LOCAL_IMAGE}:${TAG}" \
  "$REPO_ROOT" 2>&1 | tee "$LOG"

echo
docker images "${LOCAL_IMAGE}:${TAG}"
