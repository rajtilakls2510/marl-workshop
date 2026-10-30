#!/usr/bin/env bash
# Maintainer: export the image to a gzip tarball for USB / LAN distribution.
# Usage: ./docker/save.sh [tag] [output_dir]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

TAG="${1:-latest}"
OUT_DIR="${2:-$SCRIPT_DIR/../facilitator/kit}"
OUT_FILE="$OUT_DIR/${IMAGE_NAME}_${TAG}.tar.gz"

if ! docker image inspect "${LOCAL_IMAGE}:${TAG}" >/dev/null 2>&1; then
  echo "ERROR: image ${LOCAL_IMAGE}:${TAG} not found. Run ./docker/build.sh ${TAG} first." >&2; exit 1
fi
mkdir -p "$OUT_DIR"
echo "==> Saving ${LOCAL_IMAGE}:${TAG} -> $OUT_FILE (a few minutes; the image is large)"
docker save "${LOCAL_IMAGE}:${TAG}" | gzip > "$OUT_FILE"
du -h "$OUT_FILE"
