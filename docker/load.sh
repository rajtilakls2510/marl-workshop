#!/usr/bin/env bash
# Attendee (offline path): load the image from the tarball you got on a USB stick / LAN server.
# Usage: ./docker/load.sh /path/to/marl-workshop_latest.tar.gz
set -euo pipefail
FILE="${1:-}"
if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
  echo "Usage: $0 /path/to/marl-workshop_latest.tar.gz" >&2; exit 1
fi
echo "==> Loading $FILE into Docker (a few minutes, no internet needed)"
gunzip -c "$FILE" | docker load
echo
echo "==> Done. Now run:  ./docker/run.sh"
