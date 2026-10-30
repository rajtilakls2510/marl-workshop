#!/usr/bin/env bash
# Facilitator: pre-download every wheel for the Docker-free route so it can be installed offline.
# One wheelhouse per target platform (the wheels are OS-specific); needs internet once.
#
# Usage: ./facilitator/make_wheelhouse.sh <linux|windows|macos-arm|macos-intel> [cpu|auto]
#   default torch flavour is "cpu" (~200 MB); "auto" fetches the large CUDA wheels (linux/windows only).
# Attendee side:  ./scripts/setup_uv.sh --wheelhouse /path/to/wheelhouse-<platform>
# NOTE: only the "linux" wheelhouse is something we can test on the maintainer machine.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLAT="${1:?usage: $0 <linux|windows|macos-arm|macos-intel> [cpu|auto]}"; FLAVOR="${2:-cpu}"

case "$PLAT" in
  linux)       PIP_PLAT="manylinux2014_x86_64" ;;
  windows)     PIP_PLAT="win_amd64" ;;
  macos-arm)   PIP_PLAT="macosx_11_0_arm64" ;;
  macos-intel) PIP_PLAT="macosx_10_15_x86_64" ;;
  *) echo "unknown platform $PLAT" >&2; exit 1 ;;
esac
OUT="$SCRIPT_DIR/kit/wheelhouse-$PLAT"; mkdir -p "$OUT"
EXTRA=()
if [ "$FLAVOR" = "cpu" ] && [ "$PLAT" != "macos-arm" ] && [ "$PLAT" != "macos-intel" ]; then
  EXTRA=(--extra-index-url https://download.pytorch.org/whl/cpu)
elif [ "$FLAVOR" = "auto" ] && [ "$PLAT" != "macos-arm" ] && [ "$PLAT" != "macos-intel" ]; then
  EXTRA=(--extra-index-url https://download.pytorch.org/whl/cu129)
fi

# smaclite is a source distribution (no wheel upstream): build it once into the wheelhouse (pure Python).
python3 -m pip wheel --no-deps -w "$OUT" \
  "https://github.com/uoe-agents/smaclite/archive/0d6abbea1334011984ae4704a0ef7a33c36057d6.zip"

grep -v '^smaclite' "$REPO_ROOT/requirements.txt" > "$OUT/.req.txt"
echo "smaclite" >> "$OUT/.req.txt"
python3 -m pip download -d "$OUT" -r "$OUT/.req.txt" --find-links "$OUT" \
  --python-version 3.10 --platform "$PIP_PLAT" --only-binary=:all: "${EXTRA[@]}"
rm "$OUT/.req.txt"

# Also bundle the uv binary for this platform when we can fetch it.
echo "==> Wheelhouse in $OUT ($(du -sh "$OUT" | cut -f1))"
