#!/usr/bin/env bash
# Facilitator: assemble facilitator/kit/ -> copy this folder onto USB sticks.
# Contents: image tarball, repo zip, load scripts, docs, (optional) wheelhouse for the uv route.
# Usage: ./facilitator/make_usb_kit.sh [tag]      (run docker/build.sh first)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KIT="$SCRIPT_DIR/kit"; TAG="${1:-latest}"
mkdir -p "$KIT"

"$REPO_ROOT/docker/save.sh" "$TAG" "$KIT"
if git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$REPO_ROOT" archive --format=zip --prefix=marl-workshop/ -o "$KIT/marl-workshop-repo.zip" HEAD
else
  echo "WARN: repo has no commit yet: commit first so the zip contains the code." >&2
fi
cp "$REPO_ROOT/docker/load.sh" "$REPO_ROOT/docker/load.ps1" "$KIT/"
cp "$REPO_ROOT/docs/INSTALL_DOCKER.md" "$REPO_ROOT/docs/INSTALL_UV.md" "$REPO_ROOT/docs/TROUBLESHOOTING.md" "$KIT/"
cat > "$KIT/START_HERE.txt" <<'TXT'
MARL workshop - offline kit
1. Install Docker (docs: INSTALL_DOCKER.md)           -- or skip to the Docker-free route in INSTALL_UV.md
2. Unzip marl-workshop-repo.zip somewhere (e.g. your Documents folder)
3. Load the image:   Linux/macOS:  ./marl-workshop/docker/load.sh /path/to/this/folder/marl-workshop_latest.tar.gz
                     Windows:      powershell -ExecutionPolicy Bypass -File marl-workshop\docker\load.ps1 D:\marl-workshop_latest.tar.gz
4. Start:            Linux/macOS:  ./marl-workshop/docker/run.sh        Windows: double-click marl-workshop\docker\run.bat
5. Open the printed http://localhost:8888/... link in your browser.
TXT
echo; echo "==> Kit ready in $KIT"; du -sh "$KIT"/*
