#!/usr/bin/env bash
# Facilitator: serve the workshop kit to the room over the LAN (no internet needed by attendees).
#
#   ./facilitator/lan_serve.sh            HTTP file server on :8000 serving facilitator/kit/
#   ./facilitator/lan_serve.sh --registry also start a local Docker registry on :5000 holding the image
#
# Attendees then either download marl-workshop_latest.tar.gz + the repo zip from http://<your-ip>:8000
# (works everywhere, nothing to configure), or `docker pull <your-ip>:5000/marl-workshop`
# (needs "insecure-registries" set once: see docs/FACILITATOR.md).
# Tip: use a travel router / phone hotspot so the room is not fighting over the venue Wi-Fi.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/docker/config.sh"
KIT="$SCRIPT_DIR/kit"
mkdir -p "$KIT"

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"; IP="${IP:-<your-ip>}"

if [ "${1:-}" = "--registry" ]; then
  docker rm -f marl-registry >/dev/null 2>&1 || true
  docker run -d --restart=unless-stopped -p 5000:5000 --name marl-registry registry:2 >/dev/null
  docker tag "${LOCAL_IMAGE}:latest" "localhost:5000/${IMAGE_NAME}:latest"
  docker push "localhost:5000/${IMAGE_NAME}:latest"
  echo "==> LAN registry ready: attendees run   docker pull ${IP}:5000/${IMAGE_NAME}:latest"
  echo "    and then                            IMAGE=${IP}:5000/${IMAGE_NAME}:latest ./docker/run.sh"
fi

# Repo zip (tracked files only) so attendees without git can fetch the code.
if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$REPO_ROOT" archive --format=zip --prefix=marl-workshop/ -o "$KIT/marl-workshop-repo.zip" HEAD
fi

echo "==> Serving $KIT at   http://${IP}:8000   (Ctrl+C to stop)"
ls -lh "$KIT"
cd "$KIT" && exec python3 -m http.server 8000
