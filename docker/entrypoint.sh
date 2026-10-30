#!/usr/bin/env bash
# Container entrypoint.
#   (no args) / "jupyter"  -> start JupyterLab on :8888 (token: $JUPYTER_TOKEN)
#   anything else          -> run that command (e.g. `python verify_env.py`)
set -euo pipefail

SEED=/opt/marl-workshop
WORK=/workspace

# Seed an empty /workspace (fresh named volume or empty bind mount) from the baked-in copy.
if [ -d "$WORK" ] && [ -z "$(ls -A "$WORK" 2>/dev/null)" ]; then
  echo "==> /workspace is empty: copying the workshop materials into it"
  cp -a "$SEED"/. "$WORK"/ 2>/dev/null || cp -R "$SEED"/. "$WORK"/
fi

# Virtual display for VMAS/pyglet rendering (see Dockerfile). Notebook kernels inherit DISPLAY.
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/dev/null 2>&1 &
export DISPLAY=:99
for _ in $(seq 1 50); do [ -e /tmp/.X11-unix/X99 ] && break; sleep 0.1; done

if [ "${1:-jupyter}" != "jupyter" ]; then
  cd "$WORK"
  exec "$@"
fi

cd "$WORK"
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"==> GPU detected: {torch.cuda.get_device_name(0)}  (torch {torch.__version__})")
else:
    print(f"==> No GPU visible inside the container: running on CPU (torch {torch.__version__}). That is fine for the workshop.")
PY

cat <<EOF2

==========================================================
  MARL workshop is starting.
  Open this in your browser:

      http://localhost:8888/lab?token=${JUPYTER_TOKEN}

  Start with notebooks/00_check_setup.ipynb
  Stop the container with Ctrl+C.
==========================================================

EOF2

exec jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser \
  --ServerApp.root_dir="${WORK}" \
  --ServerApp.default_url="/lab/tree/notebooks/00_check_setup.ipynb" \
  --IdentityProvider.token="${JUPYTER_TOKEN}"
