#!/usr/bin/env bash
# Docker-free setup (Linux / macOS / WSL): creates ./.venv with Python 3.10 and installs everything.
#
#   ./scripts/setup_uv.sh                 auto-detect GPU (NVIDIA -> CUDA torch, else CPU)
#   ./scripts/setup_uv.sh --cpu           force the small CPU-only torch (~200 MB instead of ~3 GB)
#   ./scripts/setup_uv.sh --wheelhouse D  offline install from a folder of wheels (USB kit)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="auto"; WHEELHOUSE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) BACKEND="cpu" ;;
    --wheelhouse) WHEELHOUSE="${2:?--wheelhouse needs a directory}"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found: installing it (single static binary, no admin rights needed)"
  if [ -n "$WHEELHOUSE" ] && [ -x "$WHEELHOUSE/uv" ]; then
    mkdir -p "$HOME/.local/bin" && cp "$WHEELHOUSE/uv" "$HOME/.local/bin/uv"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null || { echo "ERROR: uv installation failed; see docs/INSTALL_UV.md (pip fallback)." >&2; exit 1; }

echo "==> Creating .venv (Python 3.10)"
uv venv --python 3.10 --allow-existing .venv

echo "==> Installing packages (torch backend: $BACKEND) -- this takes a few minutes"
if [ -n "$WHEELHOUSE" ]; then
  uv pip install --python .venv/bin/python --no-index --find-links "$WHEELHOUSE" -r requirements.txt
else
  uv pip install --python .venv/bin/python --torch-backend "$BACKEND" -r requirements.txt
fi

echo "==> Verifying"
.venv/bin/python verify_env.py

cat <<'MSG'

Setup done. Start the notebooks with:
    ./scripts/start_jupyter.sh
MSG
