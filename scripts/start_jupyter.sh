#!/usr/bin/env bash
# Start JupyterLab from the uv environment (Linux / macOS / WSL).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -x .venv/bin/jupyter ] || { echo "No .venv found: run ./scripts/setup_uv.sh first." >&2; exit 1; }
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 PYTHONPATH="$PWD"
exec .venv/bin/jupyter lab notebooks/00_check_setup.ipynb
