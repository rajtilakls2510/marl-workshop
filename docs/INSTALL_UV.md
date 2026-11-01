# Route B — no Docker, using `uv`

[`uv`](https://docs.astral.sh/uv/) is a single-file Python package manager. It downloads Python 3.10 for you, so you don't need to install or fix anything on your system, and it creates an isolated environment in the repo folder (`.venv/`). Nothing is installed system-wide, and no admin rights are needed.

## 1. Get the workshop repo
```bash
git clone https://github.com/rajtilakls2510/marl-workshop.git
cd marl-workshop
```
(or *Code -> Download ZIP* on GitHub / the USB stick, and unzip it.)

## 2. Run the setup script
| OS | Command |
|---|---|
| Linux / macOS / WSL | `./scripts/setup_uv.sh --cpu` |
| Windows | double-click `scripts\setup_uv.bat` (installs the small CPU build). For an NVIDIA GPU build run in PowerShell: `powershell -ExecutionPolicy Bypass -File scripts\setup_uv.ps1` |

* `--cpu` / `-Cpu`: small download (~0.5 GB). **Recommended unless you have an NVIDIA GPU and want to use it.** Without the flag, uv detects an NVIDIA GPU and installs CUDA PyTorch (~3 GB).
* The script installs uv if missing, creates `.venv`, installs `requirements.txt`, and runs `verify_env.py`. It ends with `ALL CHECKS PASSED`.

## 3. Start the notebooks
| OS | Command |
|---|---|
| Linux / macOS / WSL | `./scripts/start_jupyter.sh` |
| Windows | `scripts\start_jupyter.bat` |

A browser tab opens with `00_check_setup.ipynb`. Run it top to bottom.

> **Pre-trained agents are only in the Docker image.** With the uv setup, notebooks `00` and `01` work as-is; the later notebooks load trained runs, which you can re-create yourself with `python scripts/train.py --run <name>` (they land in `runs/`; `--list` shows the names). Some runs take an hour or more on a laptop - use the Docker image for the workshop if you can.

## Doing it by hand (what the script does)
```bash
# install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh        (Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex)
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements.txt     # Windows: .venv\Scripts\python.exe
uv pip install --python .venv/bin/python --reinstall-package pygame-ce pygame-ce==2.5.8   # must come last (see requirements.txt)
.venv/bin/python verify_env.py
```
Plain `pip` also works if you already have Python 3.10:
```bash
python3.10 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install --force-reinstall --no-deps pygame-ce==2.5.8
```

## Offline (USB/LAN wheelhouse)
```bash
./scripts/setup_uv.sh --wheelhouse /path/to/wheelhouse-linux       # Windows: scripts\setup_uv.ps1 -Wheelhouse D:\wheelhouse-windows
```
