# Route A — Docker

One image contains Python, PyTorch, all the environments and the trained agents. You only need Docker and the workshop repo.

## 1. Install Docker (skip if `docker run hello-world` already works)

| OS | What to install |
|---|---|
| **Windows 10/11** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) (uses WSL2; it asks you to enable it and reboot once). No admin rights / can't enable WSL2? Use [Route B](INSTALL_UV.md). |
| **macOS** (Intel or Apple Silicon) | [Docker Desktop](https://www.docker.com/products/docker-desktop/), or `brew install --cask docker`. Apple-Silicon Macs automatically get the CPU build. |
| **Linux** | Docker Engine: <https://docs.docker.com/engine/install/>. Then `sudo usermod -aG docker $USER` and log out/in (so you don't need `sudo`). |

Start Docker Desktop and wait until it says "running".

*(Optional, Linux/Windows with an NVIDIA GPU: install the NVIDIA driver, and on Linux `nvidia-container-toolkit`, to use the GPU. Not needed for the workshop.)*

## 2. Get the workshop repo
```bash
git clone https://github.com/rajtilakls2510/marl-workshop.git
cd marl-workshop
```
(or download the ZIP from GitHub / take it from the USB stick and unzip it.)

## 3. Start it
| OS | Command |
|---|---|
| Linux / macOS | `./docker/run.sh` |
| Windows | double-click `docker\run.bat` (or in PowerShell: `powershell -ExecutionPolicy Bypass -File docker\run.ps1`) |

The first run downloads the image (several GB, once). When ready it prints
```
http://localhost:8888/lab?token=marl
```
Open it in your browser and run `notebooks/00_check_setup.ipynb`.
Stop with **Ctrl+C** in the terminal. Your edits are saved in the repo folder you cloned.

Options: `./docker/run.sh --cpu` (never use the GPU) and `./docker/run.sh verify` (run the self-check only).

## Offline / USB / LAN copy
If the instructor gave you `marl-workshop_latest.tar.gz`:
```bash
./docker/load.sh /path/to/marl-workshop_latest.tar.gz        # Windows: powershell -ExecutionPolicy Bypass -File docker\load.ps1 D:\marl-workshop_latest.tar.gz
./docker/run.sh
```
`run.sh` uses the locally loaded image automatically.
