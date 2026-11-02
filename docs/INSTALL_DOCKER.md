# Route B — Docker image

One image contains Python, PyTorch, all the environments and the trained agents. You need Docker, this repository,
and one image download. **Not for Apple M-series Macs** (the image is built for Intel/AMD processors): use
[Route A, Codespaces](INSTALL_CODESPACES.md) instead.

## 1. Install Docker (skip if `docker run hello-world` already works)

| OS | What to install |
|---|---|
| **Windows 10/11** | [Docker Desktop](https://www.docker.com/products/docker-desktop/). It uses WSL 2 and asks you to enable it and reboot once. |
| **macOS (Intel)** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) (choose the *Intel chip* download). |
| **Linux** | Docker Engine: <https://docs.docker.com/engine/install/> (pick your distribution). |

## 2. After installing
* **Windows / macOS**: start Docker Desktop and wait until it says "running" (whale icon steady).
* **Linux**: allow your user to use Docker without `sudo`, then **log out and back in**:
  ```bash
  sudo usermod -aG docker $USER
  ```
  Check with `docker run hello-world`.
* **Optional - NVIDIA GPU** (only for the `:gpu` image, not needed for the workshop):
  install the NVIDIA driver; on Linux also the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`nvidia-container-toolkit`), then `sudo systemctl restart docker`.

## 3. Get this repository
```bash
git clone https://github.com/rajtilakls2510/marl-workshop.git
cd marl-workshop
```
No `git`? Green **Code** button on GitHub -> **Download ZIP**, and unzip it.

## 4. Pull ONE image
| Image | Size | For |
|---|---|---|
| `ghcr.io/rajtilakls2510/marl-workshop:cpu` | 2.1 GB | **everyone (recommended)** - the workshop runs fine on a CPU |
| `ghcr.io/rajtilakls2510/marl-workshop:gpu` | ~8 GB | only if you have an NVIDIA GPU and set it up in step 2 |

```bash
docker pull ghcr.io/rajtilakls2510/marl-workshop:cpu
```

## 5. Start it
| OS | Command |
|---|---|
| Linux / macOS | `./docker/run.sh` |
| Windows | double-click `docker\run.bat` (or in PowerShell: `powershell -ExecutionPolicy Bypass -File docker\run.ps1`) |

`run.sh` uses the image you pulled (`:gpu` only if Docker can actually use your GPU; if you pulled nothing it pulls `:cpu`).
When ready it prints
```
http://localhost:8888/lab?token=marl
```
Open it in your browser and run `notebooks/00_check_setup.ipynb`. Stop with **Ctrl+C** in the terminal.
Your edits are saved in the repository folder you cloned.

Options: `./docker/run.sh --cpu` (never use the GPU), `./docker/run.sh verify` (run the self-check only).
*The Windows scripts (`run.bat` / `run.ps1`) are written but not yet tested on a real Windows machine.*
