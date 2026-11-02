# Multi-Agent Reinforcement Learning (MARL) — hands-on workshop

Pre-trained agents, small environments, everything runs in your browser (GitHub Codespaces) or on a laptop (Docker). **You do not need a GPU and you will not train anything for hours** - we use ready-made checkpoints and watch the agents work.

## Get set up (pick ONE route)

| | Route A: GitHub Codespaces (recommended) | Route B: Docker image |
|---|---|---|
| What you need | A browser and a GitHub account | Docker on your laptop |
| Download | **Nothing** - it runs in the cloud | 2.1 GB (`:cpu` image), or ~8 GB (`:gpu`, only for NVIDIA GPUs) |
| Works on | Any machine, including Apple M-series Macs and tablets | Windows, Linux, Intel Macs - **not Apple M-series Macs** |
| Guide | [docs/INSTALL_CODESPACES.md](docs/INSTALL_CODESPACES.md) | [docs/INSTALL_DOCKER.md](docs/INSTALL_DOCKER.md) |

Both routes contain the same Python environment and all the trained agents.

### Route A: GitHub Codespaces
> **Works on:** any operating system - Windows, macOS (Intel and Apple M-series), Linux, ChromeOS, even a tablet - in a browser.  
> **Download:** nothing. Everything runs in the cloud; you only need a GitHub account.

1. Sign in to [github.com](https://github.com) (create a free account if you don't have one).
2. Open **<https://codespaces.new/rajtilakls2510/marl-workshop>** and click **Create codespace**
   (or, on this repository's page: green **Code** button -> **Codespaces** tab -> **Create codespace on main**).
   The first start takes a few minutes; a VS Code window opens in your browser.
3. **Extensions**: the *Python* and *Jupyter* extensions (by Microsoft) install automatically. If they don't show up in the
   Extensions panel (left sidebar, four squares icon), search for **Python** and **Jupyter** and click **Install**.
4. Open `notebooks/00_check_setup.ipynb`. Top right: **Select Kernel -> Python Environments -> `venv` (`/opt/venv/bin/python`)**.
5. **Run All**. All cells green = ready.

When you are done: [github.com/codespaces](https://github.com/codespaces) -> **...** next to your codespace -> **Stop** (or **Delete**),
so it doesn't use up your free monthly hours.

### Route B: Docker image
> **Works on:** Windows 10/11, Linux, and Intel Macs. **Not on Apple M-series Macs** (M1/M2/M3/M4) - use Route A there.  
> **Download:** Docker itself, plus one image: **2.1 GB** (`:cpu`, recommended) or **~8 GB** (`:gpu`, only with an NVIDIA GPU).

1. **Install Docker**: Docker Desktop on Windows (WSL 2) and Intel Macs, Docker Engine on Linux - see [docs/INSTALL_DOCKER.md](docs/INSTALL_DOCKER.md).
2. **After installing**: Linux - `sudo usermod -aG docker $USER`, then log out and back in. Windows - let Docker Desktop enable WSL 2 and reboot.
   (Optional, NVIDIA GPU on Linux: install `nvidia-container-toolkit`.)
3. **Get this repository**:
   ```bash
   git clone https://github.com/rajtilakls2510/marl-workshop.git && cd marl-workshop
   ```
   (no `git`? green **Code** button -> **Download ZIP**, then unzip)
4. **Pull ONE image**:
   ```bash
   docker pull ghcr.io/rajtilakls2510/marl-workshop:cpu      # 2.1 GB - recommended for everyone
   docker pull ghcr.io/rajtilakls2510/marl-workshop:gpu      # ~8 GB - only if you have an NVIDIA GPU
   ```
   **Apple M-series Macs**: the image is built for Intel/AMD processors and will not work - use Route A.
5. **Start it**: `./docker/run.sh` (Windows: double-click `docker\run.bat`). It uses the image you pulled and prints
   `http://localhost:8888/lab?token=marl` - open it in your browser and run `notebooks/00_check_setup.ipynb`.

Stuck? [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) or raise your hand.

## What is inside

| Path | What |
|---|---|
| `notebooks/` | The workshop notebooks: `00` setup check, `01` MPE tour, `02` Speaker-Listener (non-stationarity), `03` Predator-Prey (arms race), `04` Coop Navigation (MAPPO, parameter sharing, scaling), `05` Predator-Prey (credit assignment: MADDPG vs VDN/QMIX critics), `06` SMAClite tour, `07` training on a harder domain: SMAClite (MAPPO, VDN, QMIX), `08` vectorization with VMAS (training speed on CPU vs GPU), `09` self-play on a two-player Pong |
| `marl/` | Small, readable PyTorch implementations of the algorithms |
| `checkpoints/` | Trained agents and training logs for every run (snapshots across training, so the notebooks can compare early and late policies). Not in git: they are **inside the workshop image**, which both routes use |
| `docker/` | Start the workshop image (`run.sh` / `run.bat`) and, for maintainers, build and publish it |
| `.devcontainer/` | The GitHub Codespaces configuration |
| `verify_env.py` | Environment self-check: `python verify_env.py` |
| `scripts/render_mpe.py` | Make a video of an MPE task: `python scripts/render_mpe.py --env tag --out videos/tag.mp4` |
| `scripts/explore_smaclite.py` | Print a SMAClite map's MDP and record a random-play video: `python scripts/explore_smaclite.py --map 2s3z` |
| `scripts/train.py` | Re-train any result yourself: `python scripts/train.py --list`, then e.g. `python scripts/train.py --run sl_maddpg` (outputs in `runs/`) |

Environments used: Multi-Agent Particle Environment (via `mpe2`/PettingZoo), SMAClite, VMAS.
