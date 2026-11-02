# Multi-Agent Reinforcement Learning (MARL) — hands-on workshop

Pre-trained agents, small environments, everything runs on a laptop. **You do not need a GPU and you will not train anything for hours** - we use ready-made checkpoints and watch the agents work.

## Get set up (pick ONE route, start it now - the download is big)

| | Route A: Docker (recommended) | Route B: no Docker (`uv`) |
|---|---|---|
| Best for | Linux, macOS, Windows with Docker Desktop | Anyone without Docker / admin rights |
| Download | ~3-4 GB image (one time) | ~3 GB with GPU torch, **~0.5 GB with `--cpu`** |
| Guide | [docs/INSTALL_DOCKER.md](docs/INSTALL_DOCKER.md) | [docs/INSTALL_UV.md](docs/INSTALL_UV.md) |

**Route A, in short**
```bash
git clone https://github.com/rajtilakls2510/marl-workshop.git && cd marl-workshop
./docker/run.sh            # Windows: double-click docker\run.bat
```
**Route B, in short**
```bash
git clone https://github.com/rajtilakls2510/marl-workshop.git && cd marl-workshop
./scripts/setup_uv.sh --cpu     # Windows: double-click scripts\setup_uv.bat
./scripts/start_jupyter.sh      # Windows: scripts\start_jupyter.bat
```
No `git`? Use the green **Code -> Download ZIP** button, or ask for the USB stick.

Then open `notebooks/00_check_setup.ipynb` and run it. All cells green = ready. Stuck? [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) or raise your hand.

## What is inside

| Path | What |
|---|---|
| `notebooks/` | The workshop notebooks: `00` setup check, `01` MPE tour, `02` Speaker-Listener (non-stationarity), `03` Predator-Prey (arms race), `04` Coop Navigation (MAPPO, parameter sharing, scaling), `05` Predator-Prey (credit assignment: MADDPG vs VDN/QMIX critics), `06` SMAClite tour, `07` training on a harder domain: SMAClite (MAPPO, VDN, QMIX), `08` vectorization with VMAS (training speed on CPU vs GPU), `09` self-play on a two-player Pong |
| `marl/` | Small, readable PyTorch implementations of the algorithms |
| `checkpoints/` | Trained agents and training logs for every run (snapshots across training, so the notebooks can compare early and late policies). **Shipped only inside the Docker image** (not in git); without Docker, re-create a run with `scripts/train.py` |
| `docker/`, `scripts/` | One-command environment setup |
| `verify_env.py` | Environment self-check: `python verify_env.py` |
| `scripts/render_mpe.py` | Make a video of an MPE task: `python scripts/render_mpe.py --env tag --out videos/tag.mp4` |
| `scripts/explore_smaclite.py` | Print a SMAClite map's MDP and record a random-play video: `python scripts/explore_smaclite.py --map 2s3z` |
| `scripts/train.py` | Re-train any result yourself: `python scripts/train.py --list`, then e.g. `python scripts/train.py --run sl_maddpg` (outputs in `runs/`) |

Environments used: Multi-Agent Particle Environment (via `mpe2`/PettingZoo), SMAClite, VMAS.
