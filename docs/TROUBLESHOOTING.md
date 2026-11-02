# Troubleshooting

Start with the self-check - it tells you which part is broken: run `notebooks/00_check_setup.ipynb`, or from a terminal
```
python verify_env.py                  # in a codespace terminal (or inside the container)
./docker/run.sh verify                # Docker route, from your own terminal
VERIFY_VERBOSE=1 python verify_env.py # full error messages
```

## Codespaces (Route A)
| Symptom | Fix |
|---|---|
| "Setting up your codespace" takes long | The first start downloads the workshop image (~2 GB) on GitHub's side: a few minutes is normal. Later starts are fast. |
| No **Select Kernel** button / no `venv` in the list | The Python and Jupyter extensions are missing or still loading: Extensions panel -> install **Python** and **Jupyter** (Microsoft), then Ctrl+Shift+P -> `Developer: Reload Window`. |
| `ModuleNotFoundError: No module named 'marl'` | Wrong kernel: **Select Kernel -> Python Environments -> `venv` (`/opt/venv/bin/python`)**. |
| `NameError: name '...' is not defined` | The kernel was restarted (or a cell was run on its own): **Run All** from the top - later cells use what the first cells import. |
| Port 8888 / JupyterLab not in the PORTS tab | Optional anyway (notebooks work in VS Code). Ctrl+Shift+P -> `Codespaces: View Creation Log` to see whether the container started. |
| "You've used all of your free Codespaces" / codespace won't start | Your account's monthly allowance is used up: use Route B, or a different GitHub account. Delete old codespaces at [github.com/codespaces](https://github.com/codespaces). |
| The codespace stopped | Codespaces stop after 30 min without activity: reopen it from [github.com/codespaces](https://github.com/codespaces); your files are still there. |

## Docker (Route B)
| Symptom | Fix |
|---|---|
| `docker: command not found` | Install Docker ([INSTALL_DOCKER.md](INSTALL_DOCKER.md)), or use [Codespaces](INSTALL_CODESPACES.md). |
| `Cannot connect to the Docker daemon` / `permission denied ... docker.sock` | Start Docker Desktop (wait for "running"). Linux: `sudo systemctl start docker` and `sudo usermod -aG docker $USER`, then log out/in. |
| Windows: "WSL 2 installation is incomplete" / virtualization disabled | Enable virtualization in BIOS, run `wsl --install` as admin, reboot. If you can't: use Codespaces. |
| Apple M-series Mac: `no matching manifest for linux/arm64` or the image crashes | The image is built for Intel/AMD processors only: use [Codespaces](INSTALL_CODESPACES.md). |
| Pull is very slow | Make sure you pulled `:cpu` (2.1 GB), not `:gpu` (~8 GB). Or use Codespaces (no download). |
| `pull access denied` / `manifest unknown` | Check the image name: `ghcr.io/rajtilakls2510/marl-workshop:cpu`. |
| `port is already allocated` (8888) | Something else uses 8888: `JUPYTER_PORT=8890 ./docker/run.sh`, then open `http://localhost:8890/lab?token=marl`. |
| Browser says "connection refused" | Wait for the `http://localhost:8888/...` line in the terminal: Jupyter is still starting. |
| `could not select device driver "nvidia"` | No GPU support in Docker - harmless: `run.sh` falls back to the CPU (or use `--cpu`). |
| Files in the repo owned by root (Linux) | Use `./docker/run.sh` (it passes your user id). If created earlier: `sudo chown -R $USER .`. |
| Out of disk space | The `:cpu` image needs ~2.1 GB plus room to unpack (~5 GB free); `:gpu` ~8 GB plus ~10 GB free. Remove an image you don't use: `docker image rm ghcr.io/rajtilakls2510/marl-workshop:gpu`. |
