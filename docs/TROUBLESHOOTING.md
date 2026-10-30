# Troubleshooting

Always start with the self-check - it tells you which part is broken:
```
python verify_env.py                  # uv route (inside .venv)
./docker/run.sh verify                # docker route
VERIFY_VERBOSE=1 python verify_env.py # full error messages
```

## Docker
| Symptom | Fix |
|---|---|
| `docker: command not found` | Install Docker ([INSTALL_DOCKER.md](INSTALL_DOCKER.md)) or switch to [Route B](INSTALL_UV.md). |
| `Cannot connect to the Docker daemon` / `permission denied ... docker.sock` | Start Docker Desktop (wait for "running"). Linux: `sudo systemctl start docker` and `sudo usermod -aG docker $USER`, then log out/in. |
| Windows: "WSL 2 installation is incomplete" / virtualization disabled | Enable virtualization in BIOS, run `wsl --install` as admin, reboot. If you can't: Route B. |
| `pull access denied` / `manifest unknown` | The public image isn't reachable from your network: ask for the USB/LAN copy and use `docker/load.sh`. |
| Pull is very slow | Share the instructor's LAN copy. Or use Route B with `--cpu` (0.5 GB). |
| `port is already allocated` (8888) | Something else uses 8888: `JUPYTER_PORT=8890 ./docker/run.sh`, then open `http://localhost:8890/lab?token=marl`. |
| Browser says "connection refused" | The container printed the link only after it finished starting; wait for the `http://localhost:8888/...` banner. |
| `could not select device driver "nvidia"` | No GPU support in Docker - harmless: `run.sh` falls back to CPU automatically (or use `--cpu`). |
| Files in the repo owned by root (Linux) | Use `./docker/run.sh` (it passes your uid). If created earlier: `sudo chown -R $USER .`. |
| Apple Silicon: "no matching manifest for linux/arm64" | The image was published without an arm64 variant; ask the instructor for the arm64 build. |

## uv / Python
| Symptom | Fix |
|---|---|
| `uv: command not found` right after install | Open a new terminal (the installer edits your PATH), or `export PATH="$HOME/.local/bin:$PATH"`. |
| Windows: "running scripts is disabled" | Use the `.bat` wrappers, or `powershell -ExecutionPolicy Bypass -File ...`. |
| Install fails with network/SSL errors | Corporate proxy: set `HTTPS_PROXY`; or use the offline wheelhouse from the USB stick. |
| `ModuleNotFoundError` | You're not using the venv: use `scripts/start_jupyter.*`, or activate `.venv`. |
| SMAClite check fails: `No module named 'smaclite'` | The GitHub zip download was blocked: ask for the wheelhouse (it contains a prebuilt smaclite wheel). |
| `numpy` 2.x errors | The project is pinned to numpy 1.26; reinstall with `uv pip install -r requirements.txt`. |
| `VMAS: render` fails with pyglet / OpenGL error (Linux, no desktop, WSL) | Install OpenGL + a virtual display: `sudo apt install xvfb libgl1 libglu1-mesa`, then run `xvfb-run -a jupyter lab ...`. (The Docker image already does this.) |
| Pygame/SDL errors about a display | Set `SDL_VIDEODRIVER=dummy` (the scripts already do). |
| Out of disk space | The CUDA install needs ~8 GB free; use `--cpu` (~2 GB). |
