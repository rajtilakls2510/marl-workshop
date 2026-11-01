"""Small helpers shared by all notebooks."""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_device():
    """Return a torch device: CUDA if available, else CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ckpt_dir():
    """Directory holding the pre-trained checkpoints.

    Resolution order: $MARL_CKPT_DIR (set inside the Docker image, where the
    checkpoints are baked in) -> <repo>/checkpoints.
    """
    env = os.environ.get("MARL_CKPT_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    return _REPO_ROOT / "checkpoints"


def ckpt_path(*parts):
    """Path to a checkpoint, e.g. ckpt_path("mpe", "simple_spread", "mappo", "final.pt")."""
    return ckpt_dir().joinpath(*parts)


def ensure_display(default=":99"):
    """Make sure an X display exists for VMAS/pyglet rendering.

    The Docker image starts a virtual display (Xvfb :99) in its entrypoint, but processes not started by the
    entrypoint (VS Code / Codespaces notebook kernels, `docker exec`) do not inherit DISPLAY. Point them at
    it - and start Xvfb if it is installed but not running. On a normal desktop DISPLAY is already set.
    """
    import shutil
    import subprocess
    import time

    display = os.environ.get("DISPLAY") or default
    num = display.lstrip(":").split(".")[0]
    socket = Path(f"/tmp/.X11-unix/X{num}")
    if os.environ.get("DISPLAY") and (not num.isdigit() or socket.exists() or not shutil.which("Xvfb")):
        return os.environ["DISPLAY"]                     # real desktop / remote X / nothing we can do
    if not socket.exists() and shutil.which("Xvfb"):
        subprocess.Popen(["Xvfb", display, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(50):
            if socket.exists():
                break
            time.sleep(0.1)
    if socket.exists():
        os.environ["DISPLAY"] = display
    return os.environ.get("DISPLAY")


def vmas_frame(env, env_index=0):
    """RGB frame of a VMAS env.

    VMAS (pyglet) draws a blank image on the first render() call of an env, so the first
    call is made twice. Use this instead of env.render(...) in notebooks.
    """
    ensure_display()
    if not getattr(env, "_marl_render_warm", False):
        env.render(mode="rgb_array", env_index=env_index)
        env._marl_render_warm = True
    return env.render(mode="rgb_array", env_index=env_index)
