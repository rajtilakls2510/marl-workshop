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


def vmas_frame(env, env_index=0):
    """RGB frame of a VMAS env.

    VMAS (pyglet) draws a blank image on the first render() call of an env, so the first
    call is made twice. Use this instead of env.render(...) in notebooks.
    """
    if not getattr(env, "_marl_render_warm", False):
        env.render(mode="rgb_array", env_index=env_index)
        env._marl_render_warm = True
    return env.render(mode="rgb_array", env_index=env_index)
