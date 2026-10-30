#!/usr/bin/env python
"""Environment self-check for the MARL workshop.

Run:  python verify_env.py          (add --quick to skip the VMAS speed test)
Prints one PASS/FAIL line per check and a final verdict. Exit code 0 = all good.
"""
import os
import sys
import time
import traceback

# Render without a display (pygame would otherwise try to open a window).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

QUICK = "--quick" in sys.argv
results = []


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            detail = fn()
            results.append((name, True, f"{detail or ''} ({time.time() - t0:.1f}s)"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {e}"))
            if os.environ.get("VERIFY_VERBOSE"):
                traceback.print_exc()
        status, detail = ("PASS" if results[-1][1] else "FAIL"), results[-1][2]
        print(f"[{status}] {name:<28} {detail}", flush=True)
        return fn

    return deco


@check("python 3.10-3.11")
def _():
    assert (3, 10) <= sys.version_info[:2] <= (3, 11), f"found {sys.version.split()[0]}"
    return sys.version.split()[0]


@check("torch + training step")
def _():
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss = net(torch.randn(16, 8, device=dev)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU only"
    return f"torch {torch.__version__} on {name}"


@check("numpy / gymnasium")
def _():
    import gymnasium
    import numpy

    assert int(numpy.__version__.split(".")[0]) < 2, "numpy>=2 is not supported by this setup"
    return f"numpy {numpy.__version__}, gymnasium {gymnasium.__version__}"


@check("MPE: 3 tasks + render")
def _():
    from mpe2 import simple_spread_v3, simple_speaker_listener_v4, simple_tag_v3

    for mod in (simple_spread_v3, simple_tag_v3, simple_speaker_listener_v4):
        env = mod.parallel_env(render_mode="rgb_array", max_cycles=10)
        env.reset(seed=0)
        env.step({a: env.action_space(a).sample() for a in env.agents})
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[-1] == 3
        env.close()
    return "simple_spread, simple_tag, simple_speaker_listener"


@check("SMAClite: 2s3z step")
def _():
    import random

    import gymnasium as gym
    import smaclite  # noqa: F401  (registers the envs)

    env = gym.make("smaclite/2s3z-v0")
    env.reset(seed=0)
    avail = env.unwrapped.get_avail_actions()
    acts = tuple(random.choice([i for i, ok in enumerate(a) if ok]) for a in avail)
    env.step(acts)
    env.close()
    return f"{len(acts)} allied agents"


@check("VMAS: navigation x64 envs")
def _():
    import torch
    from vmas import make_env

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env("navigation", num_envs=64, device=dev, continuous_actions=True)
    env.reset()
    steps = 20 if QUICK else 100
    t0 = time.time()
    for _ in range(steps):
        env.step([env.get_random_action(a) for a in env.agents])
    sps = 64 * steps / (time.time() - t0)
    return f"{sps:,.0f} env-steps/s on {dev}"


@check("VMAS: render a frame")
def _():
    from vmas import make_env

    env = make_env("navigation", num_envs=2, device="cpu", continuous_actions=True)
    env.reset()
    from marl.utils import vmas_frame

    env.step([env.get_random_action(a) for a in env.agents])
    frame = vmas_frame(env)
    assert frame.ndim == 3 and frame.shape[-1] == 3
    assert (frame.sum(-1) < 765).mean() > 0.001, "rendered frame is blank (OpenGL/display problem?)"
    return f"frame {frame.shape[1]}x{frame.shape[0]}"


@check("checkpoint dir reachable")
def _():
    from marl.utils import ckpt_dir

    d = ckpt_dir()
    assert (d / "_smoke" / "README.txt").is_file(), f"{d} has no _smoke marker"
    return str(d)


@check("jupyter")
def _():
    import jupyterlab
    import matplotlib

    return f"jupyterlab {jupyterlab.__version__}, matplotlib {matplotlib.__version__}"


ok = all(r[1] for r in results)
print("\n" + ("ALL CHECKS PASSED - you are ready for the workshop." if ok else
      "SOME CHECKS FAILED - see docs/TROUBLESHOOTING.md or call the instructor.\n"
      "(Re-run with VERIFY_VERBOSE=1 for full tracebacks.)"))
sys.exit(0 if ok else 1)
