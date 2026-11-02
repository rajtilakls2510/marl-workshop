"""Vectorization: simulating many copies of a multi-agent task at once (notebook 08).

* `toy_loop` / `toy_batched` - a minimal particle world (K copies x N agents) stepped two ways, small enough
  to time live on a laptop.
* `load_throughput` / `plot_throughput` - the pre-computed benchmark of the real environments
  (scripts/benchmark_vectorization.py: mpe2 loop vs VMAS on CPU / GPU).
* `wallclock_runs` / `plot_vs_time` - the same MAPPO trained with each simulator under the same
  training-time budget.
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DT, DRAG = 0.1, 0.25


# ---------------------------------------------------------------- toy particle world
def toy_loop(K, N, steps=25, seed=0):
    """K copies of N particles, one copy and one agent at a time (how a Python env loop works).
    Returns seconds per env-step (one env-step = all agents of one copy move once)."""
    rng = np.random.default_rng(seed)
    worlds = [{"pos": rng.uniform(-1, 1, (N, 2)), "vel": np.zeros((N, 2))} for _ in range(K)]
    t0 = time.perf_counter()
    for _ in range(steps):
        for w in worlds:                                   # every copy ...
            for i in range(N):                             # ... every agent
                force = rng.uniform(-1, 1, 2)
                w["vel"][i] = w["vel"][i] * (1 - DRAG) + force * DT
                w["pos"][i] = w["pos"][i] + w["vel"][i] * DT
    return (time.perf_counter() - t0) / (steps * K)


def toy_batched(K, N, steps=25, device="cpu", seed=0):
    """The same physics on tensors of shape (K, N, 2): one line moves every agent of every copy."""
    g = torch.Generator(device=device).manual_seed(seed)
    pos = torch.rand(K, N, 2, device=device, generator=g) * 2 - 1
    vel = torch.zeros(K, N, 2, device=device)
    force = torch.empty(K, N, 2, device=device)

    def step():
        nonlocal pos, vel
        force.uniform_(-1, 1, generator=g)
        vel = vel * (1 - DRAG) + force * DT
        pos = pos + vel * DT

    step()                                                 # warm-up (memory, CUDA kernels)
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device != "cpu":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (steps * K)


def toy_benchmark(Ks=(1, 16, 256, 4096), N=3, loop_max=256, devices=("cpu",)):
    """Env-steps per second of the toy world for each K (loop only up to loop_max: it gets slow)."""
    rows = []
    for K in Ks:
        if K <= loop_max:
            rows.append({"how": "loop", "K": K, "steps_per_s": 1 / toy_loop(K, N)})
        for dev in devices:
            rows.append({"how": f"batched ({dev})", "K": K, "steps_per_s": 1 / toy_batched(K, N, device=dev)})
    return rows


# ---------------------------------------------------------------- real environments: throughput
def _vmas_dir():
    from marl.utils import ckpt_dir

    for d in (ckpt_dir() / "vmas", REPO_ROOT / "checkpoints" / "vmas"):
        if d.exists():
            return d
    raise FileNotFoundError("checkpoints/vmas not found (pre-computed results ship in the Docker image)")


def load_throughput():
    """Rows of benchmark_cpu.csv + benchmark_cuda.csv (backend, device, N, K, env_steps_per_s)."""
    rows = []
    for name in ("benchmark_cpu.csv", "benchmark_cuda.csv"):
        path = _vmas_dir() / name
        if path.exists():
            with open(path) as f:
                rows += [{**r, "N": int(r["N"]), "K": int(r["K"]), "env_steps_per_s": float(r["env_steps_per_s"])}
                         for r in csv.DictReader(f)]
    return rows


STYLE = {("loop", "cpu"): ("mpe2, Python loop (CPU)", "tab:gray", "o"),
         ("vmas", "cpu"): ("VMAS, batched (CPU)", "tab:blue", "s"),
         ("vmas", "cuda"): ("VMAS, batched (GPU)", "tab:green", "^")}


def plot_throughput(rows, N, ax=None, title=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    for (backend, dev), (label, color, marker) in STYLE.items():
        pts = sorted((r["K"], r["env_steps_per_s"]) for r in rows
                     if r["backend"] == backend and r["device"] == dev and r["N"] == N)
        if pts:
            ax.plot(*zip(*pts), marker=marker, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("copies of the task simulated together (K)")
    ax.set_ylabel("env-steps per second")
    ax.set_title(title or f"Coop Navigation, N={N} agents")
    ax.grid(alpha=0.3, which="both")
    return ax


# ---------------------------------------------------------------- the same MAPPO, timed
RUN_STYLE = {"loop_spread_N{N}_cpu64": ("mpe2 loop, CPU, 64 copies", "tab:gray"),
             "vmas_spread_N{N}_cpu64": ("VMAS, CPU, 64 copies", "tab:blue"),
             "vmas_spread_N{N}_cpu1024": ("VMAS, CPU, 1024 copies", "tab:purple"),
             "vmas_spread_N{N}_gpu4096": ("VMAS, GPU, 4096 copies", "tab:green")}


def wallclock_runs(N):
    """{run name: loaded run} for the timed runs with N agents that exist."""
    from marl.analysis import load_run

    out = {}
    for pattern in RUN_STYLE:
        name = pattern.format(N=N)
        try:
            out[name] = load_run(name)
        except FileNotFoundError:
            pass
    return out


def plot_vs_time(runs, column="landmark_dist", x="train_time", ax=None, title="", ylabel=None, N=3):
    """One curve per run: `column` against training minutes (x="train_time") or env steps (x="step")."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    for name, run in runs.items():
        label, color = RUN_STYLE[name.replace(f"N{N}", "N{N}")]
        xs = run["log"][x] / (60 if x == "train_time" else 1e6)
        ax.plot(xs, run["log"][column], color=color, label=label, marker=".")
    ax.set_xlabel("training time (minutes)" if x == "train_time" else "environment steps (millions)")
    ax.set_ylabel(ylabel or column)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def reached(run, column="landmark_dist", threshold=0.3):
    """First (training minutes, env steps) at which `column` drops to `threshold`, or None."""
    log = run["log"]
    idx = np.where(log[column] <= threshold)[0]
    return None if len(idx) == 0 else (log["train_time"][idx[0]] / 60, log["step"][idx[0]])


def vmas_video(run, step=None, seed=7, title=""):
    """Frames of one VMAS Coop Navigation episode played by a run's snapshot (default: the last one), in the
    same environment settings the run was trained with (needs a display: Docker has one)."""
    from marl.analysis import load_policy
    from marl.common.runner import vmas_rollout_frames

    ckpt = run["ckpts"][-1] if step is None else min(run["ckpts"], key=lambda c: abs(int(c.stem.split("_")[1]) - step))
    return vmas_rollout_frames(load_policy(ckpt), run["config"].get("env_kwargs", {}), seed=seed, title=title)
