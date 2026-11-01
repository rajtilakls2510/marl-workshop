"""Everything a training run does besides learning: evaluation, logging, checkpoints, videos.

Output layout:  runs/<run_name>/
    log.csv                  one row per evaluation (step, wall time, returns, behaviour metrics)
    ckpt/step_XXXXXXXX.pt    policy snapshots (for analysing policies across training time)
    videos/step_XXXXXXXX.mp4 one fixed-seed evaluation episode per snapshot
    evolution.mp4            all snapshot clips in order, labelled with the training step
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from marl.common.env import MPEBatch
from marl.metrics import EpisodeMetrics
from marl.render import make_mpe, rollout, save_video

REPO_ROOT = Path(__file__).resolve().parents[2]


def evaluate(algo, env_name, env_kwargs, episodes=32, seed=100_000, max_cycles=25):
    """Deterministic evaluation on `episodes` fixed-seed episodes -> dict of returns + metrics."""
    venv = MPEBatch(env_name, num_envs=episodes, seed=seed, max_cycles=max_cycles, **env_kwargs)
    obs = venv.reset()
    metrics = EpisodeMetrics(env_name, episodes)
    ret = np.zeros((episodes, venv.n_agents))
    done = False
    while not done:
        with torch.no_grad():
            acts = algo.act(obs, deterministic=True)
        obs, rew, done = venv.step(acts)
        ret += rew
        metrics.step(venv.worlds(), rew)
    out = metrics.finish(venv.worlds())
    for g in dict.fromkeys(venv.groups):
        idx = [i for i, gg in enumerate(venv.groups) if gg == g]
        out[f"return_{g}"] = float(ret[:, idx].mean())
    venv.close()
    return out


def agent_policy(algo, agents):
    """Adapter: per-agent callable for marl.render (decentralised execution)."""
    index = {a: i for i, a in enumerate(agents)}

    def policy(agent, obs):
        with torch.no_grad():
            return algo.act_agent(index[agent], obs[None].astype(np.float32), deterministic=True)[0]

    return policy


class Run:
    def __init__(self, name, env_name, env_kwargs, config, out_root=None, eval_episodes=32,
                 video_seed=7, max_cycles=25):
        self.name, self.env_name, self.env_kwargs = name, env_name, env_kwargs
        self.dir = Path(out_root or REPO_ROOT / "runs") / name
        (self.dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (self.dir / "videos").mkdir(exist_ok=True)
        self.eval_episodes, self.video_seed, self.max_cycles = eval_episodes, video_seed, max_cycles
        self.t0 = time.time()
        self.eval_seconds = 0.0           # time spent evaluating/recording (excluded from train_time)
        self.rows = []
        (self.dir / "config.json").write_text(json.dumps(
            {"name": name, "env": env_name, "env_kwargs": env_kwargs, **config}, indent=2, default=str))

    def evaluate_and_save(self, algo, step, extra=None, label=""):
        t_start = time.time()
        row = {"step": step, "wall_time": round(t_start - self.t0, 1),
               "train_time": round(t_start - self.t0 - self.eval_seconds, 1)}
        row.update(self.evaluate(algo))
        row.update(extra or {})
        self.rows.append(row)
        self._write_csv()
        torch.save({"step": step, **algo.policy_state()}, self.dir / "ckpt" / f"step_{step:08d}.pt")
        self.record_video(algo, step, label)
        self.eval_seconds += time.time() - t_start
        return row

    def train_seconds(self):
        """Seconds spent training so far (wall clock minus evaluation/recording time)."""
        return time.time() - self.t0 - self.eval_seconds

    def evaluate(self, algo):
        return evaluate(algo, self.env_name, self.env_kwargs, self.eval_episodes, max_cycles=self.max_cycles)

    def record_video(self, algo, step, label=""):
        env = make_mpe(self.env_name, max_cycles=self.max_cycles, **self.env_kwargs)
        title = f"{label} | train step {step / 1e6:.2f}M" if label else f"train step {step}"
        frames, _ = rollout(env, agent_policy(algo, env.possible_agents), seed=self.video_seed,
                            scale=0.6, title=title)
        env.close()
        save_video(frames, self.dir / "videos" / f"step_{step:08d}.mp4", fps=10)

    def _write_csv(self):
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(self.dir / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)

    def finish(self, algo, step):
        torch.save({"step": step, **algo.full_state()}, self.dir / "final.pt")
        self.make_montage()

    def make_montage(self):
        """Concatenate every snapshot clip into evolution.mp4 (the policy improving over training)."""
        import imageio.v2 as imageio

        frames = []
        for clip in sorted((self.dir / "videos").glob("step_*.mp4"), key=lambda p: int(p.stem.split("_")[1])):
            with imageio.get_reader(clip) as r:
                frames.extend(f for f in r)
        if frames:
            save_video(frames, self.dir / "evolution.mp4", fps=15)


# ---------------------------------------------------------------- VMAS (vectorized) runs
def evaluate_vmas(algo, env_kwargs, episodes=1024, seed=100_000, max_cycles=25, device="cpu"):
    """Deterministic evaluation of `episodes` VMAS Coop Navigation episodes simulated as one batch."""
    from marl.common.vmas_env import VMASBatch, collisions, spread_metrics

    env = VMASBatch("spread", num_envs=episodes, seed=seed, device=device, **{"max_cycles": max_cycles, **env_kwargs})
    obs = env.reset()
    ret = torch.zeros(episodes, env.n_agents, device=env.device)
    coll = torch.zeros(episodes, device=env.device)
    done = False
    while not done:
        with torch.no_grad():
            acts = [torch.as_tensor(a, device=env.device) for a in algo.act(obs, deterministic=True)]
        obs, rew, done = env.step(acts)
        ret += rew
        coll += collisions(env.world)
    m = spread_metrics(env.world)
    return {"coverage": float(m["coverage"].mean()), "landmark_dist": float(m["landmark_dist"].mean()),
            "collisions": float(coll.mean()), "return_agents": float(ret.mean())}


def vmas_rollout_frames(algo, env_kwargs, seed=7, max_cycles=25, title=""):
    """One episode of env copy 0 rendered with VMAS's own renderer (needs an OpenGL display, e.g. Xvfb)."""
    from marl.common.vmas_env import VMASBatch
    from marl.render import overlay
    from marl.utils import vmas_frame

    label = (lambda f: overlay(np.ascontiguousarray(f), [title])) if title else (lambda f: f)

    env = VMASBatch("spread", num_envs=1, seed=seed, **{"max_cycles": max_cycles, **env_kwargs})
    obs, frames, done = env.reset(), [], False
    while not done:
        frames.append(label(vmas_frame(env.env)))
        with torch.no_grad():
            acts = [torch.as_tensor(a) for a in algo.act([o.cpu() for o in obs], deterministic=True)]
        obs, _, done = env.step(acts)
    frames.append(label(vmas_frame(env.env)))
    return frames


class VMASRun(Run):
    """Run on the VMAS port of Coop Navigation: evaluation is batched on the training device; the video
    is rendered on the CPU (skipped, with a warning, when no OpenGL display is available)."""

    def __init__(self, *args, device="cpu", eval_episodes=1024, **kwargs):
        super().__init__(*args, eval_episodes=eval_episodes, **kwargs)
        self.device = device
        self.max_cycles = self.env_kwargs.get("max_cycles", self.max_cycles)

    def evaluate(self, algo):
        return evaluate_vmas(algo, self.env_kwargs, self.eval_episodes, max_cycles=self.max_cycles, device=self.device)

    def record_video(self, algo, step, label=""):
        title = f"{label} | train step {step / 1e6:.2f}M" if label else f"train step {step}"
        try:
            frames = vmas_rollout_frames(_CPUActor(algo), self.env_kwargs, self.video_seed, self.max_cycles, title)
        except Exception as e:                     # no display / OpenGL: keep training
            print(f"[{self.name}] video skipped: {type(e).__name__}: {e}")
            return
        save_video(frames, self.dir / "videos" / f"step_{step:08d}.mp4", fps=10)


class _CPUActor:
    """Wraps a PPO object so the (CPU) video env can query a policy that lives on the GPU."""

    def __init__(self, algo):
        self.algo = algo

    def act(self, obs, deterministic=True):
        return self.algo.act([o.to(self.algo.device) for o in obs], deterministic)
