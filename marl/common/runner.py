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
        self.rows = []
        (self.dir / "config.json").write_text(json.dumps(
            {"name": name, "env": env_name, "env_kwargs": env_kwargs, **config}, indent=2, default=str))

    def evaluate_and_save(self, algo, step, extra=None, label=""):
        row = {"step": step, "wall_time": round(time.time() - self.t0, 1)}
        row.update(evaluate(algo, self.env_name, self.env_kwargs, self.eval_episodes,
                            max_cycles=self.max_cycles))
        row.update(extra or {})
        self.rows.append(row)
        self._write_csv()
        torch.save({"step": step, **algo.policy_state()}, self.dir / "ckpt" / f"step_{step:08d}.pt")
        self.record_video(algo, step, label)
        return row

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
        for clip in sorted((self.dir / "videos").glob("step_*.mp4")):
            with imageio.get_reader(clip) as r:
                frames.extend(f for f in r)
        if frames:
            save_video(frames, self.dir / "evolution.mp4", fps=15)
