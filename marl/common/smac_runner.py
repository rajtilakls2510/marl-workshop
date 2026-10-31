"""Evaluation, logging, snapshots and videos for SMAClite runs (same layout as the MPE runs:
runs/<name>/log.csv, ckpt/, videos/, evolution.mp4, final.pt)."""
import time

import numpy as np
import torch

from marl.common.runner import Run
from marl.common.smac_env import TIME_LIMIT, SMACBatch, summarize


def healer_index(map_name):
    from marl.smac_tools import make_smac, team_composition

    env = make_smac(map_name, render=False)
    comp = team_composition(env)
    env.close()
    return comp.index("medivac") if "medivac" in comp else None


class AgentPolicy:
    """Adapter for playing one episode in a single (rendered) env: builds the same agent inputs
    as SMACBatch (obs, last action, id, t/T) and asks the algorithm for greedy actions."""

    def __init__(self, algo, n_agents, n_actions, time_limit=TIME_LIMIT, deterministic=True):
        self.algo, self.n, self.A, self.T = algo, n_agents, n_actions, time_limit
        self.deterministic = deterministic
        self.t, self.last = 0, np.zeros(n_agents, dtype=np.int64)

    def __call__(self, env, obs):
        inp = np.concatenate([np.array(obs, dtype=np.float32), np.eye(self.A)[self.last], np.eye(self.n),
                              np.full((self.n, 1), self.t / self.T)], axis=1).astype(np.float32)
        avail = np.array(env.get_avail_actions(), dtype=np.float32)
        with torch.no_grad():
            a = self.algo.act(inp[None], avail[None], deterministic=self.deterministic)[0]
        self.t += 1
        self.last = np.asarray(a)
        return [int(x) for x in a]


def evaluate_smac(algo, batch, episodes=32, healer=None):
    """Greedy play until `episodes` battles have finished on the given SMACBatch."""
    inp, state, avail = batch.reset()
    done = []
    while len(done) < episodes:
        with torch.no_grad():
            acts = algo.act(inp, avail, deterministic=True)
        (inp, state, avail), _, _, eps = batch.step(acts)
        done += eps
    return summarize(done[:episodes], healer)


class SMACRun(Run):
    def __init__(self, name, map_name, config, eval_workers=4, eval_episodes=32, video_seed=7, out_root=None):
        super().__init__(name, f"smac:{map_name}", {"map": map_name}, config, out_root=out_root,
                         eval_episodes=eval_episodes, video_seed=video_seed)
        self.map_name, self.healer = map_name, healer_index(map_name)
        self.eval_batch = SMACBatch(map_name, workers=eval_workers, envs_per_worker=eval_episodes // eval_workers,
                                    seed=12345)
        self.train_episodes = []

    def evaluate_and_save(self, algo, step, extra=None, label=""):
        row = {"step": step, "wall_time": round(time.time() - self.t0, 1)}
        row.update(evaluate_smac(algo, self.eval_batch, self.eval_episodes, self.healer))
        if self.train_episodes:                      # behaviour while exploring, since the last eval
            tr = summarize(self.train_episodes, self.healer)
            row.update({"train_win_rate": tr["win_rate"], "train_return": tr["return"]})
            self.train_episodes = []
        row.update(extra or {})
        self.rows.append(row)
        self._write_csv()
        torch.save({"step": step, **algo.policy_state()}, self.dir / "ckpt" / f"step_{step:08d}.pt")
        self.record_video(algo, step, label)
        return row

    def record_video(self, algo, step, label=""):
        from marl.render import save_video
        from marl.smac_tools import make_smac, rollout_smac

        env = make_smac(self.map_name)
        title = f"{label} | train step {step / 1e6:.2f}M" if label else f"train step {step}"
        frames, _ = rollout_smac(env, AgentPolicy(algo, env.n_agents, env.n_actions), seed=self.video_seed,
                                 max_steps=TIME_LIMIT, title=title)
        env.close()
        save_video(frames, self.dir / "videos" / f"step_{step:08d}.mp4", fps=6)

    def finish(self, algo, step):
        super().finish(algo, step)
        self.eval_batch.close()
