"""SMAClite for training: K environments stepped in parallel worker processes.

SMAClite runs ~100-200 steps/s per process (pure Python physics), so a training run spreads
its environments over several processes. This is plain process parallelism (each process
simulates its own battles), not the batched-tensor vectorisation of VMAS.

What the learners see for every agent ("agent input"):
    [observation, last action (one-hot), agent id (one-hot), t / T]
The global state passed to centralised critics/mixers is [state, t / T].
An episode ends when one army is dead or after T = 150 steps (treated as the end of the task:
the time feature tells the networks how much time is left).
"""
import multiprocessing as mp

import numpy as np

TIME_LIMIT = 150


def _episode_stats(n_agents):
    return {"return": 0.0, "length": 0, "damage": np.zeros(n_agents), "kills": np.zeros(n_agents),
            "healing": np.zeros(n_agents), "alive_steps": np.zeros(n_agents),
            "idle_steps": np.zeros(n_agents), "target_steps": np.zeros(n_agents)}


class _Env:
    """One SMAClite env + bookkeeping (lives inside a worker process)."""

    def __init__(self, map_name, seed, time_limit):
        from marl.smac_tools import make_smac

        self.env = make_smac(map_name, render=False)
        self.n, self.A, self.T = self.env.n_agents, self.env.n_actions, time_limit
        self.seed, self.episode = seed, 0

    def reset(self):
        self.obs, _ = self.env.reset(seed=self.seed + 1000 * self.episode)
        self.episode += 1
        self.t = 0
        self.last_actions = np.zeros(self.n, dtype=np.int64)
        self.stats = _episode_stats(self.n)
        return self.observe()

    def observe(self):
        avail = np.array(self.env.get_avail_actions(), dtype=np.float32)
        tt = self.t / self.T
        inp = np.concatenate([np.array(self.obs, dtype=np.float32), np.eye(self.A)[self.last_actions],
                              np.eye(self.n), np.full((self.n, 1), tt)], axis=1).astype(np.float32)
        state = np.append(self.env.get_state(), tt).astype(np.float32)
        return inp, state, avail

    def step(self, actions, rec):
        avail = self.env.get_avail_actions()
        for i, a in enumerate(actions):             # behaviour bookkeeping (before acting)
            if avail[i][1]:                          # agent alive (stop is always available when alive)
                self.stats["alive_steps"][i] += 1
                can_target = bool(np.any(avail[i][6:]))
                self.stats["target_steps"][i] += a >= 6
                self.stats["idle_steps"][i] += can_target and a < 6     # could attack/heal but didn't
        rec.reset()
        self.obs, r, done, _, info = self.env.step([int(a) for a in actions])
        self.t += 1
        self.last_actions = np.asarray(actions)
        s = self.stats
        s["return"] += r
        s["length"] += 1
        s["damage"] += rec.damage
        s["kills"] += rec.kills
        s["healing"] += rec.healing
        end = done or self.t >= self.T
        episode = None
        if end:
            episode = {**s, "won": bool(info["battle_won"]), "truncated": not done}
            nxt = self.reset()
        else:
            nxt = self.observe()
        return nxt, float(r), end, episode


def _worker(conn, map_name, seeds, time_limit):
    from marl.smac_tools import recording

    envs = [_Env(map_name, s, time_limit) for s in seeds]
    with recording(envs[0].env) as rec:          # attributes damage/healing to the acting unit
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send([e.reset() for e in envs])
            elif cmd == "step":
                conn.send([e.step(a, rec) for e, a in zip(envs, data)])
            elif cmd == "close":
                conn.close()
                return


class SMACBatch:
    """K = workers * envs_per_worker SMAClite envs; arrays are indexed [env, agent, ...]."""

    def __init__(self, map_name, workers=8, envs_per_worker=2, seed=0, time_limit=TIME_LIMIT):
        from marl.smac_tools import make_smac

        probe = make_smac(map_name, render=False)
        self.map_name, self.n_agents, self.n_actions = map_name, probe.n_agents, probe.n_actions
        self.input_dim = probe.obs_size + probe.n_actions + probe.n_agents + 1
        self.state_dim = probe.state_size + 1
        probe.close()
        self.num_envs, self.time_limit = workers * envs_per_worker, time_limit
        # fork is fastest and works from scripts and notebooks; workers never touch torch
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        self.conns, self.procs = [], []
        for w in range(workers):
            parent, child = ctx.Pipe()
            seeds = [seed * 100_003 + w * envs_per_worker + k for k in range(envs_per_worker)]
            p = ctx.Process(target=_worker, args=(child, map_name, seeds, time_limit), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        self.per_worker = envs_per_worker

    @staticmethod
    def _stack(outs):
        inp, state, avail = zip(*outs)
        return np.stack(inp), np.stack(state), np.stack(avail)

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return self._stack([o for c in self.conns for o in c.recv()])

    def step(self, actions):
        """actions (K, N) ints -> (inputs, state, avail) of the next step, team reward (K,),
        episode-ended flags (K,), list of finished-episode statistics."""
        for w, c in enumerate(self.conns):
            c.send(("step", actions[w * self.per_worker:(w + 1) * self.per_worker]))
        res = [o for c in self.conns for o in c.recv()]
        nxt = self._stack([r[0] for r in res])
        rew = np.array([r[1] for r in res], dtype=np.float32)
        end = np.array([r[2] for r in res], dtype=np.float32)
        episodes = [r[3] for r in res if r[3] is not None]
        return nxt, rew, end, episodes

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)


def summarize(episodes, healer=None):
    """Average behaviour over finished episodes -> flat dict of metrics."""
    if not episodes:
        return {}
    dmg = np.stack([e["damage"] for e in episodes])
    alive = np.maximum(np.stack([e["alive_steps"] for e in episodes]), 1)
    out = {"win_rate": float(np.mean([e["won"] for e in episodes])),
           "return": float(np.mean([e["return"] for e in episodes])),
           "length": float(np.mean([e["length"] for e in episodes])),
           "timeouts": float(np.mean([e["truncated"] for e in episodes]))}
    share = dmg.sum(0) / max(dmg.sum(), 1e-8)                    # each agent's share of team damage
    out["min_damage_share"] = float(share[[i for i in range(len(share)) if i != healer]].min())
    out["idle_frac"] = float((np.stack([e["idle_steps"] for e in episodes]) / alive).mean())
    for i, s in enumerate(share):
        out[f"damage_share_{i}"] = float(s)
    if healer is not None:
        out["heal_frac"] = float(np.mean([e["target_steps"][healer] / max(e["alive_steps"][healer], 1)
                                          for e in episodes]))
        out["healing"] = float(np.mean([e["healing"][healer] for e in episodes]))
    return out
