"""K independent copies of an MPE task, stepped one after another in a plain Python loop.

This is NOT "vectorization" in the VMAS sense (that is Phase 5: one batched tensor simulation).
It only exists so PPO can collect several episodes per update and evaluation can average over
many episodes. It is the slow baseline that VMAS is compared against later.

All MPE episodes have the same fixed length (max_cycles) and agents never die, so the K
copies stay in lock-step: every array has a fixed shape and every env ends its episode on
the same step.

Arrays are indexed  [env, agent, feature]  with agents in `env.agents` order.
"""
import numpy as np

from marl.render import make_mpe


def group_of(env_name, agent):
    """Which team/role an agent belongs to (used for parameter sharing and freezing)."""
    prefix = agent.rsplit("_", 1)[0]
    if env_name == "tag":
        return "predators" if prefix == "adversary" else "prey"
    return {"speaker": "speaker", "listener": "listener"}.get(prefix, "agents")


class MPEBatch:
    def __init__(self, name, num_envs=8, seed=0, max_cycles=25, **env_kwargs):
        self.name = name
        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self.envs = [make_mpe(name, render_mode=None, max_cycles=max_cycles, **env_kwargs)
                     for _ in range(num_envs)]
        e = self.envs[0]
        self.agents = list(e.possible_agents)
        self.n_agents = len(self.agents)
        self.groups = [group_of(name, a) for a in self.agents]
        self.obs_dims = [e.observation_space(a).shape[0] for a in self.agents]
        self.act_dims = [e.action_space(a).shape[0] for a in self.agents]
        self.state_dim = sum(self.obs_dims)
        self._seed = seed
        self._episode = 0
        self.t = 0

    # obs is a list (one entry per agent) of arrays (K, obs_dim_i): agents may differ in size.
    def _stack(self, per_env):
        return [np.stack([d[a] for d in per_env]).astype(np.float32) for a in self.agents]

    def reset(self):
        outs = [env.reset(seed=self._seed + self._episode * self.num_envs + k)[0]
                for k, env in enumerate(self.envs)]
        self._episode += 1
        self.t = 0
        return self._stack(outs)

    def step(self, actions):
        """actions: list (per agent) of arrays (K, act_dim_i) in [0, 1].

        Returns obs (list per agent), rewards (K, n_agents), done (bool: episode over for all).
        """
        obs, rew = [], np.zeros((self.num_envs, self.n_agents), np.float32)
        for k, env in enumerate(self.envs):
            act = {a: np.clip(actions[i][k], 0.0, 1.0).astype(np.float32)
                   for i, a in enumerate(self.agents)}
            o, r, _, _, _ = env.step(act)
            obs.append(o)
            rew[k] = [r[a] for a in self.agents]
        self.t += 1
        return self._stack(obs), rew, self.t >= self.max_cycles

    @staticmethod
    def state(obs):
        """Global state = concatenation of all agents' observations (same as env.state())."""
        return np.concatenate(obs, axis=-1)

    def worlds(self):
        return [env.unwrapped.world for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()
