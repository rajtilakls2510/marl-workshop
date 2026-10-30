"""IDDPG and MADDPG for continuous-action MPE (off-policy, replay buffer, deterministic actors).

Again ONE difference - the input of agent i's critic Q_i:
    IDDPG  (critic="local")   - Q_i(o_i, a_i):           other agents are just part of the "environment",
                                                         so their changing policies look like a changing world.
    MADDPG (critic="central") - Q_i(s, a_1, ..., a_N):   the critic sees what everybody did, so the
                                                         world it models stays stationary while others learn.
Actors act from their own observation only. Each agent has its own reward (works for cooperative,
competitive and mixed tasks). `freeze` = {group: until_step}: that group does not learn until then.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from marl.common.env import MPEBatch
from marl.common.nets import Critic, DetActor


@dataclass
class DDPGConfig:
    critic: str = "central"        # "central" (MADDPG) or "local" (IDDPG)
    num_envs: int = 4
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    batch_size: int = 1024
    buffer_size: int = 1_000_000
    warmup_steps: int = 25_000      # random actions before learning starts
    noise: float = 0.1             # std of Gaussian exploration noise (actions in [0, 1])
    update_every: int = 8           # vector-env steps between updates (8 x 4 envs = 1 update per 32
                                    # env steps; updating every step over-trains the critic -> oscillation)
    actor_reg: float = 1e-3         # penalty on pre-sigmoid outputs (keeps actions away from saturation)
    hidden: tuple = (64, 64)
    freeze: dict = field(default_factory=dict)


class Replay:
    def __init__(self, size, state_dim, act_dim, n_agents, device):
        self.s = np.zeros((size, state_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, n_agents), np.float32)
        self.s2 = np.zeros((size, state_dim), np.float32)
        self.d = np.zeros((size,), np.float32)
        self.size, self.ptr, self.full, self.device = size, 0, False, device

    def add(self, s, a, r, s2, d):
        k = len(s)
        idx = (self.ptr + np.arange(k)) % self.size
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx] = s, a, r, s2, d
        self.ptr = (self.ptr + k) % self.size
        self.full = self.full or self.ptr < k

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n):
        idx = np.random.randint(0, len(self), n)
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


class DDPG:
    def __init__(self, env: MPEBatch, cfg: DDPGConfig, device="cpu"):
        self.cfg, self.device, self.n = cfg, torch.device(device), env.n_agents
        self.obs_dims, self.act_dims, self.groups = env.obs_dims, env.act_dims, env.groups
        self.o_off = np.concatenate([[0], np.cumsum(self.obs_dims)])   # slices of the state = obs_i
        self.a_off = np.concatenate([[0], np.cumsum(self.act_dims)])
        crit_in = [(env.state_dim + int(self.a_off[-1])) if cfg.critic == "central"
                   else self.obs_dims[i] + self.act_dims[i] for i in range(self.n)]
        mk_actor = lambda i: DetActor(self.obs_dims[i], self.act_dims[i], cfg.hidden)
        self.actors = nn.ModuleList(mk_actor(i) for i in range(self.n)).to(self.device)
        self.critics = nn.ModuleList(Critic(crit_in[i], cfg.hidden) for i in range(self.n)).to(self.device)
        self.t_actors = nn.ModuleList(mk_actor(i) for i in range(self.n)).to(self.device)
        self.t_critics = nn.ModuleList(Critic(crit_in[i], cfg.hidden) for i in range(self.n)).to(self.device)
        self.t_actors.load_state_dict(self.actors.state_dict())
        self.t_critics.load_state_dict(self.critics.state_dict())
        self.opt_a = [torch.optim.Adam(a.parameters(), lr=cfg.lr_actor) for a in self.actors]
        self.opt_c = [torch.optim.Adam(c.parameters(), lr=cfg.lr_critic) for c in self.critics]
        self.replay = Replay(cfg.buffer_size, env.state_dim, int(self.a_off[-1]), self.n, self.device)

    def obs_i(self, s, i):
        return s[:, self.o_off[i]:self.o_off[i + 1]]

    def act_i(self, a, i):
        return a[:, self.a_off[i]:self.a_off[i + 1]]

    def frozen(self, i, step):
        return step < self.cfg.freeze.get(self.groups[i], 0)

    # ---------------------------------------------------------------- acting
    def act_agent(self, i, obs_i, deterministic=False):
        with torch.no_grad():
            a = self.actors[i](torch.as_tensor(obs_i, dtype=torch.float32, device=self.device)).cpu().numpy()
        if not deterministic:
            a = np.clip(a + self.cfg.noise * np.random.randn(*a.shape), 0.0, 1.0)
        return a

    def act(self, obs, deterministic=False, step=None):
        return [self.act_agent(i, obs[i], deterministic or (step is not None and self.frozen(i, step)))
                for i in range(self.n)]

    # ---------------------------------------------------------------- learning
    def critic_input(self, i, s, a_all):
        if self.cfg.critic == "central":
            return torch.cat([s, a_all], -1)
        return torch.cat([self.obs_i(s, i), self.act_i(a_all, i)], -1)

    def update(self, step):
        cfg = self.cfg
        s, a, r, s2, d = self.replay.sample(cfg.batch_size)
        with torch.no_grad():
            a2 = torch.cat([self.t_actors[j](self.obs_i(s2, j)) for j in range(self.n)], -1)
        stats = {"q_loss": 0.0, "actor_loss": 0.0}
        for i in range(self.n):
            if self.frozen(i, step):
                continue
            # critic: regress Q_i towards r_i + gamma * Q_i'(s', a')
            with torch.no_grad():
                y = r[:, i] + cfg.gamma * (1 - d) * self.t_critics[i](self.critic_input(i, s2, a2))
            q_loss = ((self.critics[i](self.critic_input(i, s, a)) - y) ** 2).mean()
            self.opt_c[i].zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.opt_c[i].step()
            # actor: change my action to increase Q_i, others' actions as stored in the buffer
            mine, logits = self.actors[i](self.obs_i(s, i), return_logits=True)
            a_new = torch.cat([mine if j == i else self.act_i(a, j) for j in range(self.n)], -1)
            actor_loss = -self.critics[i](self.critic_input(i, s, a_new)).mean() \
                + cfg.actor_reg * (logits ** 2).mean()
            self.opt_a[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.opt_a[i].step()
            stats = {"q_loss": q_loss.item(), "actor_loss": actor_loss.item()}
        with torch.no_grad():                                         # soft target update
            for net, tgt in ((self.actors, self.t_actors), (self.critics, self.t_critics)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)
        return stats

    # ---------------------------------------------------------------- saving
    def policy_state(self):
        return {"algo": "ddpg", "actors": self.actors.state_dict(), "obs_dims": self.obs_dims,
                "act_dims": self.act_dims, "config": asdict(self.cfg)}

    def full_state(self):
        return {**self.policy_state(), "critics": self.critics.state_dict()}


def train(run, env_name, env_kwargs, cfg: DDPGConfig, total_steps, eval_every, seed=0,
          device="cpu", label="", log_fn=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MPEBatch(env_name, num_envs=cfg.num_envs, seed=seed, **env_kwargs)
    algo = DDPG(env, cfg, device)
    rng = np.random.default_rng(seed)
    step, next_eval, vec_steps = 0, 0, 0
    obs = env.reset()
    while True:
        if step >= next_eval:
            row = run.evaluate_and_save(algo, step, label=label)
            log_fn(f"[{run.name}] step {step:>9,} " + " ".join(f"{k}={v:.3g}" for k, v in row.items()
                                                                    if k != "step"))
            next_eval += eval_every
        if step >= total_steps:
            break
        if step < cfg.warmup_steps:
            acts = [rng.random((env.num_envs, d)).astype(np.float32) for d in env.act_dims]
            acts = [algo.act_agent(i, obs[i], True) if algo.frozen(i, step) else acts[i]
                    for i in range(env.n_agents)]
        else:
            acts = algo.act(obs, step=step)
        obs2, rew, done = env.step(acts)
        # MPE episodes only end because of the time limit, which is not a real terminal state:
        # store done=0 so the critic bootstraps through it (as the original MADDPG code does).
        algo.replay.add(env.state(obs), np.concatenate(acts, -1), rew, env.state(obs2),
                        np.zeros(env.num_envs, np.float32))
        obs = env.reset() if done else obs2
        step += env.num_envs
        vec_steps += 1
        if step >= cfg.warmup_steps and vec_steps % cfg.update_every == 0:
            algo.update(step)
    run.finish(algo, step)
    env.close()
    return algo
