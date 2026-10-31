"""IDDPG and MADDPG for continuous-action MPE (off-policy, replay buffer, deterministic actors).

Again ONE difference - the input of agent i's critic Q_i:
    IDDPG  (critic="local")   - Q_i(o_i, a_i):           other agents are just part of the "environment",
                                                         so their changing policies look like a changing world.
    MADDPG (critic="central") - Q_i(s, a_1, ..., a_N):   the critic sees what everybody did, so the
                                                         world it models stays stationary while others learn.
Actors act from their own observation only. Each agent has its own reward (works for cooperative,
competitive and mixed tasks). `freeze` = {group: until_step}: that group does not learn until then.

Factored critics for a team that shares one reward (FACMAC-style, Peng et al. 2021) - only the
agents of group `team` (e.g. the predators); everybody else keeps the MADDPG critic:
    critic="vdn"  - Q_tot = sum_i Q_i(o_i, a_i)
    critic="qmix" - Q_tot = f_s(Q_1(o_1, a_1), ..., Q_n(o_n, a_n)), f monotonic (QMIX mixer)
Only Q_tot is fitted to the team reward. Each actor follows dQ_tot/da_i, which for VDN is exactly
dQ_i/da_i: a predator is pushed by *its own* utility, not by what the team happened to get.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from marl.common.env import MPEBatch
from marl.algos.value_decomp import QMixer, VDNMixer
from marl.common.nets import Critic, DetActor


@dataclass
class DDPGConfig:
    critic: str = "central"        # "central" (MADDPG), "local" (IDDPG), "vdn" / "qmix" (factored, for `team`)
    team: str = "predators"        # group that shares one reward (used by "vdn" / "qmix" only)
    mixer_embed: int = 32
    team_clip: float = 0.5         # grad-norm clip for the team critic + mixer
    twin_q: bool = False           # team: two critic+mixer copies, TD target uses the smaller (TD3-style)
    actor_norm: bool = False       # team actors ascend Q_tot / mean|Q_tot| (scale-free, as in TD3+BC)
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
        self.factored = cfg.critic in ("vdn", "qmix")
        self.team = [i for i in range(self.n) if self.factored and self.groups[i] == cfg.team]
        crit_in = [self.obs_dims[i] + self.act_dims[i] if self.critic_kind(i) == "local"
                   else env.state_dim + int(self.a_off[-1]) for i in range(self.n)]
        mk_actor = lambda i: DetActor(self.obs_dims[i], self.act_dims[i], cfg.hidden)
        self.actors = nn.ModuleList(mk_actor(i) for i in range(self.n)).to(self.device)
        self.critics = nn.ModuleList(Critic(crit_in[i], cfg.hidden) for i in range(self.n)).to(self.device)
        self.t_actors = nn.ModuleList(mk_actor(i) for i in range(self.n)).to(self.device)
        self.t_critics = nn.ModuleList(Critic(crit_in[i], cfg.hidden) for i in range(self.n)).to(self.device)
        self.t_actors.load_state_dict(self.actors.state_dict())
        self.t_critics.load_state_dict(self.critics.state_dict())
        self.opt_a = [torch.optim.Adam(a.parameters(), lr=cfg.lr_actor) for a in self.actors]
        self.opt_c = [torch.optim.Adam(c.parameters(), lr=cfg.lr_critic) for c in self.critics]
        if self.factored:
            mk_mix = lambda: (VDNMixer() if cfg.critic == "vdn"
                              else QMixer(len(self.team), env.state_dim, cfg.mixer_embed))
            self.mixer, self.t_mixer = mk_mix().to(self.device), mk_mix().to(self.device)
            self.t_mixer.load_state_dict(self.mixer.state_dict())
            params = [p for i in self.team for p in self.critics[i].parameters()] + list(self.mixer.parameters())
            if cfg.twin_q:                 # second copy of the team critic, indexed like self.critics
                mk_twin = lambda: nn.ModuleDict({str(i): Critic(crit_in[i], cfg.hidden) for i in self.team}).to(self.device)
                self.critics2, self.t_critics2 = mk_twin(), mk_twin()
                self.mixer2, self.t_mixer2 = mk_mix().to(self.device), mk_mix().to(self.device)
                self.t_critics2.load_state_dict(self.critics2.state_dict())
                self.t_mixer2.load_state_dict(self.mixer2.state_dict())
                params += list(self.critics2.parameters()) + list(self.mixer2.parameters())
            self.opt_team = torch.optim.Adam(params, lr=cfg.lr_critic)
            self.opt_team_actors = torch.optim.Adam([p for i in self.team for p in self.actors[i].parameters()],
                                                    lr=cfg.lr_actor)
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
    def critic_kind(self, i):
        """"central" (sees everything) or "local" (own obs + action; IDDPG and factored team members)."""
        if self.cfg.critic == "local" or i in self.team:
            return "local"
        return "central"

    def critic_input(self, i, s, a_all):
        if self.critic_kind(i) == "central":
            return torch.cat([s, a_all], -1)
        return torch.cat([self.obs_i(s, i), self.act_i(a_all, i)], -1)

    def update(self, step):
        cfg = self.cfg
        s, a, r, s2, d = self.replay.sample(cfg.batch_size)
        with torch.no_grad():
            a2 = torch.cat([self.t_actors[j](self.obs_i(s2, j)) for j in range(self.n)], -1)
        stats = {"q_loss": 0.0, "actor_loss": 0.0}
        if self.team and not self.frozen(self.team[0], step):
            stats = self.update_team(s, a, r, s2, a2)
        for i in range(self.n):
            if self.frozen(i, step) or i in self.team:
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
            pairs = [(self.actors, self.t_actors), (self.critics, self.t_critics)]
            if self.factored:
                pairs.append((self.mixer, self.t_mixer))
            if self.factored and cfg.twin_q:
                pairs += [(self.critics2, self.t_critics2), (self.mixer2, self.t_mixer2)]
            for net, tgt in pairs:
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)
        return stats

    def team_qs(self, critics, s, a_all):
        """Per-member utilities Q_i(o_i, a_i), shape (B, n_team)."""
        get = (lambda i: critics[str(i)]) if isinstance(critics, nn.ModuleDict) else (lambda i: critics[i])
        return torch.stack([get(i)(self.critic_input(i, s, a_all)) for i in self.team], -1)

    def update_team(self, s, a, r, s2, a2):
        """One TD step on Q_tot (team reward), then one joint actor step that ascends Q_tot."""
        cfg = self.cfg
        r_team = r[:, self.team[0]]                    # every team member gets this same reward
        with torch.no_grad():
            q_next = self.t_mixer(self.team_qs(self.t_critics, s2, a2), s2)
            if cfg.twin_q:
                q_next = torch.min(q_next, self.t_mixer2(self.team_qs(self.t_critics2, s2, a2), s2))
            y = r_team + cfg.gamma * q_next
        q_tot = self.mixer(self.team_qs(self.critics, s, a), s)
        q_loss = ((q_tot - y) ** 2).mean()
        if cfg.twin_q:
            q_loss = q_loss + ((self.mixer2(self.team_qs(self.critics2, s, a), s) - y) ** 2).mean()
        self.opt_team.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_([p for g in self.opt_team.param_groups for p in g["params"]], cfg.team_clip)
        self.opt_team.step()
        # actors: all team members switch to their current actions together; others as in the buffer
        new, reg = {}, 0.0
        for i in self.team:
            new[i], logits = self.actors[i](self.obs_i(s, i), return_logits=True)
            reg = reg + (logits ** 2).mean()
        a_new = torch.cat([new[j] if j in new else self.act_i(a, j) for j in range(self.n)], -1)
        q_new = self.mixer(self.team_qs(self.critics, s, a_new), s)
        if cfg.actor_norm:
            q_new = q_new / (q_new.abs().mean().detach() + 1e-6)
        actor_loss = -q_new.mean() + cfg.actor_reg * reg
        self.opt_team_actors.zero_grad()
        actor_loss.backward()
        for i in self.team:
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
        self.opt_team_actors.step()
        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item()}

    # ---------------------------------------------------------------- saving
    def policy_state(self):
        return {"algo": "ddpg", "actors": self.actors.state_dict(), "obs_dims": self.obs_dims,
                "act_dims": self.act_dims, "config": asdict(self.cfg)}

    def full_state(self):
        out = {**self.policy_state(), "critics": self.critics.state_dict()}
        if self.factored:
            out["mixer"] = self.mixer.state_dict()
        return out


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
