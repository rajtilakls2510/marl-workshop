"""IPPO and MAPPO for continuous-action MPE.

They are the same algorithm except for ONE line: what the critic is allowed to see.
    IPPO  (critic="local")   - V_i(o_i):  each agent judges its situation from its own view
    MAPPO (critic="central") - V_i(s):    during training the critic sees the global state
Actors always act from their own observation only (decentralised execution).

share=True : agents of the same group (e.g. all spread agents) use one actor and one critic
             (an agent one-hot is appended so the shared critic can tell agents apart).
share=False: every agent has its own networks.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from marl.common.env import MPEBatch
from marl.common.nets import BetaActor, Critic, RunningNorm


@dataclass
class PPOConfig:
    critic: str = "central"        # "central" (MAPPO) or "local" (IPPO)
    share: bool = True             # parameter sharing within a group
    actor_id: bool = False         # append an agent one-hot to the (shared) actor's observation
    time_feature: bool = True      # critic also sees t/T: episodes are cut at T steps, so how much
                                   # reward is still to come depends on the time left
    num_envs: int = 64             # episodes collected per update
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatches: int = 2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 10.0
    hidden: tuple = (64, 64)


class PPO:
    def __init__(self, env: MPEBatch, cfg: PPOConfig, device="cpu"):
        self.cfg, self.device, self.n = cfg, torch.device(device), env.n_agents
        self.obs_dims, self.act_dims, self.groups = env.obs_dims, env.act_dims, env.groups
        # Which network each agent uses: shared per group, or one per agent.
        keys = self.groups if cfg.share else list(range(self.n))
        self.net_of = [list(dict.fromkeys(keys)).index(k) for k in keys]
        n_nets = max(self.net_of) + 1
        first = [self.net_of.index(j) for j in range(n_nets)]           # an agent using net j
        crit_in = [(env.state_dim if cfg.critic == "central" else self.obs_dims[i]) + self.n
                   + int(cfg.time_feature) for i in first]
        self.T = env.max_cycles
        extra = self.n if cfg.actor_id else 0
        self.actors = nn.ModuleList(BetaActor(self.obs_dims[i] + extra, self.act_dims[i], cfg.hidden)
                                    for i in first).to(self.device)
        self.critics = nn.ModuleList(Critic(crit_in[j], cfg.hidden) for j in range(n_nets)).to(self.device)
        self.opt = torch.optim.Adam(list(self.actors.parameters()) + list(self.critics.parameters()),
                                    lr=cfg.lr, eps=1e-5)
        self.vnorm = [RunningNorm() for _ in range(self.n)]
        self.eye = torch.eye(self.n, device=self.device)

    def actor_in(self, i, o):
        """What actor i sees: its observation (+ its one-hot id if cfg.actor_id)."""
        if not self.cfg.actor_id:
            return o
        return torch.cat([o, self.eye[i].expand(o.shape[:-1] + (self.n,))], -1)

    # ---------------------------------------------------------------- acting
    def act_agent(self, i, obs_i, deterministic=False):
        o = torch.as_tensor(obs_i, dtype=torch.float32, device=self.device)
        return self.actors[self.net_of[i]].act(self.actor_in(i, o), deterministic).cpu().numpy()

    def act(self, obs, deterministic=False):
        return [self.act_agent(i, obs[i], deterministic) for i in range(self.n)]

    def critic_input(self, i, obs_i, state, t):
        x = state if self.cfg.critic == "central" else obs_i
        parts = [x, self.eye[i].expand(x.shape[:-1] + (self.n,))]
        if self.cfg.time_feature:
            parts.append((t / self.T).expand(x.shape[:-1] + (1,)))
        return torch.cat(parts, dim=-1)

    def value(self, i, obs_i, state, t):
        return self.critics[self.net_of[i]](self.critic_input(i, obs_i, state, t))

    # ---------------------------------------------------------------- learning
    def collect(self, env):
        """One episode in each of the K env copies -> rollout tensors (time, env, ...)."""
        T, dev = env.max_cycles, self.device
        obs = env.reset()
        buf = {"obs": [[] for _ in range(self.n)], "act": [[] for _ in range(self.n)],
               "logp": [], "val": [], "rew": [], "state": [], "time": []}
        done = False
        while not done:
            o = [torch.as_tensor(x, device=dev) for x in obs]
            s = torch.as_tensor(env.state(obs), device=dev)
            tt = torch.full((env.num_envs, 1), float(env.t), device=dev)
            with torch.no_grad():
                dists = [self.actors[self.net_of[i]].dist(self.actor_in(i, o[i])) for i in range(self.n)]
                acts = [d.sample().clamp(1e-6, 1 - 1e-6) for d in dists]
                logp = torch.stack([d.log_prob(a).sum(-1) for d, a in zip(dists, acts)], -1)
                val = torch.stack([self.vnorm[i].denormalize(self.value(i, o[i], s, tt))
                                   for i in range(self.n)], -1)
            obs, rew, done = env.step([a.cpu().numpy() for a in acts])
            for i in range(self.n):
                buf["obs"][i].append(o[i])
                buf["act"][i].append(acts[i])
            buf["logp"].append(logp), buf["val"].append(val), buf["state"].append(s), buf["time"].append(tt)
            buf["rew"].append(torch.as_tensor(rew, device=dev))
        rb = {k: torch.stack(v) for k, v in buf.items() if k not in ("obs", "act")}
        rb["obs"] = [torch.stack(v) for v in buf["obs"]]
        rb["act"] = [torch.stack(v) for v in buf["act"]]
        # GAE; every episode ends at the time limit, treated as terminal (standard for MPE).
        adv = torch.zeros_like(rb["rew"])
        last = torch.zeros_like(rb["rew"][0])
        for t in reversed(range(T)):
            next_v = rb["val"][t + 1] if t + 1 < T else torch.zeros_like(last)
            delta = rb["rew"][t] + self.cfg.gamma * next_v - rb["val"][t]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * last
            adv[t] = last
        rb["ret"], rb["adv"] = adv + rb["val"], adv
        return rb, rb["rew"].sum(0).mean(0).cpu().numpy()

    def update(self, rb):
        cfg, n = self.cfg, self.n
        flat = lambda x: x.reshape(-1, *x.shape[2:])                     # (T, K, ...) -> (T*K, ...)
        obs, act = [flat(x) for x in rb["obs"]], [flat(x) for x in rb["act"]]
        state, logp_old, adv, ret = flat(rb["state"]), flat(rb["logp"]), flat(rb["adv"]), flat(rb["ret"])
        time = flat(rb["time"])
        adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)                  # per agent
        for i in range(n):
            self.vnorm[i].update(ret[:, i])
        ret_n = torch.stack([self.vnorm[i].normalize(ret[:, i]) for i in range(n)], -1)
        B = state.shape[0]
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        for _ in range(cfg.epochs):
            for mb in torch.randperm(B, device=self.device).chunk(cfg.minibatches):
                pg, vl, ent = 0.0, 0.0, 0.0
                for i in range(n):
                    d = self.actors[self.net_of[i]].dist(self.actor_in(i, obs[i][mb]))
                    ratio = torch.exp(d.log_prob(act[i][mb]).sum(-1) - logp_old[mb, i])
                    a = adv[mb, i]
                    pg = pg - torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                    ent = ent + d.entropy().sum(-1).mean()
                    vl = vl + ((self.value(i, obs[i][mb], state[mb], time[mb]) - ret_n[mb, i]) ** 2).mean()
                loss = (pg + cfg.vf_coef * vl - cfg.ent_coef * ent) / n
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats = {"pg_loss": pg.item() / n, "v_loss": vl.item() / n, "entropy": ent.item() / n}
        return stats

    def parameters(self):
        return list(self.actors.parameters()) + list(self.critics.parameters())

    # ---------------------------------------------------------------- saving
    def policy_state(self):
        return {"algo": "ppo", "net_of": self.net_of, "actors": self.actors.state_dict(),
                "obs_dims": self.obs_dims, "act_dims": self.act_dims, "config": asdict(self.cfg)}

    def full_state(self):
        return {**self.policy_state(), "critics": self.critics.state_dict(),
                "vnorm": [v.state_dict() for v in self.vnorm]}


def train(run, env_name, env_kwargs, cfg: PPOConfig, total_steps, eval_every, seed=0,
          device="cpu", label="", log_fn=print):
    """Train IPPO/MAPPO; `run` is a marl.common.runner.Run that evaluates/saves/records."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MPEBatch(env_name, num_envs=cfg.num_envs, seed=seed, **env_kwargs)
    algo = PPO(env, cfg, device)
    steps_per_update = cfg.num_envs * env.max_cycles
    step, next_eval = 0, 0
    while True:
        if step >= next_eval:
            row = run.evaluate_and_save(algo, step, label=label)
            log_fn(f"[{run.name}] step {step:>9,} " + " ".join(f"{k}={v:.3g}" for k, v in row.items()
                                                                    if k not in ("step",)))
            next_eval += eval_every
        if step >= total_steps:
            break
        rb, train_ret = algo.collect(env)
        algo.update(rb)
        step += steps_per_update
    run.finish(algo, step)
    env.close()
    return algo
