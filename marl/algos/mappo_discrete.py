"""MAPPO for SMAClite (discrete actions, one team reward, agents can die).

* Actor  pi(a_i | input_i): shared MLP -> logits; unavailable actions are masked out.
* Critic V(state, agent id): centralised, used only during training.
* Every agent receives the SAME team reward. Agent i's advantage is
      A_i = (team return-to-go) - V(state, i)
  so it tells agent i whether the TEAM did better than expected - not whether agent i's own action
  helped. Logged credit diagnostic: `adv_agent_spread` = how much agents' advantages differ at the
  same moment, relative to how much advantages vary overall (0 = everyone gets exactly the same credit).
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from marl.common.nets import Critic, RunningNorm, mlp
from marl.common.smac_env import SMACBatch


@dataclass
class SMACPPOConfig:
    workers: int = 8
    envs_per_worker: int = 2
    rollout_len: int = 100        # steps per env per update (16 envs x 100 = 1600 team steps)
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 10.0
    hidden: tuple = (64, 64)


class CategoricalActor(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=(64, 64)):
        super().__init__()
        self.net = mlp(in_dim, n_actions, hidden, act=nn.ReLU, out_gain=0.01)

    def dist(self, x, avail):
        logits = self.net(x).masked_fill(avail == 0, -1e10)
        return torch.distributions.Categorical(logits=logits)


class MAPPO:
    algo_name = "mappo"

    def __init__(self, n_agents, n_actions, input_dim, state_dim, cfg: SMACPPOConfig, device="cpu"):
        self.cfg, self.n, self.A, self.device = cfg, n_agents, n_actions, torch.device(device)
        self.input_dim, self.state_dim = input_dim, state_dim
        self.actor = CategoricalActor(input_dim, n_actions, cfg.hidden).to(self.device)
        self.critic = Critic(state_dim + n_agents, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr, eps=1e-5)
        self.vnorm = RunningNorm()
        self.eye = torch.eye(n_agents, device=self.device)

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def act(self, inputs, avail, deterministic=False):
        """inputs (K, N, D), avail (K, N, A) -> actions (K, N) numpy ints."""
        d = self.actor.dist(self._t(inputs), self._t(avail))
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return a.cpu().numpy()

    def values(self, state):
        """state (..., S) -> V for every agent (..., N)."""
        s = state.unsqueeze(-2).expand(state.shape[:-1] + (self.n, state.shape[-1]))
        return self.critic(torch.cat([s, self.eye.expand(s.shape[:-1] + (self.n,))], -1))

    # ---------------------------------------------------------------- rollout
    def collect(self, env, cur, run=None):
        """rollout_len steps in every env (episodes may end and restart inside the rollout)."""
        T = self.cfg.rollout_len
        inp, state, avail = cur
        buf = {k: [] for k in ("inp", "avail", "state", "act", "logp", "val", "rew", "end", "alive")}
        for _ in range(T):
            x, av, s = self._t(inp), self._t(avail), self._t(state)
            with torch.no_grad():
                d = self.actor.dist(x, av)
                a = d.sample()
                buf["logp"].append(d.log_prob(a))
                buf["val"].append(self.vnorm.denormalize(self.values(s)))
            (inp, state, avail), rew, end, eps = env.step(a.cpu().numpy())
            if run is not None:
                run.train_episodes += eps
            buf["inp"].append(x), buf["avail"].append(av), buf["state"].append(s), buf["act"].append(a)
            buf["alive"].append(av[..., 1])                  # "stop" is available iff the agent is alive
            buf["rew"].append(self._t(rew)), buf["end"].append(self._t(end))
        rb = {k: torch.stack(v) for k, v in buf.items()}       # (T, K, ...)
        with torch.no_grad():
            last_v = self.vnorm.denormalize(self.values(self._t(state)))
        # GAE with the shared team reward; an episode end stops the bootstrap.
        adv = torch.zeros_like(rb["val"])
        gae = torch.zeros_like(last_v)
        for t in reversed(range(T)):
            nxt = last_v if t == T - 1 else rb["val"][t + 1]
            nonterminal = (1 - rb["end"][t]).unsqueeze(-1)
            delta = rb["rew"][t].unsqueeze(-1) + self.cfg.gamma * nxt * nonterminal - rb["val"][t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        rb["adv"], rb["ret"] = adv, adv + rb["val"]
        return rb, (inp, state, avail)

    @staticmethod
    def credit_spread(adv, alive):
        """Average over (time, env) of the std of advantages ACROSS alive agents, divided by the
        overall std of advantages. ~0 means every agent got (almost) the same credit."""
        m = alive > 0
        n_alive = m.sum(-1)
        ok = n_alive >= 2
        mean = (adv * m).sum(-1) / n_alive.clamp(min=1)
        var = (((adv - mean.unsqueeze(-1)) ** 2) * m).sum(-1) / n_alive.clamp(min=1)
        within = var[ok].sqrt().mean()
        overall = adv[m].std()
        return float(within / (overall + 1e-8))

    # ---------------------------------------------------------------- update
    def update(self, rb):
        cfg = self.cfg
        flat = lambda x: x.reshape(-1, *x.shape[2:])          # (T, K, ...) -> (B, ...)
        inp, avail, state, act = flat(rb["inp"]), flat(rb["avail"]), flat(rb["state"]), flat(rb["act"])
        logp_old, adv, ret, alive = flat(rb["logp"]), flat(rb["adv"]), flat(rb["ret"]), flat(rb["alive"])
        spread = self.credit_spread(adv, alive)
        adv = (adv - adv[alive > 0].mean()) / (adv[alive > 0].std() + 1e-8)
        self.vnorm.update(ret[alive > 0])
        ret_n = self.vnorm.normalize(ret)
        B = inp.shape[0]
        for _ in range(cfg.epochs):
            for mb in torch.randperm(B, device=self.device).chunk(cfg.minibatches):
                m = alive[mb]
                d = self.actor.dist(inp[mb], avail[mb])
                ratio = torch.exp(d.log_prob(act[mb]) - logp_old[mb])
                a = adv[mb]
                pg = -(torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a) * m).sum() / m.sum()
                ent = (d.entropy() * m).sum() / m.sum()
                vl = (((self.values(state[mb]) - ret_n[mb]) ** 2) * m).sum() / m.sum()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm)
                self.opt.step()
        return {"adv_agent_spread": spread, "entropy": ent.item(), "v_loss": vl.item()}

    # ---------------------------------------------------------------- saving
    def policy_state(self):
        # the critic is small and is kept in every snapshot so credit can be analysed at any time
        return {"algo": self.algo_name, "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "vnorm": self.vnorm.state_dict(), "n_agents": self.n, "n_actions": self.A,
                "input_dim": self.input_dim, "state_dim": self.state_dim, "config": asdict(self.cfg)}

    def full_state(self):
        return self.policy_state()


def train(run, map_name, cfg: SMACPPOConfig, total_steps, eval_every, seed=0, device="cpu", label="", log_fn=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = SMACBatch(map_name, cfg.workers, cfg.envs_per_worker, seed=seed)
    algo = MAPPO(env.n_agents, env.n_actions, env.input_dim, env.state_dim, cfg, device)
    cur = env.reset()
    step, next_eval, stats = 0, 0, {}
    while True:
        if step >= next_eval:
            row = run.evaluate_and_save(algo, step, extra=stats, label=label)
            log_fn(f"[{run.name}] step {step:>9,} " + " ".join(
                f"{k}={v:.3g}" for k, v in row.items() if k != "step" and not k.startswith("damage_share")))
            next_eval += eval_every
        if step >= total_steps:
            break
        rb, cur = algo.collect(env, cur, run)
        stats = algo.update(rb)
        step += cfg.rollout_len * env.num_envs
    run.finish(algo, step)
    env.close()
    return algo
