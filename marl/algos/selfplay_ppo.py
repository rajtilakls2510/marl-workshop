"""Self-play PPO for two-player Pong (marl.common.pong). No search, no hand-made opponent: the agent only
ever plays against itself.

One network plays BOTH sides (observations are mirrored, so every player sees itself on the left).

opponent="latest" - naive self-play: the current policy against itself; both sides' moves are training data.
opponent="pool"   - the learner plays against an opponent drawn from a POOL: half the games against its current
                    self, the rest against randomly chosen past snapshots (frozen). Only the learner's moves are
                    training data. Every evaluation adds the current policy to the pool.
"""
import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from marl.common.nets import Critic, RunningNorm
from marl.common.pong import PongBatch
from marl.selfplay import Actor, SnapshotPolicy


@dataclass
class SelfPlayConfig:
    opponent: str = "pool"          # "pool" or "latest"
    num_games: int = 2048
    rollout_len: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 8192           # time steps per minibatch
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (128, 128)
    latest_share: float = 0.5       # pool: share of games against the current policy
    pool_slots: int = 8             # pool: past snapshots playing in one rollout
    max_steps: int = 2000           # step limit of one game


class SelfPlayPPO:
    algo_name = "selfplay_ppo"

    def __init__(self, cfg: SelfPlayConfig, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = Actor(cfg.hidden).to(self.device)
        self.critic = Critic(9, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr, eps=1e-5)
        self.vnorm = RunningNorm()
        self.pool = []                                 # past actors' state_dicts (CPU)
        self.opp_nets = [Actor(cfg.hidden).to(self.device) for _ in range(cfg.pool_slots)]
        for n in self.opp_nets:
            n.requires_grad_(False)

    def add_to_pool(self):
        self.pool.append({k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()})

    # ---------------------------------------------------------------- rollout
    def _opponents(self, K):
        """Per game: which network plays the opponent seat (-1 = current policy, j = pool slot j)."""
        opp = torch.full((K,), -1, dtype=torch.long, device=self.device)
        if self.cfg.opponent != "pool" or not self.pool:
            return opp
        n_pool = int(round(K * (1 - self.cfg.latest_share)))
        slots = min(self.cfg.pool_slots, len(self.pool))
        for j, sd in enumerate(random.sample(self.pool, slots) if len(self.pool) >= slots else self.pool):
            self.opp_nets[j].load_state_dict(sd)
        opp[:n_pool] = torch.arange(n_pool, device=self.device) % slots
        return opp[torch.randperm(K, device=self.device)]

    def collect(self, env, obs):
        cfg, K, T = self.cfg, env.K, self.cfg.rollout_len
        opp = self._opponents(K)
        learner_left = torch.rand(K, device=self.device) < 0.5          # learner's seat, per game
        naive = cfg.opponent == "latest"
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        for _ in range(T):
            with torch.no_grad():
                d = self.actor.dist(obs)                                  # current policy, both seats (K, 2)
                act = d.sample()
                logp, val = d.log_prob(act), self.vnorm.denormalize(self.critic(obs))
                if not naive:                                             # pool: replace the opponent's moves
                    opp_seat = learner_left.long()                        # 1 = right seat is the opponent
                    opp_obs = obs[torch.arange(K), opp_seat]
                    opp_act = act[torch.arange(K), opp_seat].clone()
                    for j in range(min(cfg.pool_slots, len(self.pool))):
                        m = opp == j
                        if m.any():
                            opp_act[m] = self.opp_nets[j].dist(opp_obs[m]).sample()
                    act[torch.arange(K), opp_seat] = opp_act
            obs2, rew, done, _ = env.step(act)
            buf["obs"].append(obs), buf["act"].append(act), buf["logp"].append(logp), buf["val"].append(val)
            buf["rew"].append(rew), buf["done"].append(done.float())
            obs = obs2
        rb = {k: torch.stack(v) for k, v in buf.items()}                  # (T, K, 2, ...) / (T, K)
        with torch.no_grad():
            last_v = self.vnorm.denormalize(self.critic(obs))
        adv = torch.zeros_like(rb["val"])
        gae = torch.zeros_like(last_v)
        for t in reversed(range(T)):
            nxt = last_v if t == T - 1 else rb["val"][t + 1]
            nonterminal = (1 - rb["done"][t]).unsqueeze(-1)               # a finished game restarts: no bootstrap
            delta = rb["rew"][t] + cfg.gamma * nxt * nonterminal - rb["val"][t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        rb["adv"], rb["ret"] = adv, adv + rb["val"]
        # which (time, game, seat) samples are the learner's
        if naive:
            mask = torch.ones(K, 2, dtype=torch.bool, device=self.device)
        else:
            mask = torch.stack([learner_left, ~learner_left], 1) | (opp == -1).unsqueeze(-1)
        rb["mask"] = mask.unsqueeze(0).expand(T, K, 2)
        return rb, obs

    # ---------------------------------------------------------------- update
    def update(self, rb):
        cfg = self.cfg
        m = rb["mask"].reshape(-1)
        take = lambda x: x.reshape(-1, *x.shape[3:])[m]
        obs, act, logp_old, adv, ret = (take(rb[k]) for k in ("obs", "act", "logp", "adv", "ret"))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.vnorm.update(ret)
        ret_n = self.vnorm.normalize(ret)
        B = obs.shape[0]
        for _ in range(cfg.epochs):
            for mb in torch.randperm(B, device=self.device).split(cfg.minibatch):
                d = self.actor.dist(obs[mb])
                ratio = torch.exp(d.log_prob(act[mb]) - logp_old[mb])
                a = adv[mb]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vl = ((self.critic(obs[mb]) - ret_n[mb]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm)
                self.opt.step()
        return {"entropy": ent.item(), "v_loss": vl.item(), "samples": B}

    # ---------------------------------------------------------------- saving
    def policy_state(self):
        return {"algo": self.algo_name, "actor": self.actor.state_dict(), "config": asdict(self.cfg)}

    def full_state(self):
        return {**self.policy_state(), "critic": self.critic.state_dict(), "vnorm": self.vnorm.state_dict(),
                "pool_size": len(self.pool)}


def train(run, cfg: SelfPlayConfig, total_steps, eval_every, seed=0, device="cpu", label="", log_fn=print,
          max_minutes=None, eval_every_s=None):
    """`step` counts game steps (one step = both players move once in one game)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = PongBatch(cfg.num_games, device=device, max_steps=cfg.max_steps, seed=seed)
    algo = SelfPlayPPO(cfg, device)
    obs = env.reset()
    step, next_eval, next_eval_s, stats = 0, 0, 0.0, {}
    while True:
        out_of_time = max_minutes is not None and run.train_seconds() >= 60 * max_minutes
        due = (step >= next_eval) if eval_every_s is None else (run.train_seconds() >= next_eval_s)
        if due or step >= total_steps or out_of_time:
            row = run.evaluate_and_save(algo, step, extra=stats, label=label)
            log_fn(f"[{run.name}] step {step:>11,} " + " ".join(
                f"{k}={v:.3g}" for k, v in row.items() if k != "step" and isinstance(v, (int, float))))
            algo.add_to_pool()
            next_eval += eval_every
            next_eval_s += eval_every_s or 0
        if step >= total_steps or out_of_time:
            break
        rb, obs = algo.collect(env, obs)
        stats = algo.update(rb)
        step += cfg.num_games * cfg.rollout_len
    run.finish(algo, step)
    return algo
