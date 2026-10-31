"""Value decomposition for SMAClite: VDN and QMIX.

Each agent has a utility Q_i(input_i, a_i) computed by ONE shared network (inputs include the
agent id). The team value is built from the individual utilities:

    VDN :  Q_tot = sum_i Q_i                                   (every agent's part simply adds up)
    QMIX:  Q_tot = f_state(Q_1, ..., Q_N), f monotonic in each Q_i (mixing weights >= 0 produced
           by hypernetworks that look at the global state)

Only Q_tot is trained against the team reward (TD learning), but the gradient flows back into each
Q_i through the sum / the mixer: agents get *individual* credit (how much their utility should
change), and acting greedily on Q_i is consistent with maximising Q_tot (monotonicity).
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.common.nets import mlp
from marl.common.smac_env import SMACBatch


@dataclass
class VDConfig:
    mixer: str = "qmix"            # "vdn" or "qmix"
    workers: int = 8
    envs_per_worker: int = 2
    buffer_per_env: int = 5000     # transitions kept per env (16 envs -> 80k transitions)
    batch_size: int = 512
    lr: float = 5e-4
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_anneal_steps: int = 100_000
    warmup_steps: int = 10_000
    target_update_every: int = 200  # gradient updates between target-network copies
    hidden: tuple = (64, 64)
    mixing_embed: int = 32
    hypernet_hidden: int = 64
    max_grad_norm: float = 10.0


class QMixer(nn.Module):
    """Q_tot = w2 . elu(W1 q + b1) + b2, with |W1|, |w2| from hypernetworks of the state."""

    def __init__(self, n_agents, state_dim, embed=32, hyper=64):
        super().__init__()
        self.n, self.embed = n_agents, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper), nn.ReLU(), nn.Linear(hyper, n_agents * embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hyper), nn.ReLU(), nn.Linear(hyper, embed))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, qs, state):                              # qs (B, N), state (B, S) -> (B,)
        w1 = self.hyper_w1(state).abs().view(-1, self.n, self.embed)
        h = F.elu(torch.bmm(qs.unsqueeze(1), w1).squeeze(1) + self.hyper_b1(state))
        w2 = self.hyper_w2(state).abs()
        return (h * w2).sum(-1) + self.hyper_b2(state).squeeze(-1)


class VDNMixer(nn.Module):
    def forward(self, qs, state):
        return qs.sum(-1)


class Replay:
    """One stream per env; the next observation of entry l is entry l + 1 of the same stream."""

    def __init__(self, K, L, N, D, S, A, device):
        self.inp = np.zeros((K, L, N, D), np.float32)
        self.state = np.zeros((K, L, S), np.float32)
        self.avail = np.zeros((K, L, N, A), np.bool_)
        self.act = np.zeros((K, L, N), np.int64)
        self.rew = np.zeros((K, L), np.float32)
        self.end = np.zeros((K, L), np.float32)
        self.K, self.L, self.p, self.size, self.device = K, L, 0, 0, device

    def add(self, inp, state, avail, act, rew, end):
        p = self.p
        self.inp[:, p], self.state[:, p], self.avail[:, p] = inp, state, avail
        self.act[:, p], self.rew[:, p], self.end[:, p] = act, rew, end
        self.p, self.size = (p + 1) % self.L, min(self.size + 1, self.L)

    def sample(self, n):
        k = np.random.randint(0, self.K, n)
        # any filled slot except the newest one (its next observation is not stored yet)
        newest = (self.p - 1) % self.L
        offs = np.random.randint(1, self.size, n)
        l = (newest - offs) % self.L
        l2 = (l + 1) % self.L
        t = lambda x: torch.as_tensor(x, device=self.device)
        return (t(self.inp[k, l]), t(self.state[k, l]), t(self.avail[k, l]), t(self.act[k, l]),
                t(self.rew[k, l]), t(self.end[k, l]), t(self.inp[k, l2]), t(self.state[k, l2]),
                t(self.avail[k, l2]))


class ValueDecomposition:
    def __init__(self, n_agents, n_actions, input_dim, state_dim, cfg: VDConfig, device="cpu"):
        self.cfg, self.n, self.A, self.device = cfg, n_agents, n_actions, torch.device(device)
        self.input_dim, self.state_dim = input_dim, state_dim
        self.algo_name = cfg.mixer
        mk_q = lambda: mlp(input_dim, n_actions, cfg.hidden, act=nn.ReLU)
        mk_mix = lambda: (QMixer(n_agents, state_dim, cfg.mixing_embed, cfg.hypernet_hidden)
                          if cfg.mixer == "qmix" else VDNMixer())
        self.q, self.q_target = mk_q().to(self.device), mk_q().to(self.device)
        self.mixer, self.mixer_target = mk_mix().to(self.device), mk_mix().to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mixer_target.load_state_dict(self.mixer.state_dict())
        self.params = list(self.q.parameters()) + list(self.mixer.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr)
        self.updates, self.eps = 0, cfg.eps_start

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def utilities(self, inputs, avail=None):
        """Q_i for every action; unavailable actions get -inf (if avail given)."""
        q = self.q(self._t(inputs))
        return q if avail is None else q.masked_fill(self._t(avail) == 0, -1e10)

    def act(self, inputs, avail, deterministic=False):
        with torch.no_grad():
            greedy = self.utilities(inputs, avail).argmax(-1).cpu().numpy()
        if deterministic:
            return greedy
        rand = (np.random.rand(*avail.shape) * avail).argmax(-1)      # uniform over available actions
        explore = np.random.rand(*greedy.shape) < self.eps
        return np.where(explore, rand, greedy)

    def update(self, replay):
        cfg = self.cfg
        inp, state, avail, act, rew, end, inp2, state2, avail2 = replay.sample(cfg.batch_size)
        q_taken = self.q(inp).gather(-1, act.unsqueeze(-1)).squeeze(-1)           # (B, N)
        q_tot = self.mixer(q_taken, state)
        with torch.no_grad():
            # double Q: pick next actions with the online net, evaluate them with the target net
            a2 = self.q(inp2).masked_fill(~avail2, -1e10).argmax(-1, keepdim=True)
            q2 = self.q_target(inp2).gather(-1, a2).squeeze(-1)
            y = rew + cfg.gamma * (1 - end) * self.mixer_target(q2, state2)
        loss = F.mse_loss(q_tot, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
        self.opt.step()
        self.updates += 1
        if self.updates % cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())
        return {"td_loss": loss.item(), "q_tot": q_tot.mean().item()}

    def policy_state(self):
        return {"algo": self.algo_name, "q": self.q.state_dict(), "mixer": self.mixer.state_dict(),
                "n_agents": self.n, "n_actions": self.A, "input_dim": self.input_dim,
                "state_dim": self.state_dim, "config": asdict(self.cfg)}

    def full_state(self):
        return self.policy_state()


def train(run, map_name, cfg: VDConfig, total_steps, eval_every, seed=0, device="cpu", label="", log_fn=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = SMACBatch(map_name, cfg.workers, cfg.envs_per_worker, seed=seed)
    algo = ValueDecomposition(env.n_agents, env.n_actions, env.input_dim, env.state_dim, cfg, device)
    replay = Replay(env.num_envs, cfg.buffer_per_env, env.n_agents, env.input_dim, env.state_dim,
                    env.n_actions, algo.device)
    inp, state, avail = env.reset()
    step, next_eval, stats = 0, 0, {}
    while True:
        if step >= next_eval:
            row = run.evaluate_and_save(algo, step, extra={**stats, "epsilon": algo.eps}, label=label)
            log_fn(f"[{run.name}] step {step:>9,} " + " ".join(
                f"{k}={v:.3g}" for k, v in row.items() if k != "step" and not k.startswith("damage_share")))
            next_eval += eval_every
        if step >= total_steps:
            break
        algo.eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_anneal_steps)
        act = algo.act(inp, avail)
        (inp2, state2, avail2), rew, end, eps = env.step(act)
        run.train_episodes += eps
        replay.add(inp, state, avail.astype(bool), act, rew, end)
        inp, state, avail = inp2, state2, avail2
        step += env.num_envs
        if step >= cfg.warmup_steps and replay.size > 1:
            stats = algo.update(replay)
    run.finish(algo, step)
    env.close()
    return algo
