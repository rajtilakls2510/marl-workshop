"""Small networks shared by all algorithms. Actions live in [0, 1] (MPE continuous)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, out_dim, hidden=(64, 64), act=nn.ReLU, out_gain=1.0):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class BetaActor(nn.Module):
    """Stochastic policy for PPO: a Beta distribution per action dimension (support [0, 1])."""

    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden, act=nn.Tanh, out_gain=0.01)

    def dist(self, obs):
        a, b = self.net(obs).chunk(2, dim=-1)
        return torch.distributions.Beta(F.softplus(a) + 1.0, F.softplus(b) + 1.0)

    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        return d.mean if deterministic else d.sample()


class DetActor(nn.Module):
    """Deterministic policy for DDPG: sigmoid squashes to [0, 1]. Also returns pre-activations
    so the actor loss can penalise saturation (as in the original MADDPG code)."""

    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden, act=nn.ReLU, out_gain=0.01)

    def forward(self, obs, return_logits=False):
        logits = self.net(obs)
        return (torch.sigmoid(logits), logits) if return_logits else torch.sigmoid(logits)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden=(64, 64)):
        super().__init__()
        self.net = mlp(in_dim, 1, hidden, act=nn.ReLU)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RunningNorm:
    """Running mean/std (for normalising value targets in PPO)."""

    def __init__(self, eps=1e-5):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x):
        x = x.detach().reshape(-1)
        b_mean, b_var, b_n = x.mean().item(), x.var(unbiased=False).item(), x.numel()
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)

    def denormalize(self, x):
        return x * (self.var ** 0.5 + 1e-8) + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
