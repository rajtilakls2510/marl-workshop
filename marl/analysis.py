"""Analyse trained runs: load snapshots, mix agents from different training times, and measure
what changed. These are the tools that make MARL problems visible.

    run = load_run("runs/sl_maddpg")
    pol = load_policy(run["ckpts"][-1])                 # a snapshot -> policy object
    evaluate(pol, "speaker_listener", {})               # returns + behaviour metrics
    moving_target_matrix(run, "speaker_listener")       # listener_t paired with speaker_s
    crossplay_matrix(run)                               # predators_i vs prey_j (Predator-Prey)
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from marl.common.nets import BetaActor, DetActor
from marl.common.runner import agent_policy, evaluate
from marl.render import make_mpe, rollout

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_run(name):
    """Where a named run lives: $MARL_CKPT_DIR/mpe/<name> (Docker image), <repo>/checkpoints/mpe/<name>
    (committed results) or <repo>/runs/<name> (your own training)."""
    from marl.utils import ckpt_dir

    for base in (ckpt_dir() / "mpe", REPO_ROOT / "checkpoints" / "mpe", REPO_ROOT / "runs"):
        if (base / name / "log.csv").exists():
            return base / name
    raise FileNotFoundError(f"run {name!r} not found in checkpoints/mpe/ or runs/")


def load_run(path):
    """Config, log (dict of numpy columns) and sorted snapshot paths of one run (a directory or a
    run name such as "sl_maddpg")."""
    path = Path(path)
    if not path.exists():
        path = find_run(str(path))
    with open(path / "log.csv") as f:
        rows = list(csv.DictReader(f))
    log = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}
    return {"dir": path, "config": json.loads((path / "config.json").read_text()), "log": log,
            "ckpts": sorted((path / "ckpt").glob("step_*.pt"))}


class LoadedPolicy:
    """Actors restored from a snapshot; deterministic, decentralised (each agent sees its own obs)."""

    def __init__(self, ckpt):
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.step = d["step"]
        hidden = tuple(d["config"]["hidden"])
        n = len(d["obs_dims"])
        self.actor_id = d["algo"] == "ppo" and d["config"].get("actor_id", False)
        if d["algo"] == "ppo":
            self.net_of = d["net_of"]
            first = [self.net_of.index(j) for j in range(max(self.net_of) + 1)]
            extra = n if self.actor_id else 0
            self.actors = torch.nn.ModuleList(BetaActor(d["obs_dims"][i] + extra, d["act_dims"][i], hidden)
                                              for i in first)
        else:
            self.net_of = list(range(n))
            self.actors = torch.nn.ModuleList(DetActor(d["obs_dims"][i], d["act_dims"][i], hidden) for i in range(n))
        self.actors.load_state_dict(d["actors"])
        self.algo, self.n = d["algo"], n

    def act_agent(self, i, obs_i, deterministic=True):
        with torch.no_grad():
            o = torch.as_tensor(obs_i, dtype=torch.float32)
            if self.actor_id:
                o = torch.cat([o, torch.eye(self.n)[i].expand(o.shape[:-1] + (self.n,))], -1)
            actor = self.actors[self.net_of[i]]
            return (actor.act(o, True) if self.algo == "ppo" else actor(o)).numpy()

    def act(self, obs, deterministic=True):
        return [self.act_agent(i, obs[i]) for i in range(self.n)]


class MixedPolicy:
    """Agent i is controlled by policies[i] - e.g. a listener from one snapshot and a speaker
    from another."""

    def __init__(self, policies):
        self.policies, self.n = policies, len(policies)

    def act_agent(self, i, obs_i, deterministic=True):
        return self.policies[i].act_agent(i, obs_i)

    def act(self, obs, deterministic=True):
        return [self.act_agent(i, obs[i]) for i in range(self.n)]


def load_policy(ckpt):
    return LoadedPolicy(ckpt)


def _env_of(run):
    c = run["config"]
    return c["env"], c.get("env_kwargs", {})


def video(policy, env_name, env_kwargs, seed=7, title=None, episodes=1):
    env = make_mpe(env_name, **env_kwargs)
    frames, returns = rollout(env, agent_policy(policy, env.possible_agents), seed=seed,
                              episodes=episodes, scale=0.6, title=title)
    env.close()
    return frames


def pick(ckpts, k):
    """k snapshots spread evenly over training (always includes the first and the last)."""
    idx = np.unique(np.linspace(0, len(ckpts) - 1, k).round().astype(int))
    return [ckpts[i] for i in idx]


def moving_target_matrix(run, k=6, episodes=64, metric="goal_dist"):
    """Speaker-Listener: M[t, s] = performance of the listener from snapshot t paired with the
    speaker from snapshot s. Good only on the diagonal = the listener only understands the
    speaker it co-trained with (the speaker's 'language' kept drifting)."""
    env_name, kw = _env_of(run)
    snaps = [LoadedPolicy(c) for c in pick(run["ckpts"], k)]
    M = np.zeros((len(snaps), len(snaps)))
    for t, lis in enumerate(snaps):          # agent order: speaker_0, listener_0
        for s, spk in enumerate(snaps):
            M[t, s] = evaluate(MixedPolicy([spk, lis]), env_name, kw, episodes)[metric]
    return M, [p.step for p in snaps]


def crossplay_matrix(run, k=6, episodes=64, metric="contacts"):
    """Predator-Prey: M[i, j] = predator-prey contacts when predators come from snapshot i and
    the prey from snapshot j. If later predators were simply 'better', every row would improve
    monotonically; non-monotone rows/columns show the arms race / cycling."""
    env_name, kw = _env_of(run)
    snaps = [LoadedPolicy(c) for c in pick(run["ckpts"], k)]
    n = snaps[0].n                            # agent order: adversary_0..2 (predators), agent_0 (prey)
    M = np.zeros((len(snaps), len(snaps)))
    for i, pred in enumerate(snaps):
        for j, prey in enumerate(snaps):
            M[i, j] = evaluate(MixedPolicy([pred] * (n - 1) + [prey]), env_name, kw, episodes)[metric]
    return M, [p.step for p in snaps]


def speaker_messages(run):
    """Speaker-Listener: the speaker only observes the goal colour, so its whole policy is three
    messages (one per goal). Returns steps and messages[snapshot, goal, 3]."""
    env = make_mpe("speaker_listener")
    env.reset(seed=0)
    colours = np.array([l.color for l in env.unwrapped.world.landmarks], dtype=np.float32)
    env.close()
    steps, msgs = [], []
    for c in run["ckpts"]:
        p = LoadedPolicy(c)
        steps.append(p.step)
        msgs.append(p.act_agent(0, colours))
    return np.array(steps), np.array(msgs)


def message_separation(msgs):
    """Smallest distance between the messages for two different goals (0 = goals indistinguishable)."""
    out = []
    for m in msgs:
        d = [np.linalg.norm(m[a] - m[b]) for a in range(len(m)) for b in range(a + 1, len(m))]
        out.append(min(d))
    return np.array(out)


def message_drift(msgs):
    """How much the speaker's 'language' changed between consecutive snapshots."""
    return np.array([np.abs(msgs[i + 1] - msgs[i]).mean() for i in range(len(msgs) - 1)])


# ---------------------------------------------------------------------------- plotting
def plot_curves(runs, metric, ax=None, x="step", labels=None, title=None, ylabel=None, vlines=None):
    """Learning curves of `metric` for several runs (x = "step" or "wall_time")."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    for i, r in enumerate(runs):
        lab = (labels or [None] * len(runs))[i] or r["config"].get("label", r["dir"].name)
        xs = r["log"][x] / (1e6 if x == "step" else 60.0)
        ax.plot(xs, r["log"][metric], marker="o", ms=3, label=lab)
    for v, txt in (vlines or []):
        ax.axvline(v / 1e6, color="grey", ls="--", lw=1)
        ax.text(v / 1e6, ax.get_ylim()[1], " " + txt, va="top", fontsize=8, color="grey")
    ax.set_xlabel("environment steps (millions)" if x == "step" else "wall-clock (minutes)")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_matrix(M, steps, ax=None, title="", xlabel="", ylabel="", fmt="{:.2f}", cmap="viridis_r"):
    """Heat-map with values written in the cells (used for moving-target / cross-play)."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    im = ax.imshow(M, cmap=cmap)
    ticks = [f"{s / 1e6:.2f}M" for s in steps]
    ax.set_xticks(range(len(steps)), ticks, rotation=45, fontsize=8)
    ax.set_yticks(range(len(steps)), ticks, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if (M[i, j] - M.min()) / (np.ptp(M) + 1e-9) > 0.5 else "black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def evolution_frames(run, k=4, seed=7):
    """One video per snapshot (k snapshots across training), concatenated: the policy evolving."""
    env_name, kw = _env_of(run)
    frames = []
    label = run["config"].get("label", "")
    for c in pick(run["ckpts"], k):
        p = LoadedPolicy(c)
        frames += video(p, env_name, kw, seed=seed, title=f"{label} | train step {p.step / 1e6:.2f}M")
    return frames
