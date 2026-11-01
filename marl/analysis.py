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
    (packaged results, git-ignored) or <repo>/runs/<name> (your own training)."""
    from marl.utils import ckpt_dir

    for base in [b / s for b in (ckpt_dir(), REPO_ROOT / "checkpoints") for s in ("mpe", "smac", "vmas")] \
            + [REPO_ROOT / "runs"]:
        if (base / name / "log.csv").exists():
            return base / name
    raise FileNotFoundError(f"run {name!r} not found. Pre-trained runs ship only in the Docker image; "
                            f"without it, train it yourself: python scripts/train.py --run <preset>")


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
            "ckpts": sorted((path / "ckpt").glob("step_*.pt"), key=_step_of)}


def _step_of(path):
    """Training step in a snapshot/clip file name (sort numerically: past 99,999,999 steps the
    names get a 9th digit and alphabetical order breaks)."""
    return int(Path(path).stem.split("_")[1])


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
    ticks = [s if isinstance(s, str) else f"{s / 1e6:.2f}M" for s in steps]   # steps or plain labels
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


# ---------------------------------------------------------------------------- architectures
def describe_networks(run, which=("actors", "critics")):
    """Print every network of a run's final checkpoint: role, layer sizes, activation, #params."""
    path = run["dir"] / "final.pt" if isinstance(run, dict) else Path(run)
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg, algo = d["config"], d["algo"]
    agents = make_mpe(run["config"]["env"], **run["config"].get("env_kwargs", {})).possible_agents \
        if isinstance(run, dict) else [f"agent {i}" for i in range(len(d["obs_dims"]))]
    net_of = d.get("net_of", list(range(len(agents))))
    act = {"ppo": ("tanh", "ReLU"), "ddpg": ("ReLU", "ReLU")}[algo]
    head = {"ppo": "softplus+1 -> Beta(alpha, beta) per action dim", "ddpg": "sigmoid -> action in [0,1]"}[algo]
    crit_kind = "V (state value)" if algo == "ppo" else "Q (action value)"
    rows = []
    for kind in which:
        if kind not in d:
            continue
        sd = d[kind]
        n_nets = len({k.split(".")[0] for k in sd})
        for j in range(n_nets):
            ws = [sd[k] for k in sorted(sd) if k.startswith(f"{j}.") and k.endswith("weight")]
            sizes = [ws[0].shape[1]] + [w.shape[0] for w in ws]
            params = sum(v.numel() for k, v in sd.items() if k.startswith(f"{j}."))
            users = [a for a, n in zip(agents, net_of) if n == j] if algo == "ppo" else [agents[j]]
            role = "actor" if kind == "actors" else f"critic {crit_kind}"
            out = head if kind == "actors" else "scalar"
            rows.append((role, ", ".join(users), " -> ".join(map(str, sizes)),
                         act[0] if kind == "actors" else act[1], out, params))
    w = [max(len(str(r[i])) for r in rows + [("network", "used by", "layers", "hidden act.", "output", "params")])
         for i in range(6)]
    hdr = ("network", "used by", "layers", "hidden act.", "output", "params")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for r in rows:
        print("  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)))
    print(f"total parameters: {sum(r[5] for r in rows):,}")


def load_benchmark(device="cpu"):
    """Parameter-sharing benchmark table written by scripts/benchmark_sharing.py."""
    name = "benchmark_sharing.csv" if device == "cpu" else f"benchmark_sharing_{device}.csv"
    for base in (REPO_ROOT / "checkpoints" / "mpe",):
        if (base / name).exists():
            with open(base / name) as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(name)


def plot_benchmark(rows, axes=None, title=""):
    """Parameters, action time and update time vs number of agents, shared vs separate networks."""
    import matplotlib.pyplot as plt

    axes = axes if axes is not None else plt.subplots(1, 3, figsize=(13, 3.4))[1]
    for shared, lab in (("True", "shared network"), ("False", "separate networks")):
        r = [x for x in rows if x["shared"] == shared]
        n = [int(x["N"]) for x in r]
        for ax, key, yl in zip(axes, ("params", "act_ms", "update_s"),
                               ("parameters", "choose actions (ms)", "one PPO update (s)")):
            ax.plot(n, [float(x[key]) for x in r], marker="o", label=lab)
            ax.set_xlabel("number of agents N")
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend()
    if title:
        axes[1].set_title(title)
    return axes


def snapshot(run, step):
    """Policy from the snapshot closest to `step` environment steps."""
    ckpt = min(run["ckpts"], key=lambda c: abs(int(c.stem.split("_")[1]) - step))
    return LoadedPolicy(ckpt)


class FrozenAgent:
    """Wraps a policy so that agent `i` stands still (all-zero action = no movement). Used for
    counterfactuals: how much does the team lose without this agent?"""

    def __init__(self, policy, i):
        self.policy, self.i, self.n = policy, i, policy.n

    def act_agent(self, j, obs_j, deterministic=True):
        a = self.policy.act_agent(j, obs_j)
        return np.zeros_like(a) if j == self.i else a

    def act(self, obs, deterministic=True):
        return [self.act_agent(j, obs[j]) for j in range(self.n)]


def predator_stats(policy, episodes=64, seed=500, max_cycles=25):
    """Predator-Prey, per predator: touches of the prey per episode, mean distance to the prey and
    fraction of time spent off the arena (|x| or |y| > 1), and off the arena while the prey is more
    than 1.0 away (`off_far`: not chasing it). All predators get the SAME reward
    (10 x all touches), so these numbers are the only way to see who actually does the work."""
    from marl.common.env import MPEBatch

    env = MPEBatch("tag", num_envs=episodes, seed=seed, max_cycles=max_cycles)
    obs = env.reset()
    n_pred = env.groups.count("predators")
    touch, dist, off, off_far = (np.zeros(n_pred) for _ in range(4))
    for _ in range(max_cycles):
        obs, _, _ = env.step(policy.act(obs))
        for w in env.worlds():
            prey = [a for a in w.agents if not a.adversary][0]
            for i, p in enumerate(a for a in w.agents if a.adversary):
                d = np.linalg.norm(p.state.p_pos - prey.state.p_pos)
                touch[i] += d < p.size + prey.size
                dist[i] += d
                off[i] += np.abs(p.state.p_pos).max() > 1.0
                off_far[i] += np.abs(p.state.p_pos).max() > 1.0 and d > 1.0
    env.close()
    T = episodes * max_cycles
    return {"touches": touch / episodes, "dist": dist / T, "off_arena": off / T, "off_far": off_far / T}


def predator_roles(run, episodes=64, seed=500):
    """predator_stats for every snapshot of a Predator-Prey run, plus two summary curves:
    `laziest_share` = smallest predator's share of the team's touches (1/3 = equal work, 0 = one
    predator never touches the prey) and `max_off_arena` = the most-absent predator's time off-map."""
    rows = [predator_stats(LoadedPolicy(c), episodes, seed) for c in run["ckpts"]]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    total = out["touches"].sum(1)
    out["laziest_share"] = np.where(total > 0, out["touches"].min(1) / np.maximum(total, 1e-9), np.nan)
    out["max_off_arena"] = out["off_arena"].max(1)
    out["team_touches"] = total
    out["steps"] = np.array([int(c.stem.split("_")[1]) for c in run["ckpts"]])
    return out


def predator_contributions(policy, episodes=64, seed=500):
    """Counterfactual: team touches with all predators minus team touches when predator i stands
    still. A predator whose absence costs nothing contributes nothing (a 'lazy' agent)."""
    full = predator_stats(policy, episodes, seed)["touches"].sum()
    n_pred = len(predator_stats(policy, 1, seed)["touches"])
    return full, np.array([full - predator_stats(FrozenAgent(policy, i), episodes, seed)["touches"].sum()
                           for i in range(n_pred)])


# ---------------------------------------------------------------- Predator-Prey credit assignment
TAG_CREDIT_ALGOS = {"maddpg": "MADDPG", "facvdn": "FAC-VDN", "facqmix": "FAC-QMIX"}


def tag_seed_runs(algo):
    """The 3 seeds of a Predator-Prey run: tag_<algo>, tag_<algo>_s1, tag_<algo>_s2."""
    return [load_run(f"tag_{algo}{sfx}") for sfx in ("", "_s1", "_s2")]


def _summary_rows(name):
    torch.set_num_threads(1)
    run = load_run(name)
    rows = []
    for c in run["ckpts"]:
        st = predator_stats(LoadedPolicy(c), episodes=128)
        t = st["touches"]
        rows.append({"run": name, "step": int(c.stem.split("_")[1]), "team_touches": t.sum(),
                     "laziest_share": t.min() / t.sum() if t.sum() > 0 else np.nan,
                     "max_off_arena": st["off_arena"].max(), "max_off_far": st["off_far"].max(),
                     **{f"touches_{i}": v for i, v in enumerate(t)},
                     **{f"off_arena_{i}": v for i, v in enumerate(st["off_arena"])}})
    return rows


def compute_tag_credit_summary(out=None, processes=9):
    """Per-predator statistics for every snapshot of every seed (slow: a few minutes) -> CSV that
    the notebook reads with load_tag_credit_summary()."""
    from concurrent.futures import ProcessPoolExecutor

    names = [f"tag_{a}{sfx}" for a in TAG_CREDIT_ALGOS for sfx in ("", "_s1", "_s2")]
    with ProcessPoolExecutor(processes) as ex:
        rows = [r for rs in ex.map(_summary_rows, names) for r in rs]
    out = Path(out or REPO_ROOT / "checkpoints" / "mpe" / "tag_credit_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def load_tag_credit_summary():
    """{algo: {column: array (seeds, snapshots)}} from checkpoints/mpe/tag_credit_summary.csv."""
    from marl.utils import ckpt_dir

    path = next(p for p in (ckpt_dir() / "mpe" / "tag_credit_summary.csv",
                            REPO_ROOT / "checkpoints" / "mpe" / "tag_credit_summary.csv") if p.exists())
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for algo in TAG_CREDIT_ALGOS:
        per_seed = []
        for sfx in ("", "_s1", "_s2"):
            rs = sorted((r for r in rows if r["run"] == f"tag_{algo}{sfx}"), key=lambda r: int(r["step"]))
            per_seed.append({k: np.array([float(r[k]) for r in rs]) for k in rs[0] if k != "run"})
        out[algo] = {k: np.stack([d[k] for d in per_seed]) for k in per_seed[0]}
    return out


def plot_tag_credit(summary, column, ax=None, title=None, ylabel=None, colors=None, freeze=500_000):
    """Mean over seeds (line) and min-max over seeds (band) of one summary column."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    colors = colors or {"maddpg": "tab:blue", "facvdn": "tab:orange", "facqmix": "tab:green"}
    for algo, d in summary.items():
        x, y = d["step"][0] / 1e6, d[column]
        ax.plot(x, np.nanmean(y, 0), color=colors[algo], label=TAG_CREDIT_ALGOS[algo])
        ax.fill_between(x, np.nanmin(y, 0), np.nanmax(y, 0), color=colors[algo], alpha=0.15)
    if freeze:
        ax.axvline(freeze / 1e6, color="gray", ls="--", lw=1)
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel(ylabel or column)
    ax.set_title(title or column)
    ax.grid(alpha=0.3)
    return ax


def tag_paths(policy, seed=0, max_cycles=25):
    """Positions of every agent during one episode: array (T+1, n_agents, 2), agent order as env."""
    from marl.common.env import MPEBatch

    env = MPEBatch("tag", num_envs=1, seed=seed, max_cycles=max_cycles)
    obs = env.reset()
    pos = [[a.state.p_pos.copy() for a in env.worlds()[0].agents]]
    for _ in range(max_cycles):
        obs, _, _ = env.step(policy.act(obs))
        pos.append([a.state.p_pos.copy() for a in env.worlds()[0].agents])
    env.close()
    return np.array(pos)


def plot_tag_paths(policy, seed=0, ax=None, title=""):
    """One episode seen from above: predators red/orange (circle = start, x = end), prey green,
    arena dashed."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    P = tag_paths(policy, seed)
    reds = ["#d62728", "#ff7f0e", "#8c1c13"]
    for i in range(P.shape[1]):
        c = reds[i] if i < P.shape[1] - 1 else "#2ca02c"
        lab = f"predator {i}" if i < P.shape[1] - 1 else "prey"
        ax.plot(P[:, i, 0], P[:, i, 1], "-", color=c, lw=1.8, label=lab)
        ax.plot(P[0, i, 0], P[0, i, 1], "o", color=c)
        ax.plot(P[-1, i, 0], P[-1, i, 1], "x", color=c, ms=9, mew=2)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ls="--", color="gray"))
    lim = max(1.3, np.abs(P).max() + 0.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def tag_algo_crossplay(step=2_000_000, episodes=64):
    """Predators of each algorithm vs the prey of each algorithm (mean over the 3 x 3 seed pairs):
    team touches, and the most-absent predator's time off the arena."""
    names = list(TAG_CREDIT_ALGOS)
    pols = {a: [snapshot(r, step) for r in tag_seed_runs(a)] for a in names}
    touches, off = np.zeros((3, 3)), np.zeros((3, 3))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            st = [predator_stats(MixedPolicy([pp] * 3 + [qq]), episodes) for pp in pols[a] for qq in pols[b]]
            touches[i, j] = np.mean([s["touches"].sum() for s in st])
            off[i, j] = np.mean([s["off_arena"].max() for s in st])
    return touches, off, [TAG_CREDIT_ALGOS[a] for a in names]
