"""Analyse SMAClite runs: restore policies/critics, replay episodes, and measure credit assignment.

Credit diagnostic. For every step of an episode and every alive agent we compare
    * the agent's own contribution  - damage it dealt (or healing it did) at that step, and
    * the agent's learning signal  -
        MAPPO : its advantage A_i = (team return-to-go) - V(s, i)       [driven by the TEAM outcome]
        VDN/QMIX: its own utility gap Q_i(o_i, a_i) - mean_a Q_i(o_i, a) [an INDIVIDUAL value]
`credit_alignment` = correlation, across the agents alive at the same moment, between signal and
contribution (averaged over moments where agents differ). Near 0 = credit does not track who did what.
"""
import numpy as np
import torch

from marl.algos.mappo_discrete import MAPPO, SMACPPOConfig
from marl.algos.value_decomp import ValueDecomposition, VDConfig
from marl.common.smac_runner import AgentPolicy
from marl.smac_tools import make_smac, rollout_smac, team_composition


def run_map(run):
    c = run["config"]
    return c.get("map") or c["env_kwargs"]["map"]


def load_smac_algo(ckpt):
    """Rebuild a MAPPO or VDN/QMIX object (with critic/mixer when the checkpoint has them)."""
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = (d["n_agents"], d["n_actions"], d["input_dim"], d["state_dim"])
    if d["algo"] == "mappo":
        cfg = SMACPPOConfig(**{k: (tuple(v) if k == "hidden" else v) for k, v in d["config"].items()})
        algo = MAPPO(*args, cfg)
        algo.actor.load_state_dict(d["actor"])
        if "critic" in d:
            algo.critic.load_state_dict(d["critic"])
            algo.vnorm.load_state_dict(d["vnorm"])
    else:
        cfg = VDConfig(**{k: (tuple(v) if k == "hidden" else v) for k, v in d["config"].items()})
        algo = ValueDecomposition(*args, cfg)
        algo.q.load_state_dict(d["q"])
        algo.mixer.load_state_dict(d["mixer"])
    algo.step = d.get("step")
    return algo


def smac_video(algo, map_name, seed=7, title=None):
    env = make_smac(map_name)
    frames, records = rollout_smac(env, AgentPolicy(algo, env.n_agents, env.n_actions), seed=seed, title=title)
    env.close()
    return frames, records


def smac_evolution(run, k=4, seed=7):
    """The same battle replayed with snapshots from increasingly late in training."""
    from marl.analysis import pick

    frames, label, m = [], run["config"].get("label", ""), run_map(run)
    for c in pick(run["ckpts"], k):
        algo = load_smac_algo(c)
        f, _ = smac_video(algo, m, seed, title=f"{label} | train step {algo.step / 1e6:.2f}M")
        frames += f
    return frames


class _Recorder:
    """Policy wrapper that also keeps the inputs/states/avail it saw (for computing signals)."""

    def __init__(self, algo, env, deterministic=True):
        self.pol = AgentPolicy(algo, env.n_agents, env.n_actions, deterministic=deterministic)
        self.inputs, self.states, self.avails, self.actions = [], [], [], []

    def __call__(self, env, obs):
        t, last = self.pol.t, self.pol.last
        n, A = self.pol.n, self.pol.A
        inp = np.concatenate([np.array(obs, np.float32), np.eye(A)[last], np.eye(n),
                              np.full((n, 1), t / self.pol.T)], 1).astype(np.float32)
        self.inputs.append(inp)
        self.states.append(np.append(env.get_state(), t / self.pol.T).astype(np.float32))
        self.avails.append(np.array(env.get_avail_actions(), np.float32))
        a = self.pol(env, obs)
        self.actions.append(a)
        return a


def episode_signals(algo, map_name, seed=0, gamma=0.99, lam=0.95, stochastic=True):
    """Play one episode (with the policy's own exploration unless stochastic=False: SMAClite always
    starts from the same positions, so greedy episodes are nearly identical); return per-step,
    per-agent learning signals and contributions."""
    env = make_smac(map_name, render=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(algo, "eps"):
        algo.eps = 0.05
    rec = _Recorder(algo, env, deterministic=not stochastic)
    _, records = rollout_smac(env, rec, seed=seed, render=False)
    env.close()
    inp, st, av = (torch.as_tensor(np.stack(x)) for x in (rec.inputs, rec.states, rec.avails))
    act = torch.as_tensor(np.array(rec.actions))
    T = len(records)
    inp, st, av, act = inp[:T], st[:T], av[:T], act[:T]
    with torch.no_grad():
        if isinstance(algo, MAPPO):
            v = algo.vnorm.denormalize(algo.values(st)).numpy()               # (T, N)
            r = np.array([x["reward"] for x in records])
            adv, gae = np.zeros_like(v), np.zeros(v.shape[1])
            for t in reversed(range(T)):
                nxt = v[t + 1] if t + 1 < T else 0.0
                gae = r[t] + gamma * nxt - v[t] + gamma * lam * gae
                adv[t] = gae
            signal = adv
        else:
            q = algo.q(inp)                                                     # (T, N, A)
            q_taken = q.gather(-1, act.unsqueeze(-1)).squeeze(-1)
            q_mean = (q * av).sum(-1) / av.sum(-1).clamp(min=1)
            signal = (q_taken - q_mean).numpy()
    contrib = np.array([x["damage"] + x["healing"] for x in records])          # (T, N)
    alive = av[..., 1].numpy() > 0
    return {"signal": signal, "contribution": contrib, "alive": alive, "records": records}


def credit_alignment(sig):
    """Mean within-step correlation (across alive agents) between signal and contribution."""
    cors = []
    for s, c, a in zip(sig["signal"], sig["contribution"], sig["alive"]):
        if a.sum() >= 3 and c[a].std() > 0 and s[a].std() > 0:
            cors.append(np.corrcoef(s[a], c[a])[0, 1])
    return float(np.mean(cors)) if cors else float("nan"), len(cors)


def credit_report(run, episodes=10):
    """credit_alignment averaged over several greedy episodes of the run's final policy."""
    algo = load_smac_algo(run["dir"] / "final.pt")
    vals = [credit_alignment(episode_signals(algo, run_map(run), seed=s))[0] for s in range(episodes)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def plot_episode_credit(sig, map_name, ax=None, title=""):
    """Per-agent learning signal (lines) with markers where that agent dealt damage / healed."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    names = team_composition(make_smac(map_name, render=False))
    T, N = sig["signal"].shape
    for i in range(N):
        s = np.where(sig["alive"][:, i], sig["signal"][:, i], np.nan)
        line, = ax.plot(range(T), s, lw=1.2, label=f"{i} {names[i]}")
        hit = sig["contribution"][:, i] > 0
        ax.scatter(np.flatnonzero(hit), s[hit], s=18, color=line.get_color(), edgecolor="k", zorder=3)
    ax.set_xlabel("step")
    ax.set_title(title or "learning signal per agent (dots: that agent dealt damage / healed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return ax


def describe_smac_networks(run):
    """Print the networks of a run's final checkpoint: layer sizes and parameter counts."""
    d = torch.load(run["dir"] / "final.pt", map_location="cpu", weights_only=False)
    rows = []

    def add(name, sd, note):
        ws = [v for k, v in sd.items() if k.endswith("weight") and v.dim() == 2]
        sizes = " -> ".join(str(s) for s in [ws[0].shape[1]] + [w.shape[0] for w in ws]) if "hyper" not in note else note
        rows.append((name, sizes if "hyper" not in note else "hypernetworks of the state", note if "hyper" not in note else "",
                     sum(v.numel() for v in sd.values())))

    n, A = d["n_agents"], d["n_actions"]
    if d["algo"] == "mappo":
        add("actor (shared by all agents)", d["actor"], f"ReLU; logits over {A} actions, unavailable ones masked")
        add("critic V(state, agent id)", d["critic"], "ReLU; one value per agent, training only")
    else:
        add("agent utility Q_i (shared)", d["q"], f"ReLU; one value per action ({A})")
        if d["algo"] == "qmix":
            add("QMIX mixer", d["mixer"], "hyper")
        else:
            rows.append(("VDN mixer", "Q_tot = sum_i Q_i", "no parameters", 0))
    w = [max(len(str(r[i])) for r in rows) for i in range(4)]
    print(f"inputs: agent = obs + last action + agent id + t/T ({d['input_dim']}), state + t/T ({d['state_dim']}); "
          f"{n} agents, {A} actions")
    for r in rows:
        print("  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)))
    print(f"total parameters: {sum(r[3] for r in rows):,}")


def signal_similarity(sig):
    """Mean correlation OVER TIME between the learning signals of every pair of agents (only steps
    where both are alive). Close to 1 = all agents are told the same thing at the same moment."""
    s, alive = sig["signal"], sig["alive"]
    cors = []
    for i in range(s.shape[1]):
        for j in range(i + 1, s.shape[1]):
            both = alive[:, i] & alive[:, j]
            if both.sum() >= 8 and s[both, i].std() > 0 and s[both, j].std() > 0:
                cors.append(np.corrcoef(s[both, i], s[both, j])[0, 1])
    return float(np.mean(cors)) if cors else float("nan")


def similarity_report(algo, map_name, episodes=12):
    vals = [signal_similarity(episode_signals(algo, map_name, seed=s)) for s in range(episodes)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


# ---------------------------------------------------------------------------- multi-seed views
def seed_runs(map_name, algo):
    """All seeds of one algorithm on one map (e.g. smac_2s3z_qmix, smac_2s3z_qmix_s1, ...)."""
    from marl.analysis import load_run

    base, out = f"smac_{map_name.lower()}_{algo}", []
    for suffix in ("", "_s1", "_s2", "_s3", "_s4"):
        try:
            out.append(load_run(base + suffix))
        except FileNotFoundError:
            pass
    return out


def plot_seeds(runs, metric, ax, label, color):
    """Each seed as a thin line, the mean as a thick line."""
    steps = runs[0]["log"]["step"] / 1e6
    vals = []
    for r in runs:
        y = r["log"][metric]
        n = min(len(y), len(steps))
        ax.plot(r["log"]["step"][:n] / 1e6, y[:n], color=color, lw=0.8, alpha=0.35)
        vals.append(y[:n])
    n = min(len(v) for v in vals)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)       # step-0 rows have no training stats yet
        mean = np.nanmean([v[:n] for v in vals], 0)
    ax.plot(steps[:n], mean, color=color, lw=2.2,
            label=f"{label} ({len(runs)} seed{'s' if len(runs) > 1 else ''})")
    ax.set_xlabel("environment steps (millions)")
    ax.grid(alpha=0.3)


def compute_credit_summary(out_csv, episodes=8):
    """Signal similarity of every packaged SMAClite run's final policy -> CSV (slow: replays episodes)."""
    import csv
    from pathlib import Path

    from marl.analysis import REPO_ROOT

    rows = []
    for d in sorted((REPO_ROOT / "checkpoints" / "smac").glob("smac_*")):
        algo = load_smac_algo(d / "final.pt")
        m, s, n = similarity_report(algo, run_map({"config": __import__("json").loads((d / "config.json").read_text())}),
                                    episodes)
        rows.append({"run": d.name, "algo": algo.algo_name, "similarity": round(m, 3), "std": round(s, 3), "episodes": n})
        print(rows[-1], flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def load_credit_summary():
    import csv

    from marl.analysis import REPO_ROOT

    with open(REPO_ROOT / "checkpoints" / "smac" / "credit_summary.csv") as f:
        return list(csv.DictReader(f))
