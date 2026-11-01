"""Self-play helpers for the hand-made Pong (notebook 08): policies, batched matches, evaluation, the run logger.

A *policy* is a function obs (B, 9) -> actions (B,). Trained snapshots become policies with `SnapshotPolicy`.
"""
import time

import torch
import torch.nn as nn

from marl.common.nets import mlp
from marl.common.pong import OBS_DIM, PongBatch, follow_ball_policy, random_policy
from marl.common.runner import Run


class Actor(nn.Module):
    """obs (..., 9) -> logits over {down, stay, up}."""

    def __init__(self, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(OBS_DIM, 3, hidden, act=nn.ReLU, out_gain=0.01)

    def forward(self, obs):
        return self.net(obs)

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))


class SnapshotPolicy:
    """A trained actor as a policy: samples from its action distribution (greedy=True: most likely action)."""

    def __init__(self, actor, greedy=False, name=""):
        self.actor, self.greedy, self.name = actor, greedy, name

    @torch.no_grad()
    def __call__(self, obs):
        d = self.actor.dist(obs)
        return d.probs.argmax(-1) if self.greedy else d.sample()


def load_snapshot(path, device="cpu", greedy=False):
    d = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(tuple(d["config"]["hidden"])).to(device)
    actor.load_state_dict(d["actor"])
    actor.eval()
    return SnapshotPolicy(actor, greedy, name=f"{d['step'] / 1e6:.0f}M")


BASELINES = {"random": random_policy, "bot": follow_ball_policy}


@torch.no_grad()
def match(policy_a, policy_b, games=512, device="cpu", seed=0, max_steps=2000):
    """`games` complete games of A vs B, A on the left in half of them and on the right in the other half.
    Returns A's score: wins + 0.5 * draws, divided by games, plus game statistics."""
    env = PongBatch(games, device=device, seed=seed, max_steps=max_steps)
    obs = env.reset()
    a_left = torch.arange(games, device=env.device) < games // 2
    finished = torch.zeros(games, dtype=torch.bool, device=env.device)
    result = torch.zeros(games, device=env.device)
    points = hits = 0
    rally_sum = torch.zeros((), device=env.device)
    for _ in range(max_steps + 1):
        left_is_a = a_left.unsqueeze(-1)
        obs_a = torch.where(left_is_a, obs[:, 0], obs[:, 1])
        obs_b = torch.where(left_is_a, obs[:, 1], obs[:, 0])
        act_a, act_b = policy_a(obs_a), policy_b(obs_b)
        actions = torch.stack([torch.where(a_left, act_a, act_b), torch.where(a_left, act_b, act_a)], 1)
        obs, _, done, info = env.step(actions)
        live = ~finished
        points += int((info["point"] & live).sum())
        hits += int((info["hit"] & live).sum())
        rally_sum += (info["rally_at_point"] * (info["point"] & live)).sum()
        new = done & live
        if new.any():
            fs = info["final_score"]                                   # rows of the games that finished now
            idx = done.nonzero().squeeze(-1)
            score_a = torch.where(a_left[idx], fs[:, 0], fs[:, 1])
            score_b = torch.where(a_left[idx], fs[:, 1], fs[:, 0])
            r = (score_a > score_b).float() + 0.5 * (score_a == score_b).float()
            keep = live[idx]
            result[idx[keep]] = r[keep]
            finished |= new
        if finished.all():
            break
    return {"score": float(result.mean()), "rally": float(rally_sum) / max(points, 1),
            "hits_per_game": hits / games}


class SelfPlayRun(Run):
    """Run logger for self-play Pong: evaluation = matches against the random player and the scripted
    follow-the-ball bot (evaluation only); video = one game against the bot."""

    def __init__(self, name, config, device="cpu", eval_games=512, out_root=None):
        super().__init__(name, "pong", {}, config, out_root=out_root, eval_episodes=eval_games)
        self.device = device

    def evaluate(self, algo):
        me = SnapshotPolicy(algo.actor)
        out = {}
        for opp, pol in BASELINES.items():
            m = match(me, pol, self.eval_episodes, self.device, seed=12345)
            out[f"score_vs_{opp}"] = m["score"]
            if opp == "bot":
                out["rally_vs_bot"] = m["rally"]
        return out

    def record_video(self, algo, step, label=""):
        """Two clips per snapshot (kept for the notebook): the agent against ITSELF (what self-play games look
        like at this stage) and against the follow-the-ball bot."""
        from marl.pong_render import play_game
        from marl.render import save_video

        actor = Actor(algo.cfg.hidden)
        actor.load_state_dict({k: v.cpu() for k, v in algo.actor.state_dict().items()})
        me, tag = SnapshotPolicy(actor), f"{step / 1e6:.0f}M steps"
        for kind, opp, names in (("self", me, (f"agent ({tag})", f"agent ({tag})")),
                                 ("bot", follow_ball_policy, (f"agent ({tag})", "follow-the-ball bot"))):
            frames, _ = play_game(me, opp, names=names, seed=7, max_steps=900, every=2)
            save_video(frames, self.dir / "videos" / f"step_{step:010d}_{kind}.mp4", fps=30)

    def make_montage(self):
        """evolution_self.mp4 and evolution_bot.mp4: the snapshot clips in training order."""
        import imageio.v2 as imageio

        from marl.render import save_video

        for kind in ("self", "bot"):
            frames = []
            for clip in sorted((self.dir / "videos").glob(f"step_*_{kind}.mp4"), key=lambda p: int(p.stem.split("_")[1])):
                with imageio.get_reader(clip) as r:
                    frames.extend(f for f in r)
            if frames:
                save_video(frames, self.dir / f"evolution_{kind}.mp4", fps=30)

    def train_seconds(self):
        return time.time() - self.t0 - self.eval_seconds


# ---------------------------------------------------------------- analysis: tournaments, Elo, shot style
def tournament(players, games=256, device="cpu", seed=0):
    """Round robin. players: {name: policy}. Returns (names, S) with S[i, j] = score of i against j
    (win = 1, draw = 0.5), each pair playing `games` games with seats swapped half the time."""
    import numpy as np

    names = list(players)
    S = np.full((len(names), len(names)), 0.5)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = match(players[names[i]], players[names[j]], games, device, seed=seed + 1000 * i + j)["score"]
            S[i, j], S[j, i] = s, 1 - s
    return names, S


def fit_elo(S, games=256, anchor=0, iters=3000, lr=20.0):
    """Elo ratings from a score matrix (Bradley-Terry fit): P(i beats j) = 1 / (1 + 10^((r_j - r_i) / 400)).
    Ratings are shifted so that player `anchor` (e.g. the random player) has 0."""
    import numpy as np

    S = (np.asarray(S) * games + 0.5) / (games + 1)   # one pseudo-draw per pair: 0% / 100% stay finite
    n = len(S)
    r = np.zeros(n)
    off = ~np.eye(n, dtype=bool)
    for _ in range(iters):
        p = 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))
        grad = ((S - p) * off).sum(1)                  # observed minus expected score
        r += lr * grad / (n - 1)
        r -= r[anchor]
    return r


@torch.no_grad()
def shot_stats(policy, opponent=follow_ball_policy, games=512, steps=1500, device="cpu", seed=0):
    """How `policy` plays (on the left side) against `opponent`: share of its returns that are steep
    (more than 30 degrees), average return angle, rally length (hits per point)."""
    import math

    env = PongBatch(games, device=device, seed=seed)
    obs = env.reset()
    angles, points, hits = [], 0, 0
    for _ in range(steps):
        a = torch.stack([policy(obs[:, 0]), opponent(obs[:, 1])], 1)
        obs, _, _, info = env.step(a)
        mine = info["hit"] & (env.vel[:, 0] > 0)          # ball now moving right = the left player just hit it
        if mine.any():
            v = env.vel[mine]
            angles.append(torch.atan2(v[:, 1].abs(), v[:, 0]).cpu())
        points += int(info["point"].sum())
        hits += int(info["hit"].sum())
    ang = torch.cat(angles) if angles else torch.zeros(0)
    return {"steep_share": float((ang > math.radians(30)).float().mean()) if len(ang) else float("nan"),
            "mean_angle_deg": float(ang.mean() * 180 / math.pi) if len(ang) else float("nan"),
            "hits_per_point": hits / max(points, 1)}


# ---------------------------------------------------------------- notebook 08 helpers
def _table(rows, header):
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)))


def print_pong_mdp():
    """The MDP of the two-player Pong, printed from the constants in marl.common.pong."""
    import math

    from marl.common import pong as P

    print("PLAYERS: 2 (left, right) - zero-sum. One network plays both sides.\n")
    print("OBSERVATION (9 numbers per player, MIRRORED: every player sees itself on the left)")
    _table([(0, "own paddle y", "x2 -> [-0.8, 0.8]"), (1, "opponent paddle y", "x2"), (2, "ball x", "[-1, 1], own side < 0"),
            (3, "ball y", "x2 -> [-1, 1]"), (4, "ball vx", f"/ {P.MAX_SPEED} (max speed)"), (5, "ball vy", f"/ {P.MAX_SPEED}"),
            (6, "own score", f"/ {5}"), (7, "opponent score", f"/ {5}"), (8, "time", "step / step limit")],
           ("#", "feature", "scaling"))
    print("\nACTIONS (discrete, both players move at the same time)")
    _table([(0, "down", f"paddle moves {P.PADDLE_SPEED} per step"), (1, "stay", ""), (2, "up", f"paddle moves {P.PADDLE_SPEED} per step")],
           ("#", "action", ""))
    print("\nREWARD (zero-sum)")
    _table([("you score a point", "+1"), ("opponent scores a point", "-1"), ("anything else", "0")], ("event", "reward"))
    print("\nEPISODE: first to 5 points, or 2000 steps (then the higher score wins; equal = draw)")
    print("\nPHYSICS")
    _table([("court", "x in [-1, 1], y in [-0.5, 0.5]"), ("paddles", f"at x = +-{P.PADDLE_X}, height {2 * P.HALF_PADDLE}, speed {P.PADDLE_SPEED}/step"),
            ("serve", f"from the centre towards the player who lost the point, speed {P.SERVE_SPEED}/step, angle up to 30 deg"),
            ("walls", "top/bottom: the ball bounces"),
            ("paddle hit", f"return angle depends on where the ball hits (centre = flat, edge = up to {math.degrees(P.MAX_ANGLE):.0f} deg)"),
            ("speed-up", f"x{P.SPEEDUP} per hit, max {P.MAX_SPEED}/step (3 x paddle speed: steep fast balls cannot be chased)")],
           ("", ""))


def describe_selfplay_networks(run):
    """Layer table of the trained actor and critic + the PPO settings."""
    final = run["dir"] / "final.pt"
    d = torch.load(final if final.exists() else run["ckpts"][-1], map_location="cpu", weights_only=False)
    from marl.common.nets import Critic

    critic = d["critic"] if "critic" in d else Critic(OBS_DIM, tuple(d["config"]["hidden"])).state_dict()   # mid-training
    rows = []
    for name, sd, out in (("actor (policy)", d["actor"], "3 logits -> softmax over {down, stay, up}"),
                          ("critic V(obs)", critic, "1 number: expected future reward")):
        ws = [sd[k] for k in sorted(sd) if k.endswith("weight")]
        rows.append((name, " -> ".join(map(str, [ws[0].shape[1]] + [w.shape[0] for w in ws])), "ReLU", out,
                     f"{sum(v.numel() for v in sd.values()):,}"))
    _table(rows, ("network", "layers", "hidden act.", "output", "params"))
    c = d["config"]
    print(f"\nPPO: {c['num_games']} games in parallel x {c['rollout_len']} steps per update, {c['epochs']} epochs, "
          f"minibatch {c['minibatch']}, lr {c['lr']}, gamma {c['gamma']}, GAE lambda {c['gae_lambda']}, clip {c['clip']}, "
          f"entropy bonus {c['ent_coef']}")
    print(f"opponents: {c['latest_share']:.0%} of games vs the current self, the rest vs {c['pool_slots']} past snapshots "
          f"drawn from the pool")


def compute_selfplay_summary(run, k=12, games=256, device="cpu"):
    """Tournament between k snapshots + random + bot (scores, Elo) and each snapshot's playing style.
    Saved next to the run (tournament.csv, snapshots.csv); the notebook reads them with load_selfplay_summary."""
    import csv

    import numpy as np

    from marl.analysis import pick

    ckpts = pick(run["ckpts"][1:], k)
    players = {"random": random_policy, "bot": follow_ball_policy}
    steps = {"random": -1, "bot": -1}
    for c in ckpts:
        p = load_snapshot(c, device)
        players[p.name] = p
        steps[p.name] = int(c.stem.split("_")[1])
    names, S = tournament(players, games, device)
    elo = fit_elo(S, games)
    with open(run["dir"] / "tournament.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player", "step", "elo"] + names)
        for i, n in enumerate(names):
            w.writerow([n, steps[n], round(float(elo[i]), 1)] + [round(float(x), 4) for x in S[i]])
    rows = []
    for c in run["ckpts"]:
        p = load_snapshot(c, device)
        st = shot_stats(p, games=games, steps=1000, device=device)
        rows.append({"step": int(c.stem.split("_")[1]), **st})
    with open(run["dir"] / "snapshots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_selfplay_summary(run):
    """(names, steps, elo, S) of the tournament and {column: array} of the snapshot styles."""
    import csv

    import numpy as np

    with open(run["dir"] / "tournament.csv") as f:
        rows = list(csv.reader(f))
    names = [r[0] for r in rows[1:]]
    steps = np.array([int(r[1]) for r in rows[1:]])
    elo = np.array([float(r[2]) for r in rows[1:]])
    S = np.array([[float(x) for x in r[3:]] for r in rows[1:]])
    with open(run["dir"] / "snapshots.csv") as f:
        srows = list(csv.DictReader(f))
    style = {k: np.array([float(r[k]) for r in srows]) for k in srows[0]}
    return {"names": names, "steps": steps, "elo": elo, "S": S}, style


def plot_elo(tour, ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    snap = tour["steps"] >= 0
    ax.plot(tour["steps"][snap] / 1e6, tour["elo"][snap], "o-", color="tab:blue", label="self-play snapshots")
    for name, ls in (("bot", "--"), ("random", ":")):
        i = tour["names"].index(name)
        ax.axhline(tour["elo"][i], ls=ls, color="gray", label=f"{name} ({tour['elo'][i]:.0f})")
    ax.set_xlabel("training steps (millions of game steps)")
    ax.set_ylabel("Elo rating (random player = 0)")
    ax.set_title("Skill over training (round-robin tournament)")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def snapshot_clip(run, step, kind="self"):
    """Path of the stored clip (kind = "self" or "bot") of the snapshot closest to `step`."""
    clips = list((run["dir"] / "videos").glob(f"step_*_{kind}.mp4"))
    return min(clips, key=lambda p: abs(int(p.stem.split("_")[1]) - step))


def generation_ladder(run, k=5, points=3, games=256, before=100, scale=0.75, seed=0):
    """Self-play's core claim, as a video: each generation against the PREVIOUS one. Generations = the untrained
    network + k snapshots spread over training. For each pair: point replays (newer on the left) and the newer
    generation's score over `games` full games. Returns (frames, rows)."""
    import numpy as np
    from PIL import Image

    from marl.analysis import pick
    from marl.pong_render import point_replays

    gens = [load_snapshot(c) for c in [run["ckpts"][0]] + pick(run["ckpts"][1:], k)]
    frames, rows = [], []
    for old, new in zip(gens, gens[1:]):
        fr, scorers = point_replays(new, old, names=(f"new: {new.name}", f"old: {old.name}"), seed=seed,
                                    points=points, before=before)
        score = match(new, old, games, seed=seed)["score"]
        rows.append((new.name, old.name, sum(s.startswith("new") for s in scorers), len(scorers), score))
        for f in fr:
            im = Image.fromarray(f)
            frames.append(np.asarray(im.resize((int(im.width * scale) // 2 * 2, int(im.height * scale) // 2 * 2))))
    return frames, rows
