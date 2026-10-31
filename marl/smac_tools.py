"""Tools to explore SMAClite (StarCraft II micromanagement without the game).

    env = make_smac("2s3z")
    print_mdp(env)                                   # sizes, actions, unit stats
    obs = env.reset(seed=0)[0]
    decode(obs_feature_names(env, agent=0), obs[0])  # one agent's observation, named
    frames, records = rollout_smac(env, seed=0)      # random play + per-step reward breakdown

Feature names are built from the environment's own layout (smaclite/env/smaclite.py), so they
stay correct for every map.
"""
import os
import random
from contextlib import contextmanager

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

MOVE = ["move north", "move south", "move east", "move west"]


def list_maps():
    import gymnasium as gym
    import smaclite  # noqa: F401  (registers the maps)

    return sorted(k.split("/")[1][:-3] for k in gym.registry if k.startswith("smaclite/") and "custom" not in k)


def make_smac(name, render=True):
    import gymnasium as gym
    import smaclite  # noqa: F401
    import sklearn

    import contextlib
    import io

    sklearn.set_config(assume_finite=True)          # skip input checks in the neighbour search (faster)
    with contextlib.redirect_stdout(io.StringIO()):   # silence "Using the numpy RVO2 port"
        env = gym.make(f"smaclite/{name}-v0", render_mode="rgb_array" if render else None).unwrapped
    env.map_name = name
    return env


# ---------------------------------------------------------------------------- unit types
def _type_names(env):
    """Unit-type names in the order used by the one-hot features."""
    ids = env.map_info.unit_type_ids
    return [t.stats.name for t, _ in sorted(ids.items(), key=lambda kv: kv[1])] if ids else []


def team_composition(env, faction="ALLY"):
    """List of unit-type names, one per unit, in agent/enemy id order."""
    out = []
    for g in env.map_info.groups:
        if g.faction.name == faction:
            for t, count in g.units:
                out += [t.stats.name] * count
    return out


def unit_stats(env):
    """Stats of every unit type on the map (from the unit JSON files)."""
    seen, rows = set(), []
    for g in env.map_info.groups:
        for t, _ in g.units:
            s = t.stats
            if s.name in seen:
                continue
            seen.add(s.name)
            rows.append({"unit": s.name, "hp": s.hp, "shield": s.shield, "armor": s.armor,
                         "damage": s.damage, "attacks": s.attacks, "cooldown_s": s.cooldown,
                         "attack_range": s.attack_range, "speed": s.speed,
                         "role": s.combat_type.name.lower(),
                         "bonus": ", ".join(f"+{v} vs {k.name.lower()}" for k, v in (s.bonuses or {}).items())})
    return rows


# ---------------------------------------------------------------------------- names
def action_names(env, agent=0):
    allies, enemies = team_composition(env, "ALLY"), team_composition(env, "ENEMY")
    healer = allies[agent] == "medivac"
    names = ["no-op (dead agents only)", "stop"] + MOVE
    for j in range(env.n_actions - 6):
        if healer:
            names.append(f"heal ally {j} ({allies[j]})" if j < len(allies) else f"heal ally {j}")
        else:
            names.append(f"attack enemy {j} ({enemies[j]})" if j < len(enemies) else f"target {j}")
    return names


def obs_feature_names(env, agent=0):
    mi, types = env.map_info, _type_names(env)
    allies, enemies = team_composition(env, "ALLY"), team_composition(env, "ENEMY")
    names = [f"can {m}" for m in MOVE]
    for j, e in enumerate(enemies):
        p = f"enemy {j} ({e})"
        names += [f"{p}: attackable", f"{p}: distance", f"{p}: dx", f"{p}: dy", f"{p}: health"]
        names += [f"{p}: shield"] if mi.enemy_has_shields else []
        names += [f"{p}: is {t}" for t in types]
    others = [k for k in range(len(allies)) if k != agent]
    for k in others:
        p = f"ally {k} ({allies[k]})"
        names += [f"{p}: visible", f"{p}: distance", f"{p}: dx", f"{p}: dy", f"{p}: health"]
        names += [f"{p}: shield"] if mi.ally_has_shields else []
        names += [f"{p}: is {t}" for t in types]
    names += ["own health"] + (["own shield"] if mi.ally_has_shields else []) + [f"own: is {t}" for t in types]
    assert len(names) == env.obs_size, (len(names), env.obs_size)
    return names


def state_feature_names(env):
    mi, types = env.map_info, _type_names(env)
    allies, enemies = team_composition(env, "ALLY"), team_composition(env, "ENEMY")
    names = []
    for i, a in enumerate(allies):
        p = f"ally {i} ({a})"
        names += [f"{p}: health", f"{p}: {'energy' if a == 'medivac' else 'weapon cooldown'}", f"{p}: x", f"{p}: y"]
        names += [f"{p}: shield"] if mi.ally_has_shields else []
        names += [f"{p}: is {t}" for t in types]
    for j, e in enumerate(enemies):
        p = f"enemy {j} ({e})"
        names += [f"{p}: health", f"{p}: x", f"{p}: y"] + ([f"{p}: shield"] if mi.enemy_has_shields else [])
        names += [f"{p}: is {t}" for t in types]
    for i in range(len(allies)):
        names += [f"ally {i} last action = {a}" for a in action_names(env, i)]
    assert len(names) == env.state_size, (len(names), env.state_size)
    return names


def decode(names, vector, nonzero_only=True):
    """(name, value) pairs; by default only the features that are not zero."""
    return [(n, round(float(v), 3)) for n, v in zip(names, vector) if (v != 0 or not nonzero_only)]


def print_mdp(env, agent=0):
    allies, enemies = team_composition(env, "ALLY"), team_composition(env, "ENEMY")
    print(f"map {env.map_name}: {len(allies)} agents {allies} vs {len(enemies)} scripted enemies {enemies}")
    print(f"actions per agent: {env.n_actions} | observation: {env.obs_size} numbers | global state: {env.state_size} numbers")
    print("actions:", ", ".join(f"{i}={n}" for i, n in enumerate(action_names(env, agent))))


# ---------------------------------------------------------------------------- reward breakdown
class _Recorder:
    """Per step: damage dealt and kills by each ally, healing by each Medivac (raw game units)."""

    def __init__(self, n_agents):
        self.n = n_agents
        self.reset()

    def reset(self):
        self.damage = np.zeros(self.n)
        self.kills = np.zeros(self.n)
        self.healing = np.zeros(self.n)


@contextmanager
def recording(env):
    """Temporarily wrap Unit.deal_damage / Unit.heal to attribute reward to the allied unit
    that caused it. The environment itself is not modified."""
    from smaclite.env.units.unit import Unit

    rec = _Recorder(env.n_agents)
    deal, heal = Unit.deal_damage, Unit.heal

    def deal_damage(self, target):
        alive = target.hp > 0
        r = deal(self, target)
        if self.faction.name == "ALLY" and target.faction.name == "ENEMY":
            killed = alive and target.hp == 0
            rec.damage[self.id_in_faction] += r - 10 * killed
            rec.kills[self.id_in_faction] += killed
        return r

    def do_heal(self, target):
        before = target.hp
        heal(self, target)
        if self.faction.name == "ALLY":
            rec.healing[self.id_in_faction] += target.hp - before

    Unit.deal_damage, Unit.heal = deal_damage, do_heal
    try:
        yield rec
    finally:
        Unit.deal_damage, Unit.heal = deal, heal


def random_policy(seed=0):
    rng = random.Random(seed)
    return lambda env, obs: [rng.choice([a for a, ok in enumerate(av) if ok]) for av in env.get_avail_actions()]


def _team_health(units, n):
    return sum((u.hp + u.shield) / (u.max_hp + u.max_shield) for u in units.values()) / n


def _unit_box(frame):
    """Bounding box (y0, y1, x0, x1) of the green/red unit pixels in a rendered frame."""
    f = frame.astype(int)
    ys, xs = np.nonzero((f[..., 1] - f[..., 0] > 60) | (f[..., 0] - f[..., 1] > 60))
    return (ys.min(), ys.max(), xs.min(), xs.max()) if len(xs) else None


def _crop_all(frames, width=960, height=540, margin=50):
    """Zoom on the action with ONE crop for the whole episode (union of all unit positions plus a
    margin), widened to a fixed 16:9 shape so every video has the same size and the camera does
    not jump."""
    from PIL import Image

    boxes = [b for b in map(_unit_box, frames) if b is not None]
    if not boxes:
        return frames
    H, W = frames[0].shape[:2]
    y0, y1 = min(b[0] for b in boxes) - margin, max(b[1] for b in boxes) + margin
    x0, x1 = min(b[2] for b in boxes) - margin, max(b[3] for b in boxes) + margin
    bw, bh = x1 - x0, y1 - y0
    bw, bh = max(bw, bh * width / height), max(bh, bw * height / width)       # grow to 16:9
    bw, bh = min(bw, W), min(bh, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(np.clip(cx - bw / 2, 0, W - bw)), int(np.clip(cy - bh / 2, 0, H - bh))
    x1, y1 = int(x0 + bw), int(y0 + bh)
    return [np.asarray(Image.fromarray(f[y0:y1, x0:x1]).resize((width, height))) for f in frames]


def rollout_smac(env, policy=None, seed=0, max_steps=150, render=True, zoom=True, title=None):
    """Play one episode. Returns (frames, records); records[t] holds the actions, the availability
    mask, the reward split into damage / kill / win parts (normalised like the env reward), per-agent
    damage, kills and healing, units alive and team health."""
    from marl.render import overlay

    policy = policy or random_policy(seed)
    obs, _ = env.reset(seed=seed)
    norm = env.max_reward / 20.0                 # the env divides raw reward by this
    frames, texts, records, total = [], [], [], 0.0
    title = title or f"SMAClite {env.map_name}"
    with recording(env) as rec:
        for t in range(max_steps + 1):
            if render:
                frames.append(env.render())
                texts.append([f"{title}  step {t}",
                              f"allies {len(env.agents)}  enemies {len(env.enemies)}  return {total:.1f}/20"])
            if t == max_steps:
                break
            avail = np.array(env.get_avail_actions())
            actions = [int(a) for a in policy(env, obs)]
            rec.reset()
            obs, r, done, _, info = env.step(actions)
            total += r
            win = 200.0 if info["battle_won"] else 0.0
            records.append({
                "t": t, "actions": actions, "avail": avail, "reward": r,
                "r_damage": rec.damage.sum() / norm, "r_kills": 10 * rec.kills.sum() / norm, "r_win": win / norm,
                "damage": rec.damage.copy(), "kills": rec.kills.copy(), "healing": rec.healing.copy(),
                "allies_alive": len(env.agents), "enemies_alive": len(env.enemies),
                "ally_health": _team_health(env.agents, env.n_agents),
                "enemy_health": _team_health(env.enemies, env.n_enemies), "won": info["battle_won"]})
            if done:
                if render:
                    frames.append(env.render())
                    texts.append([f"{title}  step {t + 1}", "WON" if info["battle_won"] else "LOST"])
                break
    if render:
        frames = _crop_all(frames) if zoom else frames
        frames = [overlay(f, txt, 0.8) for f, txt in zip(frames, texts)]
    return frames, records


def plot_episode(records, env=None, axes=None):
    """Reward breakdown per step + team health + per-agent contribution for one episode."""
    import matplotlib.pyplot as plt

    axes = axes if axes is not None else plt.subplots(1, 3, figsize=(14, 3.6))[1]
    t = [r["t"] for r in records]
    for key, lab in (("r_damage", "damage dealt"), ("r_kills", "kill bonus"), ("r_win", "win bonus")):
        axes[0].bar(t, [r[key] for r in records], label=lab, alpha=0.8,
                    bottom=None if key == "r_damage" else np.array([sum(r[k] for k in ("r_damage", "r_kills")[:("r_damage", "r_kills", "r_win").index(key)]) for r in records]))
    axes[0].set_title("team reward per step (shared by every agent)")
    axes[0].set_xlabel("step")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, [r["ally_health"] for r in records], label="allies", color="tab:green")
    axes[1].plot(t, [r["enemy_health"] for r in records], label="enemies", color="tab:red")
    axes[1].set_title("team health (hp + shield, fraction)")
    axes[1].set_xlabel("step")
    axes[1].legend(fontsize=8)
    dmg = np.sum([r["damage"] for r in records], axis=0)
    heal = np.sum([r["healing"] for r in records], axis=0)
    labels = [f"{i}\n{n[:4]}" for i, n in enumerate(team_composition(env))] if env else [str(i) for i in range(len(dmg))]
    axes[2].bar(range(len(dmg)), dmg, label="damage dealt", color="tab:blue")
    if heal.sum() > 0:
        axes[2].bar(range(len(heal)), heal, bottom=dmg, label="healing done", color="tab:purple")
    axes[2].set_xticks(range(len(dmg)), labels, fontsize=8)
    axes[2].set_title("what each agent did (raw game units)")
    axes[2].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    return axes


# ---------------------------------------------------------------------------- printing
def print_table(rows, columns=None):
    """Print a list of dicts (or (k, v) tuples) as an aligned text table."""
    if rows and isinstance(rows[0], tuple):
        rows = [{"feature": k, "value": v} for k, v in rows]
    columns = columns or list(rows[0])
    w = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  ".join(str(c).ljust(w[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in columns))


def map_overview():
    """One row per SMAClite map: teams and MDP sizes."""
    from collections import Counter

    rows = []
    for name in list_maps():
        env = make_smac(name, render=False)
        fmt = lambda units: " + ".join(f"{n} {t}" for t, n in Counter(units).items())
        rows.append({"map": name, "agents": env.n_agents, "allies": fmt(team_composition(env, "ALLY")),
                     "enemies": fmt(team_composition(env, "ENEMY")), "actions": env.n_actions,
                     "obs": env.obs_size, "state": env.state_size})
        env.close()
    return rows


def available(env, agent):
    """Names of the actions agent can take right now."""
    names, av = action_names(env, agent), env.get_avail_actions()[agent]
    return [n for n, ok in zip(names, av) if ok]


def has_attack(env, agent):
    """True if the agent can currently attack (or heal) somebody."""
    return bool(env.get_avail_actions()[agent][6:].any())
