"""Task-specific behaviour metrics, measured from the simulator during evaluation episodes.

Returns are hard to interpret on their own; these say *what the agents actually do*.
"""
import numpy as np


def _pos(entity):
    return entity.state.p_pos


class EpisodeMetrics:
    """Accumulates metrics over one batch of lock-step episodes (one per vector env)."""

    def __init__(self, env_name, n_envs):
        self.name, self.n = env_name, n_envs
        self.collisions = np.zeros(n_envs)   # spread: agent-agent overlaps per episode
        self.contacts = np.zeros(n_envs)     # tag: predator-prey touches per episode
        self.min_prey_dist = []              # tag: closest predator to prey, per step
        self.final = {}

    def step(self, worlds, rewards):
        if self.name == "spread":
            for k, w in enumerate(worlds):
                ags = w.agents
                for i in range(len(ags)):
                    for j in range(i + 1, len(ags)):
                        if np.linalg.norm(_pos(ags[i]) - _pos(ags[j])) < ags[i].size + ags[j].size:
                            self.collisions[k] += 1
        elif self.name == "tag":
            # every predator gets +10 per predator-prey contact, so predator_0's reward counts contacts
            self.contacts += rewards[:, 0] / 10.0
            d = []
            for w in worlds:
                prey = [a for a in w.agents if not a.adversary][0]
                d.append(min(np.linalg.norm(_pos(a) - _pos(prey)) for a in w.agents if a.adversary))
            self.min_prey_dist.append(d)

    def finish(self, worlds):
        out = {}
        if self.name == "spread":
            covered, dist, assign = [], [], []
            for w in worlds:
                D = np.array([[np.linalg.norm(_pos(a) - _pos(l)) for l in w.landmarks] for a in w.agents])
                covered.append(np.mean(D.min(axis=0) < 0.1))      # landmark has an agent on it
                dist.append(D.min(axis=0).mean())
                assign.append(D.argmin(axis=0))                   # which agent sits on each landmark
            out = {"coverage": float(np.mean(covered)), "landmark_dist": float(np.mean(dist)),
                   "collisions": float(self.collisions.mean()),
                   "distinct_agents_on_landmarks": float(np.mean([len(set(a)) / len(a) for a in assign]))}
        elif self.name == "tag":
            out = {"contacts": float(self.contacts.mean()),
                   "prey_min_dist": float(np.mean(self.min_prey_dist))}
        elif self.name == "speaker_listener":
            d = [np.linalg.norm(_pos(w.agents[0].goal_a) - _pos(w.agents[0].goal_b)) for w in worlds]
            touch = worlds[0].agents[0].goal_a.size + worlds[0].agents[0].goal_b.size   # listener touches goal
            out = {"goal_dist": float(np.mean(d)), "success": float(np.mean(np.array(d) < touch))}
        return out
