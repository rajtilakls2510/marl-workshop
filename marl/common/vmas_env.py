"""K copies of a VMAS task simulated as ONE batch of tensors (this is real vectorization).

Same interface as marl.common.env.MPEBatch, so the same PPO code trains on either:
    MPEBatch  - K separate mpe2 environments stepped one after another in a Python loop (numpy)
    VMASBatch - one VMAS world whose every tensor has a leading batch dimension K; a step is a
                handful of tensor operations for all K copies at once, on the CPU or the GPU

Observations, rewards and states stay torch tensors on `device` (no copies to numpy).
Actions: our actors output values in [0, 1]; VMAS agents take a 2-D force in [-u, u], so a -> (2a - 1) u.

Only Coop Navigation is wired up (VMAS's port "simple_spread"). The port follows the same rules as mpe2's
simple_spread but is not identical (2-D force actions, slightly different physics, and the distance part
of the reward is counted once per agent), so rewards are not comparable across the two - compare the
behaviour metrics from `spread_metrics` instead.
"""
import torch

SCENARIOS = {"spread": "simple_spread"}


class VMASBatch:
    torch_native = True

    def __init__(self, name, num_envs=64, seed=0, max_cycles=25, device="cpu", N=3, **kwargs):
        from vmas import make_env

        self.name, self.num_envs, self.max_cycles = name, num_envs, max_cycles
        self.device = torch.device(device)
        self.env = make_env(SCENARIOS[name], num_envs=num_envs, device=self.device, continuous_actions=True,
                            seed=seed, n_agents=N, **kwargs)
        self.world = self.env.world
        self.agents = [a.name for a in self.world.agents]
        self.n_agents = len(self.agents)
        self.groups = ["agents"] * self.n_agents
        obs = self.env.reset()
        self.obs_dims = [o.shape[-1] for o in obs]
        self.act_dims = [self.env.get_agent_action_size(a) for a in self.world.agents]
        self.u_range = [a.u_range for a in self.world.agents]
        self.state_dim = sum(self.obs_dims)
        self.t = 0

    def reset(self):
        self.t = 0
        return [o.float() for o in self.env.reset()]

    def step(self, actions):
        """actions: list (per agent) of tensors (K, act_dim) in [0, 1]. Returns obs (list per agent),
        rewards (K, n_agents), done (bool: all copies end together at the time limit)."""
        acts = [(2 * torch.as_tensor(a, device=self.device).clamp(0, 1) - 1) * u
                for a, u in zip(actions, self.u_range)]
        obs, rews, _, _ = self.env.step(acts)
        self.t += 1
        return [o.float() for o in obs], torch.stack(rews, -1).float(), self.t >= self.max_cycles

    @staticmethod
    def state(obs):
        return torch.cat(obs, -1)

    def close(self):
        pass


def spread_metrics(world):
    """Same definitions as marl.metrics for mpe2 (measured at the end of an episode), batched over envs:
    landmark_dist = mean over landmarks of the distance to the closest agent; coverage = fraction of
    landmarks with an agent within 0.1. Returns per-env tensors."""
    A = torch.stack([a.state.pos for a in world.agents], 1)          # (K, N, 2)
    L = torch.stack([l.state.pos for l in world.landmarks], 1)        # (K, M, 2)
    D = torch.linalg.vector_norm(A[:, :, None] - L[:, None], dim=-1)  # (K, N, M)
    closest = D.min(1).values                                         # (K, M)
    return {"landmark_dist": closest.mean(1), "coverage": (closest < 0.1).float().mean(1)}


def collisions(world):
    """Number of overlapping agent pairs in each env, (K,)."""
    ags = world.agents
    out = torch.zeros(world.batch_dim, device=world.device)
    for i in range(len(ags)):
        for j in range(i + 1, len(ags)):
            d = torch.linalg.vector_norm(ags[i].state.pos - ags[j].state.pos, dim=-1)
            out += (d < ags[i].shape.radius + ags[j].shape.radius).float()
    return out
