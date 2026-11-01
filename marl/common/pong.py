"""Two-player Pong, K games simulated at once as torch tensors (vectorized, like VMAS in notebook 07).

Court: x in [-1, 1] (left player's paddle at x = -0.95, right player's at +0.95), y in [-0.5, 0.5].

* Actions per player: 0 = down, 1 = stay, 2 = up.
* The ball bounces off the top and bottom walls. When it hits a paddle, the bounce angle depends on WHERE it hits
  (centre = flat return, edge = steep return) and it gets 5% faster, up to a cap - so players can aim, and a
  steep fast ball cannot simply be chased: the paddle must already be where the ball will arrive.
* A point is scored when the ball passes a paddle: +1 for the scorer, -1 for the other (zero-sum).
  A game ends when someone reaches `points` or after `max_steps`; finished games restart on their own.
* Observation (9 numbers per player) is MIRRORED: every player sees the game as if it were the LEFT player,
  so one network can play either side.
"""
import math

import torch

PADDLE_X = 0.95
HALF_PADDLE = 0.1
BALL_R = 0.015
PADDLE_SPEED = 0.02
SERVE_SPEED = 0.025
MAX_SPEED = 0.06
SPEEDUP = 1.05
MAX_ANGLE = math.radians(60)
OBS_DIM = 9


class PongBatch:
    def __init__(self, num_games=1024, device="cpu", points=5, max_steps=2000, seed=0):
        self.K, self.device = num_games, torch.device(device)
        self.points, self.max_steps = points, max_steps
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.ball, self.vel = z(self.K, 2), z(self.K, 2)       # ball position / velocity (per step)
        self.paddle = z(self.K, 2)                              # y of [left, right] paddle
        self.score = z(self.K, 2)
        self.t = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self.hits = torch.zeros(self.K, dtype=torch.long, device=self.device)        # hits in this game
        self.rally = torch.zeros(self.K, dtype=torch.long, device=self.device)       # hits since the serve
        self.reset()

    # ---------------------------------------------------------------- helpers
    def _rand(self, n):
        return torch.rand(n, device=self.device, generator=self.gen)

    def _serve(self, mask, towards=None):
        """Ball to the centre, moving towards `towards` (0 = left, 1 = right; random if None) at a random angle."""
        n = int(mask.sum())
        if n == 0:
            return
        side = (self._rand(n) < 0.5).long() if towards is None else towards[mask]
        angle = (self._rand(n) * 2 - 1) * math.radians(30)
        direction = side.float() * 2 - 1                          # -1 = towards left, +1 = towards right
        self.ball[mask] = torch.stack([torch.zeros(n, device=self.device), (self._rand(n) - 0.5) * 0.4], -1)
        self.vel[mask] = SERVE_SPEED * torch.stack([direction * torch.cos(angle), torch.sin(angle)], -1)
        self.rally[mask] = 0

    def _reset_games(self, mask):
        self.paddle[mask] = 0.0
        self.score[mask] = 0.0
        self.t[mask] = 0
        self.hits[mask] = 0
        self._serve(mask)

    def reset(self):
        self._reset_games(torch.ones(self.K, dtype=torch.bool, device=self.device))
        return self.obs()

    # ---------------------------------------------------------------- observation
    def obs(self):
        """(K, 2, 9): for each player, from its own (mirrored) point of view:
        own paddle y, opponent paddle y, ball x, ball y, ball vx, ball vy, own score, opponent score, time left."""
        x, y = self.ball[:, 0], self.ball[:, 1]
        vx, vy = self.vel[:, 0] / MAX_SPEED, self.vel[:, 1] / MAX_SPEED
        time = self.t.float() / self.max_steps
        out = []
        for p, sign in ((0, 1.0), (1, -1.0)):                    # the right player sees the court flipped in x
            out.append(torch.stack([self.paddle[:, p] * 2, self.paddle[:, 1 - p] * 2, sign * x, y * 2, sign * vx,
                                    vy, self.score[:, p] / self.points, self.score[:, 1 - p] / self.points,
                                    time], -1))
        return torch.stack(out, 1)

    # ---------------------------------------------------------------- dynamics
    def step(self, actions):
        """actions (K, 2) ints in {0, 1, 2}. Returns obs (K, 2, 9), rewards (K, 2), done (K,), info."""
        a = torch.as_tensor(actions, device=self.device).long()
        self.paddle = (self.paddle + (a - 1).float() * PADDLE_SPEED).clamp(-0.5 + HALF_PADDLE, 0.5 - HALF_PADDLE)
        prev = self.ball.clone()
        self.ball = self.ball + self.vel
        # walls (top / bottom): reflect
        for wall in (0.5 - BALL_R, -0.5 + BALL_R):
            over = (self.ball[:, 1] - wall) * (1 if wall > 0 else -1) > 0
            self.ball[over, 1] = 2 * wall - self.ball[over, 1]
            self.vel[over, 1] = -self.vel[over, 1]
        hit_any = torch.zeros(self.K, dtype=torch.bool, device=self.device)
        # paddles: did the ball cross the paddle's x during this step, within the paddle's extent?
        for p, px, sign in ((0, -PADDLE_X + BALL_R, -1.0), (1, PADDLE_X - BALL_R, 1.0)):
            crossing = (self.vel[:, 0] * sign > 0) & ((prev[:, 0] - px) * sign <= 0) & ((self.ball[:, 0] - px) * sign > 0)
            frac = ((px - prev[:, 0]) / self.vel[:, 0].where(self.vel[:, 0] != 0, torch.ones_like(prev[:, 0])))
            y_at = (prev[:, 1] + frac.clamp(0, 1) * (self.ball[:, 1] - prev[:, 1]))
            offset = (y_at - self.paddle[:, p]) / (HALF_PADDLE + BALL_R)
            hit = crossing & (offset.abs() <= 1)
            if hit.any():
                speed = (self.vel[hit].norm(dim=-1) * SPEEDUP).clamp(max=MAX_SPEED)
                angle = offset[hit].clamp(-1, 1) * MAX_ANGLE
                self.vel[hit] = torch.stack([-sign * speed * torch.cos(angle), speed * torch.sin(angle)], -1)
                self.ball[hit, 0] = px
                self.ball[hit, 1] = y_at[hit]
                hit_any |= hit
        self.hits += hit_any.long()
        self.rally += hit_any.long()
        # points: the ball left the court behind a paddle
        right_scores = self.ball[:, 0] < -1.0
        left_scores = self.ball[:, 0] > 1.0
        rew = torch.zeros(self.K, 2, device=self.device)
        rew[left_scores, 0], rew[left_scores, 1] = 1.0, -1.0
        rew[right_scores, 0], rew[right_scores, 1] = -1.0, 1.0
        self.score[:, 0] += left_scores.float()
        self.score[:, 1] += right_scores.float()
        point = left_scores | right_scores
        rally_at_point = self.rally.clone()
        # next serve goes towards the player who lost the point
        self._serve(point, towards=left_scores.long())         # 1 = right player lost the point
        self.t += 1
        done = (self.score.max(1).values >= self.points) | (self.t >= self.max_steps)
        info = {"point": point, "hit": hit_any, "rally_at_point": rally_at_point,
                "final_score": self.score[done].clone(), "game_hits": self.hits[done].clone(),
                "game_steps": self.t[done].clone(), "truncated": done & (self.score.max(1).values < self.points)}
        self._reset_games(done)
        return self.obs(), rew, done, info

    def state(self, i=0):
        """Positions of game i (for rendering)."""
        return {"ball": self.ball[i].tolist(), "paddle": self.paddle[i].tolist(), "score": self.score[i].tolist(),
                "rally": int(self.rally[i]), "t": int(self.t[i])}


# ---------------------------------------------------------------- reference policies (evaluation only)
def random_policy(obs):
    """obs (B, 9) -> random actions (B,)."""
    return torch.randint(0, 3, obs.shape[:1], device=obs.device)


def follow_ball_policy(obs, deadband=0.02):
    """Scripted yardstick, NEVER used for training: move the paddle towards the ball's current height."""
    diff = obs[:, 3] - obs[:, 0]                       # ball y - own paddle y (both scaled x2)
    return torch.where(diff > deadband, 2, torch.where(diff < -deadband, 0, 1)).to(torch.long)
