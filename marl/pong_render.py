"""Drawing and playing single games of the hand-made Pong (marl.common.pong) for videos.

    frames, stats = play_game(left_policy, right_policy, names=("snapshot 0.5M", "snapshot 40M"))
    show_video(frames, fps=30)

A policy is any function obs (B, 9) tensor -> actions (B,) tensor (see marl.common.pong for the
built-in random / follow-the-ball ones, marl.selfplay for trained snapshots).
"""
import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from marl.common.pong import BALL_R, HALF_PADDLE, PADDLE_X, PongBatch

W, H, TOP = 800, 500, 70                  # frame size; height of the score bar above the court
MARGIN = 24
BG, COURT, LINE = (14, 18, 38), (20, 26, 52), (70, 82, 130)
COLORS = ((64, 210, 255), (255, 150, 60))  # left (cyan), right (orange)
WHITE = (240, 244, 255)

# seven-segment digits: segments a b c d e f g
_SEG = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc", 7: "abc", 8: "abcdefg", 9: "abcdfg"}


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


FONT, SMALL = _font(18), _font(14)


def _to_px(x, y):
    """Court coordinates (x in [-1, 1], y in [-0.5, 0.5]) -> pixels."""
    cw, ch = W - 2 * MARGIN, H - TOP - 2 * MARGIN
    return MARGIN + (x + 1) / 2 * cw, TOP + MARGIN + (0.5 - y) * ch


def _digit(draw, x, y, d, color, w=22, h=40, t=5):
    segs = {"a": (x, y, x + w, y + t), "b": (x + w - t, y, x + w, y + h // 2), "c": (x + w - t, y + h // 2, x + w, y + h),
            "d": (x, y + h - t, x + w, y + h), "e": (x, y + h // 2, x + t, y + h), "f": (x, y, x + t, y + h // 2),
            "g": (x, y + h // 2 - t // 2, x + w, y + h // 2 + t // 2)}
    for s in _SEG[int(d) % 10]:
        draw.rounded_rectangle(segs[s], radius=2, fill=color)


def _court():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    (x0, y0), (x1, y1) = _to_px(-1, 0.5), _to_px(1, -0.5)
    d.rounded_rectangle((x0 - 6, y0 - 6, x1 + 6, y1 + 6), radius=10, fill=COURT, outline=LINE, width=2)
    cx = (x0 + x1) / 2
    for yy in np.arange(y0, y1, 22):
        d.rectangle((cx - 2, yy, cx + 2, yy + 11), fill=LINE)
    return img


_COURT = None


def render_frame(state, names=("left", "right"), trail=(), info=""):
    """One RGB frame (H, W, 3) from PongBatch.state(i); `trail` = recent ball positions (oldest first)."""
    global _COURT
    if _COURT is None:
        _COURT = _court()
    img = _COURT.copy()
    # paddles with a soft glow (drawn on a blurred layer, then sharp on top)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    rects = []
    for p, (sign, color) in enumerate(((-1, COLORS[0]), (1, COLORS[1]))):
        x, y = _to_px(sign * PADDLE_X, state["paddle"][p])
        hh = HALF_PADDLE / 1.0 * (H - TOP - 2 * MARGIN)
        rects.append(((x - 6, y - hh, x + 6, y + hh), color))
        gd.rounded_rectangle((x - 9, y - hh - 3, x + 9, y + hh + 3), radius=8, fill=color)
    px, py = _to_px(*state["ball"])
    rpx = max(5, BALL_R * (W - 2 * MARGIN) / 2)
    gd.ellipse((px - 2 * rpx, py - 2 * rpx, px + 2 * rpx, py + 2 * rpx), fill=(150, 150, 170))
    halo = glow.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.8))
    img = ImageChops.screen(img, halo)                   # light spreads around paddles and ball
    d = ImageDraw.Draw(img)
    for r, color in rects:
        d.rounded_rectangle(r, radius=6, fill=color)
    # ball with a fading trail
    for k, (bx, by) in enumerate(trail):
        a = (k + 1) / (len(trail) + 1)
        px, py = _to_px(bx, by)
        c = tuple(int(BG[j] + (WHITE[j] - BG[j]) * a * 0.5) for j in range(3))
        r = rpx * (0.5 + 0.5 * a)
        d.ellipse((px - r, py - r, px + r, py + r), fill=c)
    d.ellipse((px - rpx, py - rpx, px + rpx, py + rpx), fill=WHITE)
    # score bar: names, seven-segment scores, rally counter
    d.rectangle((0, 0, W, TOP - 8), fill=BG)
    s0, s1 = (int(v) for v in state["score"])
    _digit(d, W // 2 - 60, 14, s0, COLORS[0])
    _digit(d, W // 2 + 38, 14, s1, COLORS[1])
    d.text((W // 2, 34), ":", fill=LINE, font=FONT, anchor="mm")
    d.text((MARGIN, 24), names[0], fill=COLORS[0], font=FONT, anchor="lm")
    d.text((W - MARGIN, 24), names[1], fill=COLORS[1], font=FONT, anchor="rm")
    d.text((MARGIN, 48), f"rally {state['rally']}", fill=LINE, font=SMALL, anchor="lm")
    if info:
        d.text((W - MARGIN, 48), info, fill=LINE, font=SMALL, anchor="rm")
    return np.asarray(img)


def play_game(left, right, names=("left", "right"), seed=0, points=5, max_steps=2000, every=1, trail_len=6,
              device="cpu"):
    """Play ONE game (left policy vs right policy) and render it. Returns (frames, stats).
    `every` = render every n-th step (shorter videos for long games)."""
    env = PongBatch(1, device=device, points=points, max_steps=max_steps, seed=seed)
    obs = env.reset()
    frames, trail, hits = [], [], 0
    for t in range(max_steps):
        st = env.state(0)
        trail = (trail + [tuple(st["ball"])])[-trail_len:]
        if t % every == 0:
            frames.append(render_frame(st, names, trail[:-1]))
        with torch.no_grad():
            a = torch.stack([left(obs[:, 0]), right(obs[:, 1])], 1)
        obs, rew, done, info = env.step(a)
        hits += int(info["hit"][0])
        if info["point"][0]:
            trail = []
        if done[0]:
            score = info["final_score"][0].tolist()
            final = {"ball": [0.0, 0.0], "paddle": st["paddle"], "score": score, "rally": 0, "t": t + 1}
            frames += [render_frame(final, names, (), info="game over")] * 30
            return frames, {"score": score, "steps": t + 1, "hits": hits}
    return frames, {"score": env.score[0].tolist(), "steps": max_steps, "hits": hits}


def point_replays(left, right, names=("left", "right"), seed=0, points=3, before=150, hold=20, max_steps=8000,
                  device="cpu"):
    """Replays of how points are won: plays one long game and keeps, for each of the first `points` points, the
    last `before` steps before the ball goes out (plus a short pause on the score). Long rallies would make a
    full-game video too long to watch - this shows the decisive shots at normal speed. Returns (frames, scorers).
    (Only the kept steps are drawn: the game itself runs at full simulation speed.)"""
    from collections import deque

    torch.manual_seed(seed)                              # agents sample their moves: same seed -> same replays
    env = PongBatch(1, device=device, points=points + 1, max_steps=max_steps, seed=seed)
    obs = env.reset()
    buf, frames, scorers = deque(maxlen=before), [], []
    for _ in range(max_steps):
        buf.append(env.state(0))
        with torch.no_grad():
            a = torch.stack([left(obs[:, 0]), right(obs[:, 1])], 1)
        obs, rew, done, info = env.step(a)
        if info["point"][0]:
            scorer = names[0] if rew[0, 0] > 0 else names[1]
            scorers.append(scorer)
            states, trail = list(buf), []
            for st in states:
                trail = (trail + [tuple(st["ball"])])[-6:]
                frames.append(render_frame(st, names, trail[:-1], info=f"replay of point {len(scorers)}"))
            score = info["final_score"][0].tolist() if done[0] else env.state(0)["score"]
            frames += [render_frame({**states[-1], "ball": [0.0, 0.0], "score": score, "rally": 0}, names, (),
                                    info=f"point: {scorer}")] * hold
            buf.clear()
            if len(scorers) >= points:
                break
        if done[0]:
            break
    return frames, scorers
