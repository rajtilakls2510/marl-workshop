"""Render MPE episodes to video files, inline notebook videos, or live in a notebook.

    from marl.render import make_mpe, rollout, save_video, show_video, play_live

    env = make_mpe("spread")                    # also "tag", "speaker_listener"
    frames, returns = rollout(env, seed=0)      # random policy
    save_video(frames, "spread.mp4")            # or .gif
    show_video(frames)                          # inline in Jupyter
    play_live(make_mpe("tag"))                  # live, with a Stop button

A policy is any callable ``policy(agent_name, obs) -> action``; ``None`` means random actions.
"""
import base64
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

MPE_ENVS = {
    "spread": ("simple_spread_v3", "Coop Navigation"),
    "tag": ("simple_tag_v3", "Predator-Prey"),
    "speaker_listener": ("simple_speaker_listener_v4", "Speaker-Listener"),
}

# Agent-name prefix -> team label shown in the overlay.
_TEAMS = {"adversary": "predators", "agent": "agents", "speaker": "team", "listener": "team"}


def make_mpe(name, continuous=True, render_mode="rgb_array", max_cycles=25, **kwargs):
    """Create an MPE parallel env by short name ("spread", "tag", "speaker_listener")."""
    import importlib

    if name not in MPE_ENVS:
        raise ValueError(f"unknown MPE env {name!r}; choose from {list(MPE_ENVS)}")
    module = importlib.import_module(f"mpe2.{MPE_ENVS[name][0]}")
    env = module.parallel_env(
        continuous_actions=continuous, render_mode=render_mode, max_cycles=max_cycles, **kwargs
    )
    env.short_name = name
    return env


def team_of(env, agent):
    """Team label of an agent (in Predator-Prey the single good agent is the prey)."""
    prefix = agent.rsplit("_", 1)[0]
    if getattr(env, "short_name", "") == "tag" and prefix == "agent":
        return "prey"
    return _TEAMS.get(prefix, prefix)


def _team_returns(env, returns):
    teams = {}
    for agent, ret in returns.items():
        teams.setdefault(team_of(env, agent), []).append(ret)
    return {team: float(np.mean(v)) for team, v in teams.items()}


def overlay(frame, lines, scale=1.0):
    """Draw text lines in the top-left corner of an RGB frame (returns a new array)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    size = max(12, int(20 * scale))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(20, 20, 20), font=font)
        y += size + 4
    return np.asarray(img)


def _resize(frame, scale):
    if scale == 1.0:
        return frame
    from PIL import Image

    h, w = frame.shape[:2]
    return np.asarray(Image.fromarray(frame).resize((int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)))


def _random_policy(env, seed):
    for i, agent in enumerate(env.possible_agents):
        env.action_space(agent).seed(seed + i)
    return lambda agent, obs: env.action_space(agent).sample()


def episode(env, policy=None, seed=0, annotate=True, scale=1.0, title=None):
    """Generator over one episode: yields (frame, returns_so_far) after reset and after each step."""
    policy = policy or _random_policy(env, seed)
    obs, _ = env.reset(seed=seed)
    returns = {a: 0.0 for a in env.agents}
    title = title or MPE_ENVS.get(getattr(env, "short_name", ""), ("", ""))[1]
    t = 0

    def frame():
        f = _resize(env.render(), scale)
        if annotate:
            teams = "   ".join(f"{k}: {v:+.1f}" for k, v in _team_returns(env, returns).items())
            f = overlay(f, [f"{title}  step {t}", teams], scale)
        return f

    yield frame(), dict(returns)
    while env.agents:
        actions = {a: policy(a, obs[a]) for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        for a, r in rewards.items():
            returns[a] += float(r)
        t += 1
        yield frame(), dict(returns)


def rollout(env, policy=None, seed=0, episodes=1, annotate=True, scale=1.0, title=None):
    """Run episode(s) and return (frames, returns_per_episode)."""
    frames, all_returns = [], []
    for ep in range(episodes):
        returns = None
        for f, returns in episode(env, policy, seed + ep, annotate, scale, title):
            frames.append(f)
        all_returns.append(returns)
    return frames, all_returns


def save_video(frames, path, fps=10):
    """Write frames to .mp4 (needs imageio-ffmpeg) or .gif, chosen by the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    elif path.suffix.lower() == ".mp4":
        import imageio.v2 as imageio

        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                pixelformat="yuv420p") as w:
            for f in frames:
                w.append_data(f)
    else:
        raise ValueError(f"unsupported video format {path.suffix!r} (use .mp4 or .gif)")
    return path


def show_video(frames_or_path, fps=10, width=420):
    """Display a video inline in Jupyter, from a list of frames or an .mp4/.gif file."""
    from IPython.display import HTML

    if isinstance(frames_or_path, (str, Path)):
        path = Path(frames_or_path)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            path = save_video(frames_or_path, tmp.name, fps)
        except Exception:  # no MP4 encoder available: fall back to GIF
            path = save_video(frames_or_path, tmp.name[:-4] + ".gif", fps)
    data = base64.b64encode(path.read_bytes()).decode()
    if path.suffix.lower() == ".gif":
        return HTML(f'<img src="data:image/gif;base64,{data}" width="{width}">')
    return HTML(f'<video width="{width}" controls autoplay loop muted>'
                f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>')


def play_live(env, policy=None, fps=10, episodes=3, seed=0, scale=0.6):
    """Play episodes live inside a notebook (no display needed) with a Stop button.

    Runs in a background thread so the Stop button stays responsive; returns immediately.
    The thread is available as ``play_live.last_thread`` (useful for tests).
    """
    import io

    import ipywidgets as widgets
    from IPython.display import display
    from PIL import Image

    image = widgets.Image(format="jpeg")
    stop = widgets.Button(description="Stop", button_style="danger")
    status = widgets.Label()
    stopped = threading.Event()
    stop.on_click(lambda _: stopped.set())
    display(widgets.VBox([image, widgets.HBox([stop, status])]))

    def to_jpeg(frame):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def run():
        for ep in range(episodes):
            for frame, returns in episode(env, policy, seed + ep, scale=scale):
                if stopped.is_set():
                    status.value = "stopped"
                    return
                image.value = to_jpeg(frame)
                time.sleep(1.0 / fps)
            status.value = f"episode {ep + 1}/{episodes} done"
        status.value = "finished"

    def guarded():
        try:
            run()
        except Exception as e:  # surface errors in the widget instead of losing them in the thread
            status.value = f"error: {type(e).__name__}: {e}"
            raise

    play_live.last_thread = threading.Thread(target=guarded, daemon=True)
    play_live.last_thread.start()


def describe(env, seed=0):
    """Print each agent's team, observation size and action space."""
    env.reset(seed=seed)
    print(f"{'agent':<14}{'team':<12}{'obs dim':<10}action space")
    for a in env.agents:
        print(f"{a:<14}{team_of(env, a):<12}{env.observation_space(a).shape[0]:<10}{env.action_space(a)}")
