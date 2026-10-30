#!/usr/bin/env python
"""Render Multi-Agent Particle Environment (MPE) episodes to a video file or a live window.

Examples (run from the repo root):
    python scripts/render_mpe.py --env spread --out videos/spread.mp4
    python scripts/render_mpe.py --env tag --episodes 3 --out videos/tag.gif
    python scripts/render_mpe.py --env speaker_listener --live        # window; not inside Docker

Envs: spread (Coop Navigation), tag (Predator-Prey), speaker_listener (Speaker-Listener).
Agents act randomly for now; trained policies plug into the same tool later.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl.render import MPE_ENVS, make_mpe, rollout, save_video  # noqa: E402


def in_docker():
    return Path("/.dockerenv").exists()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=list(MPE_ENVS), default="spread")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-cycles", type=int, default=25, help="steps per episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--N", type=int, default=None, help="number of agents (spread only)")
    p.add_argument("--discrete", action="store_true", help="discrete actions (default: continuous)")
    p.add_argument("--out", default=None, help="output .mp4 or .gif (default: videos/<env>.mp4)")
    p.add_argument("--live", action="store_true", help="show a live window instead of writing a file")
    args = p.parse_args()

    kwargs = {"N": args.N} if args.N and args.env == "spread" else {}
    common = dict(continuous=not args.discrete, max_cycles=args.max_cycles, **kwargs)

    if args.live:
        if in_docker():
            sys.exit("--live needs a desktop window, which Docker does not have.\n"
                     "Use the live cell in notebooks/01_mpe_tour.ipynb, or write a video with --out.")
        import os
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            del os.environ["SDL_VIDEODRIVER"]
        env = make_mpe(args.env, render_mode="human", **common)
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            while env.agents:
                env.step({a: env.action_space(a).sample() for a in env.agents})
                time.sleep(1.0 / args.fps)
        env.close()
        return

    env = make_mpe(args.env, **common)
    frames, returns = rollout(env, seed=args.seed, episodes=args.episodes)
    out = save_video(frames, args.out or f"videos/{args.env}.mp4", fps=args.fps)
    env.close()
    print(f"wrote {out}  ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
    for i, r in enumerate(returns):
        print(f"  episode {i}: " + ", ".join(f"{a} {v:+.2f}" for a, v in r.items()))


if __name__ == "__main__":
    main()
