#!/usr/bin/env python
"""Explore a SMAClite map from the terminal: print its MDP and save a random-play video.

    python scripts/explore_smaclite.py --list
    python scripts/explore_smaclite.py --map 2s3z
    python scripts/explore_smaclite.py --map MMM --agent 0 --obs --out videos/MMM_random.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl.smac_tools import (decode, make_smac, map_overview, obs_feature_names, print_mdp,  # noqa: E402
                             print_table, rollout_smac, unit_stats)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="table of all maps")
    p.add_argument("--map", default="2s3z")
    p.add_argument("--agent", type=int, default=0, help="agent whose actions/observation are shown")
    p.add_argument("--obs", action="store_true", help="print the agent's full observation layout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="video path (.mp4/.gif); default videos/<map>_random.mp4")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    if args.list:
        print_table(map_overview())
        return
    env = make_smac(args.map, render=not args.no_video)
    print_mdp(env, args.agent)
    print()
    print_table(unit_stats(env))
    if args.obs:
        obs, _ = env.reset(seed=args.seed)
        print(f"\nobservation layout of agent {args.agent} ({env.obs_size} numbers) at the start of an episode:")
        print_table(decode(obs_feature_names(env, args.agent), obs[args.agent], nonzero_only=False))

    frames, records = rollout_smac(env, seed=args.seed, render=not args.no_video)
    ret = sum(r["reward"] for r in records)
    print(f"\nrandom play: {len(records)} steps, won={records[-1]['won']}, return {ret:.2f}/20 "
          f"(damage {sum(r['r_damage'] for r in records):.2f}, kills {sum(r['r_kills'] for r in records):.2f}, "
          f"win {sum(r['r_win'] for r in records):.2f})")
    if not args.no_video:
        from marl.render import save_video

        out = save_video(frames, args.out or f"videos/{args.map}_random.mp4", fps=6)
        print("video:", out)


if __name__ == "__main__":
    main()
