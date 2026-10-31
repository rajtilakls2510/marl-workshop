#!/usr/bin/env python
"""Train a named MARL run (see marl/presets.py).

    python scripts/train.py --run sl_maddpg
    python scripts/train.py --run spread_mappo_N3 --steps 500000 --set lr=1e-3 --name my_try
    python scripts/train.py --list

Outputs go to runs/<name>/ : log.csv, ckpt/, videos/, evolution.mp4, final.pt
"""
import argparse
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_value(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="preset name")
    p.add_argument("--list", action="store_true")
    p.add_argument("--name", help="output folder name (default: preset name)")
    p.add_argument("--steps", type=int, help="override total env steps")
    p.add_argument("--eval-every", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2, help="torch CPU threads (keep small when running many)")
    p.add_argument("--out", default=None, help="runs root directory (default: <repo>/runs)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config fields")
    args = p.parse_args()

    from marl.presets import LABELS, PRESETS

    if args.list or not args.run:
        for k, v in PRESETS.items():
            print(f"{k:<28} {v}")
        return

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    from marl.common.runner import Run

    preset = dict(PRESETS[args.run])
    if preset["env"] == "smac":
        return train_smac(args, preset)
    env_name, algo = preset.pop("env"), preset.pop("algo")
    steps = args.steps or preset.pop("steps")
    preset.pop("steps", None)
    eval_every = args.eval_every or preset.pop("eval_every")
    preset.pop("eval_every", None)
    env_kwargs = {"N": preset.pop("N")} if "N" in preset else {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k in ("N", "local_ratio"):
            env_kwargs[k] = parse_value(v)
        else:
            preset[k] = parse_value(v)

    if algo == "ppo":
        from marl.algos.ppo import PPOConfig as Cfg, train
    else:
        from marl.algos.ddpg import DDPGConfig as Cfg, train
    cfg = Cfg(**preset)
    label = LABELS[algo][cfg.critic] + (f" N={env_kwargs['N']}" if "N" in env_kwargs else "")
    if algo == "ppo" and not cfg.share:
        label += " (unshared)"
    name = args.name or args.run
    run = Run(name, env_name, env_kwargs, {"algo": algo, "label": label, "steps": steps, "seed": args.seed,
                                            **vars(cfg)}, out_root=args.out)
    print(f"==> {name}: {label} on {env_name} {env_kwargs}, {steps:,} steps, device {args.device}", flush=True)
    train(run, env_name, env_kwargs, cfg, steps, eval_every, seed=args.seed, device=args.device,
          label=label, log_fn=lambda s: print(s, flush=True))


def train_smac(args, preset):
    """SMAClite runs (team reward, discrete actions, worker processes)."""
    from marl.common.smac_runner import SMACRun

    preset.pop("env")
    map_name, algo = preset.pop("map"), preset.pop("algo")
    steps = args.steps or preset.pop("steps")
    preset.pop("steps", None)
    eval_every = args.eval_every or preset.pop("eval_every")
    preset.pop("eval_every", None)
    for kv in args.set:
        k, v = kv.split("=", 1)
        preset[k] = parse_value(v)
    if algo == "smac_mappo":
        from marl.algos.mappo_discrete import SMACPPOConfig as Cfg, train
        label = "MAPPO"
    else:
        from marl.algos.value_decomp import VDConfig as Cfg, train
        label = preset.get("mixer", "qmix").upper()
    cfg = Cfg(**preset)
    name = args.name or args.run
    run = SMACRun(name, map_name, {"algo": algo, "label": f"{label} {map_name}", "steps": steps,
                                   "seed": args.seed, **vars(cfg)}, out_root=args.out)
    print(f"==> {name}: {label} on SMAClite {map_name}, {steps:,} steps, device {args.device}", flush=True)
    train(run, map_name, cfg, steps, eval_every, seed=args.seed, device=args.device,
          label=f"{label} {map_name}", log_fn=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
