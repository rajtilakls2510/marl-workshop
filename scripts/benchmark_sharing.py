#!/usr/bin/env python
"""What does parameter sharing buy computationally? MAPPO on Coop Navigation, shared vs
separate networks, for growing team sizes.

Measured per configuration (same rollout data for both variants, so only the networks differ):
  params       - number of trainable parameters (actors + critics)
  act_ms       - time for all N agents to pick actions in 64 parallel episodes (one env step)
  update_s     - time for one PPO update (10 epochs over 64 episodes x 25 steps)
  gpu_mem_mb   - peak GPU memory during the update (only with --device cuda)

    python scripts/benchmark_sharing.py                    # N = 3, 6, 10, 20 on CPU
    python scripts/benchmark_sharing.py --quick            # N = 3, 6 (about a minute)
    python scripts/benchmark_sharing.py --device cuda      # GPU timings + memory
Writes checkpoints/mpe/benchmark_sharing[_<device>].csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from marl.algos.ppo import PPO, PPOConfig  # noqa: E402
from marl.common.env import MPEBatch  # noqa: E402


def timed(fn, repeats):
    fn()                                         # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="3,6,10,20")
    p.add_argument("--quick", action="store_true", help="N = 3, 6 and fewer repeats")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    ns = [3, 6] if args.quick else [int(x) for x in args.ns.split(",")]
    reps_act, reps_upd = (20, 1) if args.quick else (100, 3)
    out = Path(args.out or ROOT / "checkpoints" / "mpe" /
               f"benchmark_sharing{'' if args.device == 'cpu' else '_' + args.device}.csv")

    rows = []
    for n in ns:
        env = MPEBatch("spread", num_envs=64, seed=0, N=n)
        rb = None
        for share in (True, False):
            torch.manual_seed(0)
            algo = PPO(env, PPOConfig(critic="central", share=share), device=args.device)
            if rb is None:                                       # one rollout per N, reused
                rb, _ = algo.collect(env)
            obs = env.reset()
            params = sum(q.numel() for q in algo.parameters())
            act_s = timed(lambda: algo.act(obs), reps_act)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            upd_s = timed(lambda: algo.update(rb), reps_upd)
            mem = torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else float("nan")
            row = {"N": n, "shared": share, "networks": len(algo.agents_of) * 2, "params": params,
                   "act_ms": round(act_s * 1e3, 3), "update_s": round(upd_s, 3), "gpu_mem_mb": round(mem, 1)}
            rows.append(row)
            print(row, flush=True)
        env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
