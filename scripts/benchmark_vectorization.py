#!/usr/bin/env python
"""How fast can we simulate K copies of Coop Navigation? Env-steps per second (one env-step = all agents
of one copy move once), random actions, no learning.

    loop  - K mpe2 environments stepped one after another in Python (marl.common.env.MPEBatch)
    vmas  - one VMAS world with batch dimension K (marl.common.vmas_env.VMASBatch), on --device

    python scripts/benchmark_vectorization.py --device cpu --out checkpoints/vmas/benchmark_cpu.csv
    python scripts/benchmark_vectorization.py --device cuda --backends vmas --out checkpoints/vmas/benchmark_cuda.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from marl.common.env import MPEBatch
from marl.common.vmas_env import VMASBatch

LOOP_MAX = 1024          # beyond this the loop is just slower and slower in proportion


def bench(env, K, min_time=1.0, numpy_actions=False, device="cpu"):
    """Steps all K copies until at least `min_time` seconds have passed; returns env-steps/s."""
    def actions():
        if numpy_actions:
            return [np.random.rand(K, d).astype(np.float32) for d in env.act_dims]
        return [torch.rand(K, d, device=device) for d in env.act_dims]

    env.reset()
    for _ in range(3):                                   # warm-up (allocations, CUDA kernels)
        env.step(actions())
    if device != "cpu":
        torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while True:
        _, _, done = env.step(actions())
        n += 1
        if done:
            env.reset()
        if n % 5 == 0:
            if device != "cpu":
                torch.cuda.synchronize()
            if time.perf_counter() - t0 >= min_time:
                break
    return K * n / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--backends", nargs="*", default=["loop", "vmas"])
    p.add_argument("--N", type=int, nargs="*", default=[3])
    p.add_argument("--K", type=int, nargs="*", default=[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for N in args.N:
        for backend in args.backends:
            for K in args.K:
                if backend == "loop" and K > LOOP_MAX:
                    continue
                if backend == "loop":
                    env = MPEBatch("spread", num_envs=K, N=N)
                    sps = bench(env, K, numpy_actions=True)
                    dev = "cpu"
                else:
                    env = VMASBatch("spread", num_envs=K, N=N, device=args.device)
                    sps = bench(env, K, device=args.device)
                    dev = args.device
                env.close()
                rows.append({"backend": backend, "device": dev, "N": N, "K": K, "env_steps_per_s": round(sps, 1)})
                print(f"{backend:>5} {dev:>5}  N={N}  K={K:>6}  {sps:>12,.0f} env-steps/s", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
