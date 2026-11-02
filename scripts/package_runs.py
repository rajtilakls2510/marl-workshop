#!/usr/bin/env python
"""Copy finished training runs from runs/<name>/ into checkpoints/<suite>/<name>/ (what is shipped in
the repo and baked into the Docker image); suite = smac / vmas / selfplay / mpe.

Kept: log.csv, config.json, ckpt/ (all snapshots - the analyses compare policies across training
time), final.pt, evolution.mp4. Not kept: per-snapshot clips in videos/ (notebooks regenerate
them from the snapshots) - except for self-play runs, whose clips are shown in notebook 09.

    python scripts/package_runs.py                 # every finished run
    python scripts/package_runs.py sl_maddpg tag_maddpg
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs"


def destination(src):
    env = json.loads((src / "config.json").read_text()).get("env", "")
    suite = {"smac": "smac", "vmas": "vmas", "pong": "selfplay"}.get(env.split("_")[0].split(":")[0], "mpe")
    return ROOT / "checkpoints" / suite / src.name


def main():
    names = sys.argv[1:] or sorted(p.name for p in SRC.iterdir() if (p / "final.pt").exists())
    for name in names:
        src = SRC / name
        if not (src / "final.pt").exists():
            print(f"skip {name}: not finished (no final.pt)")
            continue
        dst = destination(src)
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "ckpt").mkdir(parents=True)
        for f in ("config.json", "final.pt", "evolution.mp4", "evolution_self.mp4", "evolution_bot.mp4"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        for f in src.glob("*.csv"):                  # log.csv + precomputed analyses (e.g. tournament.csv)
            shutil.copy2(f, dst / f.name)
        if dst.parent.name == "selfplay":            # self-play keeps every snapshot clip (shown in notebook 09)
            shutil.copytree(src / "videos", dst / "videos")
        for f in (src / "ckpt").glob("*.pt"):
            shutil.copy2(f, dst / "ckpt" / f.name)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"packaged {name:<28} -> {dst.parent.name}/  {size:5.1f} MB")


if __name__ == "__main__":
    main()
