#!/usr/bin/env python
"""Copy finished training runs from runs/<name>/ into checkpoints/mpe/<name>/ (what is shipped in
the repo and baked into the Docker image).

Kept: log.csv, config.json, ckpt/ (all snapshots - the analyses compare policies across training
time), final.pt, evolution.mp4. Not kept: per-snapshot clips in videos/ (notebooks regenerate
them from the snapshots).

    python scripts/package_runs.py                 # every finished run
    python scripts/package_runs.py sl_maddpg tag_maddpg
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "runs", ROOT / "checkpoints" / "mpe"


def main():
    names = sys.argv[1:] or sorted(p.name for p in SRC.iterdir() if (p / "final.pt").exists())
    for name in names:
        src, dst = SRC / name, DST / name
        if not (src / "final.pt").exists():
            print(f"skip {name}: not finished (no final.pt)")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "ckpt").mkdir(parents=True)
        for f in ("log.csv", "config.json", "final.pt", "evolution.mp4"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        for f in (src / "ckpt").glob("*.pt"):
            shutil.copy2(f, dst / "ckpt" / f.name)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"packaged {name:<28} {size:5.1f} MB")


if __name__ == "__main__":
    main()
