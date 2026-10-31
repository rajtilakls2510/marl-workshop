# Pre-trained runs

Training is not done live during the workshop; every analysis notebook loads results from here.

**The runs are not in git** - they are baked into the Docker image (`/opt/marl-workshop/checkpoints`, found via
`$MARL_CKPT_DIR`). Without Docker, re-create a run with `python scripts/train.py --run <run>` (output in `runs/`).
Facilitators: `python scripts/package_runs.py` copies finished runs from `runs/` into this folder before building the image.

`mpe/<run>/` (MPE) and `smac/<run>/` (SMAClite) — one folder per training run (names as in `marl/presets.py`):
- `log.csv` — evaluation after every snapshot: returns + behaviour metrics, wall-clock time
- `ckpt/step_XXXXXXXX.pt` — policy snapshots across training (the analyses compare early vs late policies)
- `final.pt` — final actors and critics
- `evolution.mp4` — the same episode replayed with each snapshot
- `config.json` — everything needed to reproduce it: `python scripts/train.py --run <run>`

Load in Python: `from marl.analysis import load_run, load_policy; run = load_run("sl_maddpg")`.
