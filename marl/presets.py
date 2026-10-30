"""Named training runs. Every result in the notebooks can be reproduced with
    python scripts/train.py --run <name>
"""

PRESETS = {
    # ---- Speaker-Listener: cooperative non-stationarity + communication -----------------
    "sl_maddpg": dict(env="speaker_listener", algo="ddpg", critic="central", steps=1_000_000, eval_every=50_000),
    "sl_iddpg": dict(env="speaker_listener", algo="ddpg", critic="local", steps=1_000_000, eval_every=50_000),
    "sl_mappo": dict(env="speaker_listener", algo="ppo", critic="central", steps=2_000_000, eval_every=100_000),
    "sl_ippo": dict(env="speaker_listener", algo="ppo", critic="local", steps=2_000_000, eval_every=100_000),
    # ---- Predator-Prey: competitive non-stationarity / arms race (prey frozen first 25%) --
    "tag_maddpg": dict(env="tag", algo="ddpg", critic="central", steps=2_000_000, eval_every=100_000,
                       freeze={"prey": 500_000}),
    "tag_iddpg": dict(env="tag", algo="ddpg", critic="local", steps=2_000_000, eval_every=100_000,
                      freeze={"prey": 500_000}),
    # ---- Coop Navigation: MADDPG -> MAPPO bridge, coordination, parameter sharing --------
    # NOTE: shared-parameter MAPPO/IPPO huddle in the middle on continuous spread (see report);
    # the "spread_mappo_N*" presets are the shared ones, "_unshared" give each agent its own nets.
    "spread_maddpg_N3": dict(env="spread", N=3, algo="ddpg", critic="central", steps=3_000_000, eval_every=150_000,
                             buffer_size=500_000),
    "spread_maddpg_N6": dict(env="spread", N=6, algo="ddpg", critic="central", steps=3_000_000, eval_every=150_000,
                             buffer_size=300_000),
    "spread_mappo_N3": dict(env="spread", N=3, algo="ppo", critic="central", share=True, steps=3_000_000, eval_every=150_000),
    "spread_mappo_N4": dict(env="spread", N=4, algo="ppo", critic="central", share=True, steps=3_000_000, eval_every=150_000),
    "spread_mappo_unshared_N4": dict(env="spread", N=4, algo="ppo", critic="central", share=False, steps=3_000_000, eval_every=150_000),
    "spread_mappo_N6": dict(env="spread", N=6, algo="ppo", critic="central", share=True, steps=8_000_000, eval_every=400_000),
    "spread_ippo_N3": dict(env="spread", N=3, algo="ppo", critic="local", share=False, steps=3_000_000, eval_every=150_000),
    "spread_mappo_unshared_N3": dict(env="spread", N=3, algo="ppo", critic="central", share=False, steps=3_000_000, eval_every=150_000),
    "spread_mappo_unshared_N6": dict(env="spread", N=6, algo="ppo", critic="central", share=False, steps=8_000_000, eval_every=400_000),
}

LABELS = {"ddpg": {"central": "MADDPG", "local": "IDDPG"}, "ppo": {"central": "MAPPO", "local": "IPPO"}}
