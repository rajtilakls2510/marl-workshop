"""Named training runs. Every result in the notebooks can be reproduced with
    python scripts/train.py --run <name>
"""

_BUDGET = dict(steps=10**12, eval_every=10**12)    # time-budgeted runs: stop on `minutes`

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
    # ---- Predator-Prey: credit assignment ("lazy predator") - factored critics for the predators
    "tag_facvdn": dict(env="tag", algo="ddpg", critic="vdn", steps=2_000_000, eval_every=100_000,
                       freeze={"prey": 500_000}),
    # QMIX needs twin critics (TD3-style min): without them its mixer overestimates Q_tot ~70x and the
    # predators' utilities carry almost no action signal (see the Predator-Prey credit notebook)
    "tag_facqmix": dict(env="tag", algo="ddpg", critic="qmix", twin_q=True, steps=2_000_000, eval_every=100_000,
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
    # ---- Vectorization: the same MAPPO on VMAS's port of Coop Navigation (one batched simulation) ----
    # Same MAPPO (shared, central critic) everywhere; every run gets the same TRAINING-TIME budget and is
    # evaluated every 30 s of training time, so the curves compare quality vs wall-clock minutes.
    # Minibatch size: 800 time steps with 64 copies (as the mpe2 runs), 6,400 with large batches.
    "loop_spread_N3_cpu64": dict(env="spread", N=3, algo="ppo", critic="central", share=True, num_envs=64,
                                 device="cpu", minutes=10, eval_every_s=30, **_BUDGET),
    # VMAS episodes are 100 steps: VMAS agents are slower than mpe2's (0.88 vs 4.2 units in 25 steps at full
    # force), so with 25 steps most landmarks are physically out of reach (~75% of episodes).
    "vmas_spread_N3_cpu64": dict(env="vmas_spread", N=3, max_cycles=100, algo="ppo", critic="central", share=True,
                                 num_envs=64, minibatches=8, device="cpu", minutes=10, eval_every_s=30, **_BUDGET),
    "vmas_spread_N3_cpu1024": dict(env="vmas_spread", N=3, max_cycles=100, algo="ppo", critic="central", share=True,
                                   num_envs=1024, minibatches=16, device="cpu", minutes=10, eval_every_s=30, **_BUDGET),
    "vmas_spread_N3_gpu4096": dict(env="vmas_spread", N=3, max_cycles=100, algo="ppo", critic="central", share=True,
                                   num_envs=4096, minibatches=64, device="cuda", minutes=10, eval_every_s=30, **_BUDGET),
    "vmas_spread_N6_cpu64": dict(env="vmas_spread", N=6, max_cycles=100, algo="ppo", critic="central", share=True,
                                 num_envs=64, minibatches=8, device="cpu", minutes=30, eval_every_s=60, **_BUDGET),
    "vmas_spread_N6_gpu4096": dict(env="vmas_spread", N=6, max_cycles=100, algo="ppo", critic="central", share=True,
                                   num_envs=4096, minibatches=64, device="cuda", minutes=30, eval_every_s=60, **_BUDGET),
    # ---- Self-play on the hand-made two-player Pong (one network plays both sides) ---------------
    # ent_coef 0.002: with 0.01 the policy stayed near-random (entropy 0.88 of max 1.10) and Elo plateaued early
    "pong_selfplay_pool": dict(env="pong", algo="selfplay_ppo", opponent="pool", ent_coef=0.002, device="cuda",
                               minutes=25, eval_every_s=90, **_BUDGET),
    # ---- SMAClite: credit assignment (team reward) - MAPPO vs value decomposition --------
    "smac_2s3z_mappo": dict(env="smac", map="2s3z", algo="smac_mappo", steps=1_000_000, eval_every=50_000),
    "smac_2s3z_vdn": dict(env="smac", map="2s3z", algo="smac_vd", mixer="vdn", steps=1_000_000, eval_every=50_000),
    "smac_2s3z_qmix": dict(env="smac", map="2s3z", algo="smac_vd", mixer="qmix", steps=1_000_000, eval_every=50_000),
    "smac_mmm_mappo": dict(env="smac", map="MMM", algo="smac_mappo", steps=1_000_000, eval_every=100_000),
    "smac_mmm_vdn": dict(env="smac", map="MMM", algo="smac_vd", mixer="vdn", steps=1_000_000, eval_every=100_000),
    "smac_mmm_qmix": dict(env="smac", map="MMM", algo="smac_vd", mixer="qmix", steps=1_000_000, eval_every=100_000),
}

LABELS = {"ddpg": {"central": "MADDPG", "local": "IDDPG", "vdn": "FAC-VDN", "qmix": "FAC-QMIX"}, "ppo": {"central": "MAPPO", "local": "IPPO"}}
