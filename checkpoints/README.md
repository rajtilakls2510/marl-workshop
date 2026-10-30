# Pre-trained checkpoints

Training is not done live during the workshop; every demo loads a checkpoint from here.

Layout (filled in as the phases progress): `checkpoints/<suite>/<env>/<algo>/*.pt`
Load them with `marl.utils.ckpt_path("mpe", "simple_spread", "mappo", "final.pt")`.
