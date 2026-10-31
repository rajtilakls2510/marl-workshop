# Facilitator guide (maintainer-facing)

## Build / publish the image
```bash
./docker/build.sh                 # local image marl-workshop:latest  (GPU-capable torch, ~8 GB on disk)
FLAVOR=cpu ./docker/build.sh cpu  # slim CPU-only variant, tag :cpu (~2 GB) -- emergency option for slow Wi-Fi
GITHUB_USER=<you> ./docker/push.sh   # multi-arch push to ghcr.io/<you>/marl-workshop (see header of push.sh for login)
```
After the first push, set the package visibility to **Public** in GitHub (otherwise attendees can't pull anonymously).
The GitHub user (`rajtilakls2510`) is set in `docker/config.sh` and `docker/run.ps1`.

Why GHCR and not Docker Hub: Docker Hub throttles anonymous pulls per IP (100 / 6 h), and a whole room behind one venue NAT shares an IP.

## Setup is done live: plan for the bandwidth
The venue Wi-Fi will not survive N people pulling ~3-4 GB. Bring the offline kit:
1. `./docker/build.sh && ./facilitator/make_usb_kit.sh` -> `facilitator/kit/` (image tarball, repo zip, load scripts, docs). Copy to USB sticks.
2. For the Docker-free route: `./facilitator/make_wheelhouse.sh <linux|windows|macos-arm|macos-intel> cpu` and copy `kit/wheelhouse-*`.
3. LAN: `./facilitator/lan_serve.sh [--registry]` on your laptop (ideally on your own router/hotspot); tell people `http://<your-ip>:8000`.
   Registry route needs the attendee to allow the insecure registry once:
   * Docker Desktop: Settings -> Docker Engine -> add `"insecure-registries": ["<your-ip>:5000"]` -> Apply & restart.
   * Linux: `/etc/docker/daemon.json` same key, `sudo systemctl restart docker`.
   The HTTP tarball route needs no configuration, so prefer it.
4. Start the opening talk immediately after telling people the link; downloads run while you speak.

## Checklist before the day
- [ ] GHCR package set to public; repo public.
- [ ] Fresh-machine test of both routes on Windows and macOS (the `.ps1/.bat` scripts and the arm64 image are written but untested here).
- [ ] All checkpoints trained, packaged into `checkpoints/` (git-ignored) and baked into the image; `./docker/run.sh verify` green.
- [ ] USB kits and a LAN server rehearsed.
