# syntax=docker/dockerfile:1.7
#
# MARL workshop image: Python 3.10 + (CUDA) PyTorch + MPE2 / SMAClite / VMAS + JupyterLab,
# with the trained checkpoints baked in. Headless by design (no X11): frames are rendered
# with rgb_array and shown inline in notebooks, so the same image works on Linux/Win/Mac.
#
# Build context = repo root:   docker build -f docker/Dockerfile .      (see docker/build.sh)
#
# TORCH_FLAVOR: cu129 (default, GPU-capable, ~8 GB) | cpu (~2 GB).
# On arm64 (Apple Silicon) there are no CUDA wheels, so the CPU build is forced automatically.

# ---------------------------------------------------------------- builder
FROM python:3.10-slim AS builder
ARG TORCH_FLAVOR=cu129
ARG TARGETARCH
# Generous network settings: big CUDA wheels over slow or flaky Wi-Fi.
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10

RUN pip install --no-cache-dir uv

# The venv lives at the same absolute path in both stages so its shebangs stay valid.
COPY requirements.txt /tmp/requirements.txt
# The uv download cache is a BuildKit cache mount: it persists across builds (not in the image),
# so changing requirements.txt only downloads the changed packages, not all of torch again.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    FLAVOR="${TORCH_FLAVOR}"; \
    if [ "${TARGETARCH:-amd64}" != "amd64" ]; then FLAVOR=cpu; fi; \
    uv venv /opt/venv; \
    uv pip install --python /opt/venv/bin/python --torch-backend "${FLAVOR}" -r /tmp/requirements.txt; \
    # triton is only needed for torch.compile (not used in the workshop): saves ~0.7 GB.
    uv pip uninstall --python /opt/venv/bin/python triton || true

# ---------------------------------------------------------------- runtime
FROM python:3.10-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MESA_SHADER_CACHE_DISABLE=true \
    HOME=/home/marl \
    MARL_CKPT_DIR=/opt/marl-workshop/checkpoints \
    JUPYTER_TOKEN=marl

# VMAS renders with pyglet (OpenGL), which needs a display even for rgb_array frames:
# a virtual X server (Xvfb) + Mesa software GL make that work headlessly. ~60 MB.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb xauth libgl1 libglu1-mesa libgl1-mesa-dri fontconfig fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Any uid must be able to write HOME (run scripts pass --user $(id -u) on Linux so that
# files created in the bind-mounted repo are owned by the host user).
RUN useradd -m -u 1000 marl && chmod 777 /home/marl

COPY --from=builder /opt/venv /opt/venv

# Seed copy of the repo (code, notebooks, checkpoints). Used as-is for `docker run` without
# a bind mount; the entrypoint copies it into /workspace if that directory is empty.
COPY . /opt/marl-workshop
RUN chmod +x /opt/marl-workshop/docker/entrypoint.sh && \
    ln -s /opt/marl-workshop/docker/entrypoint.sh /usr/local/bin/entrypoint.sh && \
    mkdir -p /workspace && chown marl:marl /workspace

# Build-time smoke test: fail the build instead of shipping a broken image.
# (xvfb-run leaves root-owned lock/socket files behind which would block Xvfb at runtime: remove them.)
RUN cd /opt/marl-workshop && xvfb-run -a python verify_env.py --quick; rc=$?; \
    rm -f /tmp/.X*-lock; rm -rf /tmp/.X11-unix /tmp/matplotlib; exit $rc

ENV PYTHONPATH=/workspace
WORKDIR /workspace
USER marl
EXPOSE 8888
ENTRYPOINT ["entrypoint.sh"]
CMD ["jupyter"]
