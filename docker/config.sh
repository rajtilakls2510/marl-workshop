#!/usr/bin/env bash
# Single place for names. Sourced by the other scripts.
GITHUB_USER="${GITHUB_USER:-rajtilakls2510}"
IMAGE_NAME="marl-workshop"
REGISTRY_IMAGE="ghcr.io/${GITHUB_USER}/${IMAGE_NAME}"   # public pull address
REGISTRY_CPU="${REGISTRY_IMAGE}:cpu"                     # CPU-only PyTorch, ~2 GB (recommended)
REGISTRY_GPU="${REGISTRY_IMAGE}:gpu"                     # CUDA PyTorch, ~8 GB (only useful with an NVIDIA GPU)
LOCAL_IMAGE="${IMAGE_NAME}"                              # tag used by local builds / docker load
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="marl"
