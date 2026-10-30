#!/usr/bin/env bash
# Single place for names. Sourced by the other scripts.
# TODO(maintainer): set GITHUB_USER once the public repo/registry namespace exists.
GITHUB_USER="${GITHUB_USER:-rajtilakls2510}"
IMAGE_NAME="marl-workshop"
REGISTRY_IMAGE="ghcr.io/${GITHUB_USER}/${IMAGE_NAME}"   # public pull address
LOCAL_IMAGE="${IMAGE_NAME}"                              # tag used by local builds / docker load
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="marl"
