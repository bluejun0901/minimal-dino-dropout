#!/usr/bin/env bash

set -e

# Local cache directories
export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

mkdir -p "$UV_CACHE_DIR"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"

echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "UV_LINK_MODE=$UV_LINK_MODE"