#!/usr/bin/env bash
# Video worker entrypoint. Sets sage attention env, hands off to worker.py.
set -euo pipefail

echo "comfy-video-worker starting on $(hostname)"
echo "fleet=${FLEET:-video} queue=${QUEUE_URL:-unset}"

# Reduce log noise (v3 N1).
export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=warning
export DIFFUSERS_VERBOSITY=warning

# SageAttention env: prefer Ampere/Ada-optimized kernels.
# (Sage is enabled at the ComfyUI process via --use-sage-attention, set in worker.py)
export SAGE_ATTENTION_BACKEND=auto

cd /opt/worker
exec python -u worker.py
