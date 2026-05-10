#!/usr/bin/env bash
# Image worker entrypoint. Hands off to worker.py which manages ComfyUI subprocess.
set -euo pipefail

echo "comfy-image-worker starting on $(hostname)"
echo "fleet=${FLEET:-image} queue=${QUEUE_URL:-unset}"

# Reduce noise from common chatty tools (v3 N1 — log spend control).
export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=warning
export DIFFUSERS_VERBOSITY=warning

cd /opt/worker
exec python -u worker.py
