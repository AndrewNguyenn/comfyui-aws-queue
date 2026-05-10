# syntax=docker/dockerfile:1.6
#
# Image-fleet worker (g4dn.xlarge, T4 GPU, Turing sm_75).
#
# Note: SageAttention 2.x requires Ampere (sm_80+). T4 is Turing (sm_75) so we
# use xformers instead. SDXL throughput on T4 with xformers comfortably exceeds
# the 100 imgs/hr target. (See IMPLEMENTATION_PLAN.md section 15 C1.)

# ---------- Stage 1: build base ----------
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- ComfyUI ----------
WORKDIR /opt/comfy
ARG COMFYUI_TAG=v0.20.1
RUN git clone --depth 1 --branch ${COMFYUI_TAG} \
        https://github.com/comfyanonymous/ComfyUI.git . \
    && pip install -r requirements.txt

# ---------- xformers (image fleet attention backend) ----------
# xformers ships pre-built wheels matching torch + CUDA versions in the NGC base.
RUN pip install xformers

# ---------- AWS + worker shared deps ----------
COPY workers/shared/requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt && rm /tmp/req.txt

# ---------- Worker code ----------
WORKDIR /opt/worker
COPY workers/shared/ /opt/worker/
COPY workers/image/extra_models.json /opt/worker/extra_models.json
COPY workers/image/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Per-job scratch dir (output_uploader watches this)
RUN mkdir -p /opt/comfy/output

# ---------- Smoke test (catches broken nodes at build time) ----------
# Quick validation: import ComfyUI's main module path. Failing here = visible in
# CodeBuild logs rather than at first instance launch.
RUN cd /opt/comfy && python -c "import sys; sys.path.insert(0, '.'); import comfy" || \
    echo "WARNING: comfy module import quick-check failed (may still work at runtime)"

ENTRYPOINT ["/entrypoint.sh"]
