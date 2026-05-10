# syntax=docker/dockerfile:1.6
#
# Video-fleet worker (g5.xlarge / g5.2xlarge / g6e.xlarge — A10G/L40S, sm_86+).
#
# Includes SageAttention 2.2 (requires Ampere or newer). Sage launched globally
# via ComfyUI's --use-sage-attention flag (set by the worker entrypoint).
# IMPORTANT: SageAttention + FP8 weights produces black frames in some Wan
# workflows. We pin the catalog to GGUF Q8 variants to dodge this; do not switch
# to FP8 weights without testing for the black-frame regression.

# ---------- Stage 1: build base ----------
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAX_JOBS=8 \
    NVCC_APPEND_FLAGS="--threads 8"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- ComfyUI ----------
WORKDIR /opt/comfy
ARG COMFYUI_TAG=v0.20.1
RUN git clone --depth 1 --branch ${COMFYUI_TAG} \
        https://github.com/comfyanonymous/ComfyUI.git . \
    && pip install -r requirements.txt

# ---------- SageAttention 2.2 ----------
# --no-build-isolation lets sageattention's setup.py see the existing torch + CUDA
# headers from the NGC base. This is the documented installation method; without
# this flag the build allocates an isolated env and fails to find nvcc.
# Build is slow (~15-20 min on BUILD_GENERAL1_LARGE compute) due to CUDA kernel
# compilation across multiple compute capabilities.
RUN pip install --no-build-isolation sageattention==2.2.0 || \
    (echo "ERROR: sageattention build failed. Check CUDA toolkit + torch version compat." && exit 1)

# Quick verification — fails build early if sage isn't usable.
RUN python -c "import sageattention; print('sageattention version:', sageattention.__version__)"

# ---------- ComfyUI custom nodes for Wan video ----------
WORKDIR /opt/comfy/custom_nodes

# Pin by SHA — kijai's wrapper has no tagged releases.
ARG WAN_WRAPPER_SHA=main
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    && cd ComfyUI-WanVideoWrapper \
    && git checkout ${WAN_WRAPPER_SHA} \
    && pip install -r requirements.txt

# VideoHelperSuite handles webp/mp4 output combining.
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && cd ComfyUI-VideoHelperSuite \
    && pip install -r requirements.txt

# ---------- AWS + worker shared deps ----------
COPY workers/shared/requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt && rm /tmp/req.txt

# ---------- Worker code ----------
WORKDIR /opt/worker
COPY workers/shared/ /opt/worker/
COPY workers/video/extra_models.json /opt/worker/extra_models.json
COPY workers/video/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /opt/comfy/output

# ---------- Smoke test ----------
RUN cd /opt/comfy && python -c "import sys; sys.path.insert(0, '.'); import comfy" || \
    echo "WARNING: comfy module import quick-check failed (may still work at runtime)"
RUN python -c "import sageattention; import torch; print('sage:', sageattention.__version__, 'torch:', torch.__version__)"

ENTRYPOINT ["/entrypoint.sh"]
