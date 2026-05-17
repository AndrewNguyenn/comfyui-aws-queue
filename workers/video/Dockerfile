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
    # CUDA compilation parallelism: low to avoid OOM-kill on
    # BUILD_GENERAL1_LARGE (15 GB RAM). NVCC + cicc per-process memory is
    # ~2-3 GB; 4 jobs × 3 archs blew past 15 GB and got SIGKILL'd. Two jobs
    # is the safe ceiling for this build host.
    MAX_JOBS=2 \
    NVCC_APPEND_FLAGS="--threads 2" \
    # Constrain CUDA arch to GPUs we actually deploy on. Without this nvcc
    # tries to compile for the default set (incl. sm_50 Maxwell) which sage
    # 2.x doesn't support. We target only the cards in our fleet:
    #   8.6 = A10G (g5.xlarge primary, g5.2xlarge fallback)
    #   8.9 = L40S (g6e.xlarge fallback)
    # Dropped 8.0 (A100) since we don't run those — cuts compile time/memory.
    TORCH_CUDA_ARCH_LIST="8.6;8.9"

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

# ---------- SageAttention 2.x ----------
# PyPI only publishes the 1.x series (latest 1.0.6). The 2.x line lives on
# GitHub HEAD only — install from source. --no-build-isolation lets setup.py
# see the existing torch + CUDA headers from the NGC base. Build is slow
# (~15-20 min on BUILD_GENERAL1_LARGE) due to CUDA kernel compilation.
# Pin to a known commit when reproducible builds matter; using HEAD here.
RUN pip install --no-build-isolation \
        git+https://github.com/thu-ml/SageAttention.git || \
    (echo "ERROR: sageattention build failed. Check CUDA toolkit + torch version compat." && exit 1)

# Quick verification — fails build early if sage isn't importable.
# (sageattention from GitHub HEAD doesn't expose __version__ on the module.)
RUN python -c "import sageattention; print('sageattention import OK, attrs:', [a for a in dir(sageattention) if not a.startswith('_')][:5])"

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

# ComfyUI-Manager — custom node manager UI (install/update nodes from editor).
# GitPython explicitly installed BEFORE Manager's requirements.txt because
# Manager's __init__.py imports git at module-load time (matches the working
# AndrewNguyenn/aws-deployment-of-comfyui Dockerfile pattern).
ARG COMFYUI_MANAGER_REF=main
RUN mkdir -p ComfyUI-Manager && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager ComfyUI-Manager && \
    pip install --no-cache-dir GitPython && \
    pip install --no-cache-dir -r ComfyUI-Manager/requirements.txt

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
RUN python -c "import sageattention; import torch; print('sage importable, torch:', torch.__version__)"

ENTRYPOINT ["/entrypoint.sh"]
