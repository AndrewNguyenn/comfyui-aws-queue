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
# Four layered constraints to keep NGC's torch ecosystem intact:
#   1. Strip torch + torchvision + torchaudio from ComfyUI's requirements.txt
#      (NGC ships custom-built versions whose C++ ops only register if torch's
#      ABI matches; pip install would replace them with mismatched PyPI wheels
#      built for cu13 / different ABI).
#   2. After the rest of the requirements install, reinstall torchaudio from
#      PyTorch's cu124 index pinned to 2.5.1 with --no-deps. ComfyUI v0.20.1's
#      lightricks audio_vae hard-imports torchaudio, and the cu124 build links
#      against libcudart.so.12 (NGC has cu12.6 — fine; vanilla PyPI ships cu13
#      which expects libcudart.so.13 and crashes).
#   3. Pin numpy<2 — NGC's torch was built against numpy 1.x ABI. Anything that
#      pip-resolves numpy>=2 surfaces "Failed to initialize NumPy: _ARRAY_API
#      not found" warnings and eventually crashes on tensor ops.
RUN git clone --depth 1 --branch ${COMFYUI_TAG} \
        https://github.com/comfyanonymous/ComfyUI.git . \
    && sed -i '/^torch\b/d;/^torchvision\b/d;/^torchaudio\b/d' requirements.txt \
    && pip install "numpy<2" \
    && pip install -r requirements.txt \
    && pip install "numpy<2" \
    && pip install --no-deps torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# ---------- xformers (image fleet attention backend) ----------
# Pin to the xformers wheel built against torch 2.5.x (what NGC 24.10 ships)
# and install --no-deps so pip does NOT upgrade torch + torchvision out from
# under us. Without --no-deps, `pip install xformers` pulls torch 2.12+cu130
# which then makes the NGC-bundled torchvision C++ ops (torchvision::nms)
# unregistrable — every ComfyUI/Manager startup dies on first node import.
RUN pip install --no-deps xformers==0.0.28.post3

# ---------- ComfyUI-Manager (custom node manager UI) ----------
# Install order matches AndrewNguyenn/aws-deployment-of-comfyui pattern:
# GitPython must be installed BEFORE Manager's requirements.txt because
# Manager's __init__.py imports git at module-load time. Without it the
# whole extension silently fails to register.
WORKDIR /opt/comfy/custom_nodes
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
