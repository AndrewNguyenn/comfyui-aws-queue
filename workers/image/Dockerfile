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
# Constraints to keep NGC's torch ecosystem intact:
#   1. Strip torch + torchvision + torchaudio from ComfyUI's requirements.txt
#      (NGC ships custom-built torch+torchvision whose C++ ops only register
#      if torch's ABI matches; pip would replace them with mismatched PyPI
#      wheels. NGC ships no torchaudio at all.)
#   2. Pin numpy<2 — NGC's torch built against numpy 1.x ABI.
#   3. Stub torchaudio: NGC has no torchaudio, but ComfyUI v0.20.1 hard-imports
#      it in comfy/ldm/lightricks/vae/audio_vae.py at module load. Any vanilla
#      torchaudio wheel either needs CUDA 13 (libcudart.so.13 not present) or
#      references torch C++ symbols that don't exist in NGC's customized torch
#      (undefined _ZNK5torch8autograd4Node4nameEv at runtime). Image workflows
#      never call into audio_vae, so a no-op stub satisfies the import without
#      breaking anything we actually run. If we add audio workflows later,
#      revisit by switching base image to vanilla CUDA + vanilla torch stack.
RUN git clone --depth 1 --branch ${COMFYUI_TAG} \
        https://github.com/comfyanonymous/ComfyUI.git . \
    && sed -i '/^torch\b/d;/^torchvision\b/d;/^torchaudio\b/d' requirements.txt \
    && pip install "numpy<2" \
    && pip install -r requirements.txt \
    && pip install "numpy<2" \
    && pip uninstall -y torchaudio 2>/dev/null || true \
    && mkdir -p /usr/local/lib/python3.10/dist-packages/torchaudio \
    && printf '%s\n' \
         '"""torchaudio stub — NGC ships no torchaudio, audio workflows unused on image fleet."' \
         'See workers/image/Dockerfile for why this exists."""' \
         '__version__ = "0.0.0-stub"' \
         > /usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py

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
