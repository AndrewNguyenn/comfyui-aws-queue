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
        fuse libfuse2 xfsprogs \
    && rm -rf /var/lib/apt/lists/*

# Mountpoint for Amazon S3 — same role as in image worker. Each catalog type
# is mounted at /opt/comfy/models/<type>/ by entrypoint.sh; first read
# streams from S3 and caches on the NVMe instance store.
ARG MOUNT_S3_VERSION=1.10.0
RUN curl -fsSL -o /tmp/mount-s3.deb \
        "https://s3.amazonaws.com/mountpoint-s3-release/${MOUNT_S3_VERSION}/x86_64/mount-s3-${MOUNT_S3_VERSION}-x86_64.deb" \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/mount-s3.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/mount-s3.deb \
    && mount-s3 --version

# ---------- ComfyUI ----------
WORKDIR /opt/comfy
ARG COMFYUI_TAG=v0.27.0
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

# Pin by SHA — kijai's wrapper has no tagged releases. This SHA is the first
# one we validated that contains SCAIL/nodes.py (WanVideoAddSCAILPoseEmbeds /
# WanVideoAddSCAILReferenceEmbeds); anything older cannot run the SCAIL-2
# pipeline at all.
ARG WAN_WRAPPER_SHA=088128b224242e110d3906c6750e9a3a348a659b
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

# ---------- Mark the hand-cloned packs as baked ----------
# The runtime manifest_installer skips any pack carrying a .baked marker.
# bake_nodes.py writes these itself, but the three packs above are cloned
# directly by this Dockerfile and were missing markers — so a manifest listing
# any of them made every cold worker re-clone and re-pip-install a pack that
# was already in the image.
RUN for d in ComfyUI-WanVideoWrapper ComfyUI-VideoHelperSuite ComfyUI-Manager; do \
        echo "baked at image build" > /opt/comfy/custom_nodes/$d/.baked; \
    done

# ---------- AWS + worker shared deps ----------
COPY workers/shared/requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt && rm /tmp/req.txt

# ---------- Worker code ----------
WORKDIR /opt/worker
COPY workers/shared/ /opt/worker/
COPY workers/video/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---------- Bake the SCAIL-2 pose-control node packs ----------
# KJNodes (ImageResizeKJv2 / SimpleCalculatorKJ / ImageConcatMulti),
# WanAnimatePreprocess (ONNX ViTPose + YOLO detection) and SCAIL-pose (NLF
# predict + pose render). Routed through bake_nodes.py rather than more hand
# clones so they get the foundation repairs: WanVideoWrapper, KJNodes and both
# pose packs each pull an opencv variant, and mixed variants clobber one cv2/
# dir and crash ComfyUI outright on startup. _force_clean_opencv() runs last
# and collapses them to a single pinned headless build.
COPY workers/video/baked_nodes.txt /tmp/baked_nodes.txt
RUN cd /opt/worker && python3 bake_nodes.py /tmp/baked_nodes.txt

# ---------- Re-assert the torchaudio stub ----------
# ComfyUI v0.27.0 lists torchaudio in its own requirements.txt, so the core
# install pulls a REAL torchaudio 2.11.0 — built against torch 2.11/CUDA 13,
# while this NGC base is torch 2.5/CUDA 12.6. That mismatched pairing needs
# libcudart.so.13 and crash-loops ComfyUI on every boot. bake_nodes.py already
# re-stubs as its last step; assert it here too (hardcoded site-packages path)
# and verify the import, so a broken stub fails the BUILD instead of the fleet.
# Same treatment as the image worker — see workers/image/Dockerfile.
RUN pip uninstall -y torchaudio 2>/dev/null || true \
    && rm -rf /usr/local/lib/python3.10/dist-packages/torchaudio \
              /usr/local/lib/python3.10/dist-packages/torchaudio-*.dist-info \
    && mkdir -p /usr/local/lib/python3.10/dist-packages/torchaudio \
    && printf '%s\n' \
         '"""torchaudio stub — a real wheel needs a CUDA this base lacks."""' \
         '__version__ = "0.0.0-stub"' \
         > /usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py \
    && python -c "import torchaudio; print('torchaudio OK:', torchaudio.__version__)"

RUN mkdir -p /opt/comfy/output

# ---------- Smoke test ----------
RUN cd /opt/comfy && python -c "import sys; sys.path.insert(0, '.'); import comfy" || \
    echo "WARNING: comfy module import quick-check failed (may still work at runtime)"
RUN python -c "import sageattention; import torch; print('sage importable, torch:', torch.__version__)"

ENTRYPOINT ["/entrypoint.sh"]
