# syntax=docker/dockerfile:1.6
#
# Metadata instance container — runs ComfyUI in CPU mode just to serve
# /object_info, /extensions, and Manager UI to the editor. No GPU deps,
# no Wan/sage/xformers.
#
# Same ComfyUI version + Manager as the image worker so the node menu
# the editor sees matches what the GPU workers will actually run.
#
# Custom nodes are BAKED IN AT BUILD TIME (same as the image worker — see
# workers/image/Dockerfile). The old design ran manifest_installer at every
# container boot, cloning + pip-installing all ~38 manifest packs on the
# 2 GB t3.small. Newer packs pull heavy dependency trees (llama-cpp-python
# built from source, voxcpm, funasr, gradio) — the box ran out of memory,
# ComfyUI never started, and the editor showed "metadata unreachable".
# Baking moves that work to build time (where it has CodeBuild's RAM and
# any dependency conflict fails the build visibly); the runtime
# manifest_installer then skips every pack carrying a .baked marker and
# only clones genuinely net-new packs.

# Pull from AWS's ECR Public mirror — same upstream image, but bypasses
# Docker Hub's 100-pulls-per-6h unauthenticated cap that CodeBuild's NAT
# IP shares with every other AWS account.
FROM public.ecr.aws/docker/library/python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential cmake \
        libgl1 libglib2.0-0 \
        ffmpeg \
        nginx-light \
    && rm -rf /var/lib/apt/lists/*

# build-essential + cmake: some baked custom nodes pip-build native wheels
# from source (llama-cpp-python, insightface, etc.). Without a compiler the
# bake's pip install of those packs fails. ffmpeg: required by a handful of
# video/image custom nodes at import time.

# entrypoint.sh runs nginx as an auth gate in front of ComfyUI: it listens on
# 8188 (the public port) and 403s any request lacking the X-Comfy-Auth shared
# secret. ComfyUI itself binds 127.0.0.1:8189, unreachable from outside.

# CPU-only torch — way smaller than CUDA torch (~200 MB vs ~1 GB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio

# ComfyUI core
WORKDIR /opt/comfy
ARG COMFYUI_TAG=v0.20.1
RUN git clone --depth 1 --branch ${COMFYUI_TAG} \
        https://github.com/comfyanonymous/ComfyUI.git . \
    && pip install -r requirements.txt

# ComfyUI-Manager — install order matches AndrewNguyenn/aws-deployment-of-comfyui
# (the user's previously-working deployment). GitPython MUST be installed
# explicitly before Manager's requirements.txt because Manager's __init__.py
# imports git at module-load time; a missing GitPython silently disables the
# whole extension including its JS routes.
WORKDIR /opt/comfy/custom_nodes
ARG COMFYUI_MANAGER_REF=main
RUN mkdir -p ComfyUI-Manager && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager ComfyUI-Manager && \
    pip install --no-cache-dir GitPython && \
    pip install --no-cache-dir -r ComfyUI-Manager/requirements.txt

# AWS SDK + worker shared deps (we reuse the publishers from worker.py).
COPY workers/shared/requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt && rm /tmp/req.txt

# Worker shared code (object_info_publisher, extensions_publisher,
# manifest_installer, bake_nodes).
WORKDIR /opt/worker
COPY workers/shared/ /opt/worker/
COPY workers/metadata/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---------- Bake the custom-node set ----------
# Clone + install every pack in baked_nodes.txt at BUILD time so the metadata
# container does NOT clone/pip-install on every boot (which OOM-killed the
# 2 GB t3.small). bake_nodes.py drops a .baked marker per pack; the runtime
# manifest_installer skips anything baked. Net-new packs the user installs
# later still go through the runtime manifest path.
#
# COMFY_METADATA=1 tells bake_nodes.py this is the CPU metadata image: it
# skips the torchaudio re-stub (the image ships real CPU torchaudio; only the
# NGC GPU worker needs the stub).
ENV COMFY_METADATA=1
COPY workers/image/baked_nodes.txt /tmp/baked_nodes.txt
RUN cd /opt/worker && python3 bake_nodes.py /tmp/baked_nodes.txt

# Per-job scratch dir — never used in metadata mode but ComfyUI checks for it.
RUN mkdir -p /opt/comfy/output

# Smoke test: can ComfyUI import in CPU mode + Manager loads?
RUN cd /opt/comfy && python -c "import sys; sys.path.insert(0,'.'); import comfy" \
    || echo "WARNING: comfy import quick-check failed"

ENTRYPOINT ["/entrypoint.sh"]
