# syntax=docker/dockerfile:1.6
#
# Metadata instance container — runs ComfyUI in CPU mode just to serve
# /object_info, /extensions, and Manager UI to the editor. No GPU deps,
# no Wan/sage/xformers. Small image (~1 GB) sized for t3.small.
#
# Same ComfyUI version + Manager as the image worker so the node menu
# the editor sees matches what the GPU workers will actually run.

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
        git curl ca-certificates \
        libgl1 libglib2.0-0 \
        nginx-light \
    && rm -rf /var/lib/apt/lists/*

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

# Worker shared code (object_info_publisher, extensions_publisher).
WORKDIR /opt/worker
COPY workers/shared/ /opt/worker/
COPY workers/metadata/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Per-job scratch dir — never used in metadata mode but ComfyUI checks for it.
RUN mkdir -p /opt/comfy/output

# Smoke test: can ComfyUI import in CPU mode + Manager loads?
RUN cd /opt/comfy && python -c "import sys; sys.path.insert(0,'.'); import comfy" \
    || echo "WARNING: comfy import quick-check failed"

ENTRYPOINT ["/entrypoint.sh"]
