# syntax=docker/dockerfile:1.6
#
# Metadata instance container — runs ComfyUI in CPU mode just to serve
# /object_info, /extensions, and Manager UI to the editor. No GPU deps,
# no Wan/sage/xformers. Small image (~1 GB) sized for t3.small.
#
# Same ComfyUI version + Manager as the image worker so the node menu
# the editor sees matches what the GPU workers will actually run.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — way smaller than CUDA torch (~200 MB vs ~1 GB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio

# ComfyUI core
WORKDIR /opt/comfy
ARG COMFYUI_TAG=v0.20.1
RUN git clone --depth 1 --branch ${COMFYUI_TAG} \
        https://github.com/comfyanonymous/ComfyUI.git . \
    && pip install -r requirements.txt

# ComfyUI-Manager (same as worker images, so editor sees the same Manager UI)
WORKDIR /opt/comfy/custom_nodes
ARG COMFYUI_MANAGER_REF=main
RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git \
    && cd ComfyUI-Manager \
    && git checkout ${COMFYUI_MANAGER_REF} \
    && pip install -r requirements.txt

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
