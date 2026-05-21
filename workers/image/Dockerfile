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
        fuse libfuse2 xfsprogs \
    && rm -rf /var/lib/apt/lists/*

# ---------- Mountpoint for Amazon S3 ----------
# Mount s3://comfy-models/<type>/ at /opt/comfy/models/<type>/ so ComfyUI sees
# every catalog model as a local file. mount-s3 streams from S3 on first access
# and caches blocks on local disk (the NVMe instance store, mounted at
# /opt/cache by entrypoint.sh). Replaces the old cache_manager.py download
# + symlink path.
# Pinned: mount-s3's cache semantics + --max-cache-size accounting are still
# being tuned across releases; an unpinned `latest` would silently swap in a
# new version on every build. Bump deliberately when a new feature is needed.
ARG MOUNT_S3_VERSION=1.10.0
RUN curl -fsSL -o /tmp/mount-s3.deb \
        "https://s3.amazonaws.com/mountpoint-s3-release/${MOUNT_S3_VERSION}/x86_64/mount-s3-${MOUNT_S3_VERSION}-x86_64.deb" \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/mount-s3.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/mount-s3.deb \
    && mount-s3 --version

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

# ---------- transformers + segment-anything (custom-node foundation) ----------
# ComfyUI's requirements floor lets pip grab the LATEST transformers, which
# now needs torch.distributed.tensor.device_mesh — absent from NGC's torch
# 2.5.x. That breaks transformers' lazy imports (AutoModel, BlipProcessor…),
# so every custom node that touches transformers (Easy-Use, LayerStyle,
# WAS, Florence2…) fails to import. Pin transformers to a torch-2.5-era
# release. segment-anything is a hard dep of ComfyUI-Impact-Pack that its
# requirements resolution drops — install it explicitly.
RUN pip install "transformers==4.46.3" "tokenizers<0.21" segment-anything

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
COPY workers/image/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---------- Bake common custom nodes ----------
# Clone + install the custom-node set at BUILD time so cold workers don't
# spend ~13 min doing it on every boot. Dependency conflicts surface here
# (visible in CodeBuild) instead of as a runtime ComfyUI crash-loop.
# bake_nodes.py reuses manifest_installer's logic (numpy/torch pin, single
# clean OpenCV, cv2 typing patch) and drops a .baked marker per pack; the
# runtime manifest_installer skips anything baked. Net-new packs the user
# installs later still go through the runtime manifest path.
COPY workers/image/baked_nodes.txt /tmp/baked_nodes.txt
RUN cd /opt/worker && python3 bake_nodes.py /tmp/baked_nodes.txt

# ---------- Re-assert the torchaudio stub ----------
# The bake pulls a real torchaudio (demucs / openai-whisper deps). ComfyUI
# v0.20.1 hard-imports torchaudio and a real wheel needs libcudart.so.13 —
# a CUDA the NGC image lacks → ComfyUI crashes on every boot. Definitively
# re-stub here (hardcoded site-packages path) and verify the import works,
# so a broken stub fails the BUILD instead of every worker.
RUN pip uninstall -y torchaudio 2>/dev/null || true \
    && rm -rf /usr/local/lib/python3.10/dist-packages/torchaudio \
              /usr/local/lib/python3.10/dist-packages/torchaudio-*.dist-info \
    && mkdir -p /usr/local/lib/python3.10/dist-packages/torchaudio \
    && printf '%s\n' \
         '"""torchaudio stub — NGC ships none; audio unused on the image fleet."""' \
         '__version__ = "0.0.0-stub"' \
         > /usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py \
    && python -c "import torchaudio; print('torchaudio OK:', torchaudio.__version__)"

# ---------- Patch SaveImageWithMetaData ----------
# comfyui_image_metadata_extension's metadata capture introspects ComfyUI's
# now-async execution cache synchronously → crashes → the image never saves.
# Make the capture non-fatal so the image always saves. Build-verified: the
# script exits non-zero (fails the build) if the anchor is missing.
COPY workers/image/patch_metadata_ext.py /tmp/patch_metadata_ext.py
RUN python3 /tmp/patch_metadata_ext.py

# ---------- Patch ComfyUI's model-folder cache ----------
# ComfyUI caches the model list and only re-scans on a dir mtime change —
# but mount-s3 never bumps mtimes when a model is added to S3, so a model
# added to a running worker is invisible until restart (→ 0-output jobs).
# Add a 60 s TTL so newly-added models are picked up. Build-verified.
COPY workers/image/patch_folder_cache.py /tmp/patch_folder_cache.py
RUN python3 /tmp/patch_folder_cache.py

# Per-job scratch dir (output_uploader watches this)
RUN mkdir -p /opt/comfy/output

# ---------- Smoke test (catches broken nodes at build time) ----------
# Quick validation: import ComfyUI's main module path. Failing here = visible in
# CodeBuild logs rather than at first instance launch.
RUN cd /opt/comfy && python -c "import sys; sys.path.insert(0, '.'); import comfy" || \
    echo "WARNING: comfy module import quick-check failed (may still work at runtime)"

ENTRYPOINT ["/entrypoint.sh"]
