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
# Two build PROFILES from one Dockerfile, selected by build args. Everything
# torch-version-specific below keys off these, so the Ada and Blackwell images
# cannot drift apart the way two Dockerfiles would.
#
#   ada (default)   NGC 24.10  torch 2.5.0a0  CUDA 12.6  py3.10   sm_86, sm_89
#                   -> comfy-video-worker            g5 / g6 / g6e fleets
#   blackwell       NGC 26.07  torch 2.13.0a0 CUDA 13.3  py3.12   + sm_120
#                   -> comfy-video-worker-blackwell  g7e fleet
#
# Blackwell (RTX PRO 6000, compute capability 12.0) cannot run the ada image
# at all: CUDA 12.8 is the first toolkit whose nvcc emits sm_120, and an
# explicit TORCH_CUDA_ARCH_LIST without +PTX ships no PTX to JIT from, so the
# first kernel launch hard-fails. See codebuild/buildspec-video-worker*.yml.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${BASE_IMAGE} AS base

# Re-declared after FROM (Docker scopes pre-FROM ARGs to the FROM line only).
ARG TORCH_ARCH_LIST="8.6;8.9"
# "ada" | "blackwell" — gates the torch-2.5-only workarounds below.
ARG BUILD_PROFILE=ada
# Site-packages root, for the patch scripts. 24.10 is py3.10, 26.07 is py3.12.
ARG PYTHON_SITE=/usr/local/lib/python3.10/dist-packages

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
    # 2.x doesn't support. Per profile (see the ARG above the FROM):
    #   8.6  = A10G   (g5)
    #   8.9  = L40S   (g6e)      + L4 (g6)
    #   12.0 = RTX PRO 6000 (g7e, Blackwell profile only; +PTX so a future
    #          arch can at least JIT instead of hard-failing)
    # Dropped 8.0 (A100) since we don't run those — cuts compile time/memory.
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" \
    BUILD_PROFILE="${BUILD_PROFILE}" \
    PYTHON_SITE="${PYTHON_SITE}"

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
ARG COMFYUI_TAG=v0.33.0
RUN git clone --depth 1 --branch ${COMFYUI_TAG} \
        https://github.com/comfyanonymous/ComfyUI.git . \
    && pip install -r requirements.txt

# ---------- Pin torchaudio to the NGC torch (ada profile) ----------
# MiniMax H3 is native from ComfyUI v0.30.0 and comfy_extras/nodes_minimax_h3.py
# does `import torchaudio` at module load — with a stub the import fails and
# EVERY MiniMax node silently disappears from the graph.
#
# BLACKWELL PROFILE: the 2.5.1/cu124 reinstall below is ada-only, but the
# extension PATCH applies to both. First blackwell build proved it: NGC 26.07
# ships a custom torch on CUDA 13.3, while ComfyUI's requirements resolve the
# PyPI torchaudio 2.11 built for CUDA 13.0, and torchaudio's
# _check_cuda_version() refuses that pairing at import:
#   "PyTorch has CUDA version 13.3 whereas TorchAudio has CUDA version 13.0"
# Same failure shape as ada (a compiled half that cannot load), same cure —
# patch_torchaudio_ext.py wraps that exact call and degrades to the pure-
# Python DSP, which is all MiniMax H3 uses. There is no matching wheel to
# install instead: nothing on PyPI is built against NGC's 13.3 torch.
#
# We cannot just take what ComfyUI's requirements.txt resolves to: that is
# torchaudio 2.11, built for torch 2.11 / CUDA 13, while this NGC base is
# 2.5.0a0+...nv24.10 on CUDA 12.6 — the libcudart.so.13 mismatch that
# crash-loops ComfyUI (the reason the image fleet stubs it at all).
#
# Install the matching 2.5.x instead, --no-deps so it cannot drag its own torch
# over NGC's custom build. Only torchaudio.functional.resample is actually used
# (in _encode_ref_audio) and that is pure PyTorch — a sinc kernel plus conv1d —
# so it works even if the compiled extension declines to load.
# ---------- Pin NumPy back under 2.0 ----------
# ComfyUI v0.33's dependency set pulls numpy 2.2.6, but this NGC base's torch
# 2.5 and its opencv are compiled against NumPy 1.x. Under numpy 2 the C
# bridge silently dies:
#
#   cv2 ... AttributeError: _ARRAY_API not found
#   torch.from_numpy(...) -> RuntimeError: Numpy is not available
#
# and the failure does not surface until a node actually converts an array —
# LoadImage failed this way AFTER a successful 84-second prompt execution, so
# the worker looked healthy right up to the point it produced nothing.
#
# 1.26.4 is what NGC 24.10 ships and what the image fleet still runs (its
# ComfyUI v0.27 asks for numpy>=1.25.0, which 1.26.4 already satisfies, so pip
# never upgraded it there — that is the only reason that fleet is unaffected).
# Downgrading is safe for opencv: wheels built against numpy 2 stay compatible
# with 1.x, while the reverse is not true.
#
# This runs BEFORE the bake so bake_nodes' generated constraint file pins
# numpy==1.26.4 and no custom node can drag it forward again.
# BLACKWELL PROFILE: torch 2.13 is built against NumPy 2.x, so the reason for
# this pin (NGC 24.10's torch/opencv compiled against 1.x) does not exist
# there. Leave whatever ComfyUI's requirements resolved.
ARG NUMPY_VERSION=1.26.4
RUN if [ "$BUILD_PROFILE" = "ada" ]; then \
        pip install --no-deps --force-reinstall "numpy==${NUMPY_VERSION}"; \
    else \
        echo "[$BUILD_PROFILE] keeping requirements-resolved numpy"; \
    fi \
    && python -c "import numpy, torch; \
t = torch.from_numpy(numpy.zeros(3, dtype='float32')); \
assert t.shape[0] == 3; \
print('numpy', numpy.__version__, '| torch.from_numpy OK ->', tuple(t.shape))"

# Guard the version bump's whole purpose: if COMFYUI_TAG is ever moved back
# below v0.30.0 this file disappears and every MiniMax node goes with it, which
# would otherwise only surface as "the nodes aren't in the editor".
RUN test -f /opt/comfy/comfy_extras/nodes_minimax_h3.py \
    || (echo "ERROR: comfy_extras/nodes_minimax_h3.py missing — COMFYUI_TAG=${COMFYUI_TAG} predates native MiniMax H3 (needs >= v0.30.0)"; \
        echo "--- diagnostics: pwd=$(pwd)"; ls -la /opt/comfy | head -20; echo "--- comfy_extras:"; ls /opt/comfy/comfy_extras 2>&1 | head -5; \
        echo "--- git:"; git -C /opt/comfy log --oneline -1 2>&1; git -C /opt/comfy status --short 2>&1 | head -5; \
        exit 1)

# The matching 2.5.1 wheel still cannot import as shipped: its prebuilt
# libtorchaudio.so is linked against UPSTREAM torch 2.5.1 while NGC ships a
# custom build, so the extension load dies with an undefined-symbol OSError and
# takes the whole import with it. patch_torchaudio_ext.py makes that load
# non-fatal — we only need the pure-python DSP, and the C++ half covers file
# I/O and sox, which nothing here uses.
ARG TORCHAUDIO_VERSION=2.5.1
COPY workers/video/patch_torchaudio_ext.py /tmp/patch_torchaudio_ext.py
RUN if [ "$BUILD_PROFILE" = "ada" ]; then \
        pip install --no-deps --force-reinstall \
            --index-url https://download.pytorch.org/whl/cu124 \
            torchaudio==${TORCHAUDIO_VERSION}; \
    else \
        echo "[$BUILD_PROFILE] keeping requirements-resolved torchaudio; patching its extension load"; \
    fi \
    && python3 /tmp/patch_torchaudio_ext.py \
    && python -c "import torch, torchaudio, torchaudio.transforms as T; \
print('torch', torch.__version__, '| torchaudio', torchaudio.__version__); \
w = torch.zeros(1, 2, 16000); \
r = torchaudio.functional.resample(w, 16000, 32000); \
assert r.shape[-1] == 32000, r.shape; \
m = T.MelSpectrogram(sample_rate=16000, n_mels=64)(torch.rand(1, 16000)); \
assert m.shape[1] == 64, m.shape; \
print('torchaudio DSP OK — resample', tuple(r.shape), 'mel', tuple(m.shape))"

# ---------- SageAttention 2.x ----------
# PyPI only publishes the 1.x series (latest 1.0.6). The 2.x line lives on
# GitHub HEAD only — install from source. --no-build-isolation lets setup.py
# see the existing torch + CUDA headers from the NGC base. Build is slow
# (~15-20 min on BUILD_GENERAL1_LARGE) due to CUDA kernel compilation.
# Pin to a known commit when reproducible builds matter; using HEAD here.
#
# FATAL on the ada profile, non-fatal on blackwell. The video fleet runs
# ComfyUI with --use-sage-attention (workers/shared/worker.py, FLEET=video),
# which imports sageattention at startup — a sage-less ada image would
# crash-loop that whole fleet while the build reported success. The minimax
# fleet never enables it, and it is the only fleet the blackwell profile
# serves, so there a failed sage build must not block the image that unlocks
# a GPU family. (It did build for sm_120 on 2026-08-26; this is the guard.)
RUN set -e; if pip install --no-build-isolation git+https://github.com/thu-ml/SageAttention.git \
       && python -c "import sageattention; print('sageattention import OK, attrs:', [a for a in dir(sageattention) if not a.startswith('_')][:5])"; then \
      echo "sageattention OK on profile=$BUILD_PROFILE arch=$TORCH_CUDA_ARCH_LIST"; \
    elif [ "$BUILD_PROFILE" = "ada" ]; then \
      echo "ERROR: sageattention build/import failed on profile=ada; the video fleet runs --use-sage-attention" >&2; exit 1; \
    else \
      echo "WARNING: no sageattention on profile=$BUILD_PROFILE arch=$TORCH_CUDA_ARCH_LIST (never enabled on minimax) — continuing"; \
    fi


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
# Read by manifest_installer at bake time AND on every worker boot. Without it
# the foundation repair re-stubs the torchaudio we just pinned — which is
# exactly what happened on the first build of this image, because the ENV was
# declared after the bake step instead of before it.
ENV COMFY_KEEP_TORCHAUDIO=1

COPY workers/video/baked_nodes.txt /tmp/baked_nodes.txt
RUN cd /opt/worker && python3 bake_nodes.py /tmp/baked_nodes.txt

# ---------- Patch comfy_kitchen for torch 2.5 ----------
# ComfyUI >= v0.30 imports comfy_kitchen from comfy/quant_ops.py, and its na3d
# custom op is declared with PEP 585 generics (list[int]) that torch 2.5 refuses
# to parse — the error propagates out of `import comfy.utils` and crash-loops
# ComfyUI before it can serve a single job. Rewrites that one signature to the
# typing.* forms torch 2.5 accepts; na3d keeps working.
# BLACKWELL PROFILE: torch 2.13 parses PEP 585 generics in custom-op
# signatures natively, so the patch is unnecessary — and its anchor may not
# even exist in the comfy_kitchen that 26.07's ComfyUI resolves, which would
# fail the build for no reason. Skipped.
COPY workers/video/patch_comfy_kitchen.py /tmp/patch_comfy_kitchen.py
RUN if [ "$BUILD_PROFILE" = "ada" ]; then \
        python3 /tmp/patch_comfy_kitchen.py; \
    else \
        echo "[$BUILD_PROFILE] torch >= 2.7 accepts PEP 585 custom-op signatures; patch skipped"; \
    fi

# ---------- Prove ComfyUI can actually import ----------
# The single most valuable check in this file. Everything above can succeed
# while ComfyUI still fails to start, which then surfaces only as a production
# crash-loop — exactly how the comfy_kitchen break was found. comfy.utils pulls
# in torch, quant_ops and comfy_kitchen, so importing it here turns that class
# of failure into a failed BUILD — it is what caught the comfy_kitchen break.
RUN cd /opt/comfy && python -c "import comfy.utils; print('comfy.utils import OK')"

# Node REGISTRATION cannot be verified here: nodes.py imports
# comfy.diffusers_load -> comfy.sd -> model_management, which calls
# torch.cuda.current_device() at import time, and CodeBuild has no GPU
# ("RuntimeError: Found no NVIDIA driver"). So check what is checkable without
# one — that the module parses and still declares the node id the dispatcher
# template references. comfy.utils above is the load-bearing check; this only
# guards against a rename.
RUN cd /opt/comfy && python -c "import ast; src=open('comfy_extras/nodes_minimax_h3.py').read(); ast.parse(src); assert 'MiniMaxH3ReferenceToVideo' in src, 'node id renamed'; print('nodes_minimax_h3 parses and declares MiniMaxH3ReferenceToVideo')"

# ---------- Restore the foundation after the bake ----------
# The bake's last step re-installs opencv and transformers to repair whatever a
# node did; those repairs now run under the constraint file, but re-pin numpy
# here anyway — this is the check that caught numpy 2.2.6 coming back, and it
# costs nothing to make the guarantee unconditional.
# Belt and braces: COMFY_KEEP_TORCHAUDIO should stop the foundation repair from
# touching it, but a custom node's requirements could still pull a different
# wheel. Fail the BUILD here rather than lose every MiniMax node in production.
# Re-apply the patch first: a node pack that reinstalled torchaudio would have
# restored the unpatched (fatal) extension load.
# Per profile: ada re-pins numpy 1.x + re-patches torchaudio 2.5 and asserts
# both; blackwell asserts the foundation is still torch-2.13-compatible
# (numpy 2.x, torchaudio 2.11+) and that nothing dragged either BACK.
RUN if [ "$BUILD_PROFILE" = "ada" ]; then \
        pip install --no-deps --force-reinstall "numpy==${NUMPY_VERSION}" \
        && python3 /tmp/patch_torchaudio_ext.py \
        && python -c "import numpy; assert numpy.__version__.startswith('1.'), 'numpy dragged forward: ' + numpy.__version__" \
        && python -c "import torchaudio; assert torchaudio.__version__.startswith('2.5'), 'torchaudio clobbered: ' + torchaudio.__version__"; \
    else \
        python3 /tmp/patch_torchaudio_ext.py \
        && python -c "import numpy; assert not numpy.__version__.startswith('1.'), 'numpy dragged BACK to 1.x on blackwell: ' + numpy.__version__" \
        && python -c "import torchaudio; v=tuple(int(x) for x in torchaudio.__version__.split('+')[0].split('.')[:2]); assert v >= (2, 11), 'torchaudio too old for torch 2.13: ' + torchaudio.__version__"; \
    fi \
    && python -c "import numpy, torch, cv2; \
torch.from_numpy(numpy.zeros(3, dtype='float32')); \
print('post-bake numpy', numpy.__version__, '| cv2', cv2.__version__, '| from_numpy OK')" \
    && python -c "import torch, torchaudio, torchaudio.transforms as T; \
print('post-bake torch', torch.__version__, '| torchaudio', torchaudio.__version__); \
w = torch.zeros(1, 2, 16000); \
assert torchaudio.functional.resample(w, 16000, 32000).shape[-1] == 32000; \
assert T.MelSpectrogram(sample_rate=16000, n_mels=64)(torch.rand(1, 16000)).shape[1] == 64; \
print('torchaudio OK after bake')"

RUN mkdir -p /opt/comfy/output

# ---------- Smoke test ----------
RUN cd /opt/comfy && python -c "import sys; sys.path.insert(0, '.'); import comfy" || \
    echo "WARNING: comfy module import quick-check failed (may still work at runtime)"
RUN set -e; if python -c "import sageattention; import torch; print('sage importable, torch:', torch.__version__)"; then :; \
    elif [ "$BUILD_PROFILE" = "ada" ]; then echo "ERROR: sage not importable in the final ada image" >&2; exit 1; \
    else python -c "import torch; print('sage NOT importable (non-fatal, never enabled on minimax), torch:', torch.__version__)"; fi

ENTRYPOINT ["/entrypoint.sh"]
