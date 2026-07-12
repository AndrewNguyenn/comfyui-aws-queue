"""
Z-Image workflow substitution.

The "Z-Image" family (Tongyi-MAI, 6B) is a **Lumina2-architecture** text-to-image
model loaded as three separate pieces —
``UNETLoader(<z-image>.safetensors)`` + ``CLIPLoader(zImageTurbo_turbo_txt.safetensors,
type=lumina2)`` + ``VAELoader(ae_zimgturbo.safetensors)``. Exactly like Anima
(Qwen-arch), a normal SDXL/checkpoint workflow that loads a Z-Image model through
``CheckpointLoaderSimple`` gets a ``None`` CLIP and dies — masked as an 840 s worker
timeout because the worker only polls ``/history``.

So, mirroring ``anima.py``: when a submitted prompt references a Z-Image model we
**discard the submitted graph** and run the official Z-Image workflow template
instead, carrying over (a) the user's chosen Z-Image model, (b) their
positive/negative prompt text, and (c) an OPTIONAL user-uploaded reference image
routed into the workflow's VLM — Florence-2 (node 558/483) captions it and the
caption is joined into the positive prompt, so the model generates a series of
variations *described from* the uploaded image. With no image, the VLM branch is
pruned and it runs as plain txt2img from the user's prompt.

We substitute the **ZImageStandardFull** template (``zimage_templates/``): the
captured production graph — VLM caption + WD14 tags → prompt → KSampler → Ultimate
SD Upscale → hand FaceDetailer → film-grain/vignette post → SaveImageWithMetaData —
cleaned for headless (the missing-asset LUT + color-match LoadImage spliced out, the
un-baked ``Sampler Selector`` inlined, debug sinks dropped).

Detection is an **explicit allowlist** (``ZIMAGE_MODELS``) — deliberately NOT a name
heuristic, same rationale as Anima. DUPLICATED in ``comfytaggenerator/index.html``
(``ZIMAGE_MODELS`` / ``cbIsZImage``, which gate the upload UI). Keep both in sync
when adding a model — the frontend can't import this.

The substitution never raises: any failure leaves the original workflow intact.
"""
from __future__ import annotations

import copy
import json
import os
import random
from typing import Any, Optional

# Reuse Anima's prompt tracer verbatim — the frontend submits the same catpony
# graph (sampler.positive -> CLIPTextEncode.text -> String Literal), so the
# extraction is identical. Public function; no private coupling.
from anima import extract_prompts

# rgthree's Seed node validates seed <= 2**50; stay under it.
_SEED_MAX = 2 ** 50

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Explicit allowlist of Z-Image (Lumina2-arch) model *filenames*, normalized
# (basename, extension stripped, lowercased). Add new Z-Image models here as they
# are cataloged. A Z-Image model must be cataloged as ``diffusion_models`` (and live
# under the diffusion_models/ S3 prefix) so the template's UNETLoader finds it.
# DUPLICATED in comfytaggenerator/index.html (ZIMAGE_MODELS). Keep both in sync.
ZIMAGE_MODELS: frozenset[str] = frozenset(
    {
        # Community ZImageTurbo-base checkpoints (selectable in the prompt builder).
        "moodypromix_zitv13",
        "moodyrealmix_zitv7",
        "divingzimageturbo_v60fp16",
        # fp8 variants (~5.7GB) — the default for the cloud fleet: the fp16 full
        # stack + 3x SD-upscale overran the 840s worker cap, fp8 halves VRAM and
        # uses the L4 fp8 tensor cores. Template's text encoder is also fp8.
        "moodypromix_zitv12dpofp8",
        "moodyrealmix_zitv7fp8",
        "divingzimageturbo_v60fp8",
        # Official Tongyi-MAI variants (catalog when added; harmless until then).
        "zimagebase_base",
        "zimageturbo_turbo",
    }
)

# Node input keys that carry a model filename on a loader-style node.
_MODEL_INPUT_KEYS = ("ckpt_name", "unet_name", "model_name")
_MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".sft", ".gguf", ".bin")


def _normalize_model(name: str) -> str:
    base = os.path.basename(str(name)).strip()
    low = base.lower()
    for ext in _MODEL_EXTS:
        if low.endswith(ext):
            return low[: -len(ext)]
    return low


def detect_zimage_model(workflow: dict) -> Optional[str]:
    """Return the original model filename if ``workflow`` references a Z-Image
    model, else ``None``. Scans every node's loader-style inputs."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in _MODEL_INPUT_KEYS:
            val = inputs.get(key)
            if isinstance(val, str) and _normalize_model(val) in ZIMAGE_MODELS:
                return val
    return None


# ---------------------------------------------------------------------------
# Template + injection
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "zimage_templates", "ZImageStandardFull.api.json"
)
_template_cache: Optional[dict] = None

# Node ids in ZImageStandardFull (captured graph). Used with defensive .get guards;
# the template is version-controlled here so the ids are stable.
_UNET_CLASS = "UNETLoader"
_POSITIVE_CLASS = "easy positive"   # node 91 — inputs.positive
_NEGATIVE_CLASS = "easy negative"   # node 92 — inputs.negative
_STYLES_CLASS = "easy stylesSelector"  # node 178 — inputs.positive <- VLM join (458)
_SAVER_NODE = "260"                 # SaveImageWithMetaData — reachability root
# VLM branch node ids (input image -> Florence2 + WD14 -> join -> styles.positive).
_LOADIMAGE_NODE = "437"
_VLM_BRANCH_NODES = ("437", "483", "484", "480", "458", "558")


def _load_template() -> dict:
    global _template_cache
    if _template_cache is None:
        with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
            _template_cache = json.load(fh)
    return copy.deepcopy(_template_cache)


def _first_node_id(workflow: dict, class_type: str) -> Optional[str]:
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return nid
    return None


def _inject_seed(workflow: dict, seed: int) -> None:
    """Set a concrete seed on every literal seed widget (the Seed Generator's
    ``seed`` int; the sampler/upscale/detailer all read it through links)."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), int):
                inputs[key] = seed


def _reachable_from(wf: dict, root: str) -> set[str]:
    seen: set[str] = set()
    stack = [str(root)]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in wf:
            continue
        seen.add(nid)
        node = wf.get(nid)
        if not isinstance(node, dict):
            continue
        for val in (node.get("inputs") or {}).values():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], (str, int)):
                stack.append(str(val[0]))
    return seen


def _gc_unreachable(wf: dict, root: str) -> None:
    keep = _reachable_from(wf, root)
    for nid in [n for n in wf if n not in keep]:
        del wf[nid]


def _set_input_image(workflow: dict, image_name: str) -> None:
    """Point the VLM's LoadImage at the user-uploaded file (the worker stages it
    into ComfyUI's ``input/`` before the run). Keeps the VLM branch intact."""
    node = workflow.get(_LOADIMAGE_NODE)
    if isinstance(node, dict):
        node.setdefault("inputs", {})["image"] = image_name


def _prune_vlm_branch(workflow: dict) -> None:
    """No uploaded image → run plain txt2img: rewire the styles selector's
    positive straight from ``easy positive`` (skipping the VLM caption join),
    then GC the now-orphaned VLM nodes (LoadImage/Florence2/WD14/joins/loader)."""
    styles = _first_node_id(workflow, _STYLES_CLASS)
    pos = _first_node_id(workflow, _POSITIVE_CLASS)
    if styles and pos and isinstance(workflow.get(styles), dict):
        workflow[styles].setdefault("inputs", {})["positive"] = [pos, 0]
    # Drop the VLM nodes explicitly, then sweep anything else now unreachable.
    for nid in _VLM_BRANCH_NODES:
        workflow.pop(nid, None)
    _gc_unreachable(workflow, _SAVER_NODE)


# Heavy post-processing the 16 GB image workers can't sustain, bypassed so
# Z-Image jobs complete reliably (see comfytaggenerator diagnosis 2026-07-12):
#   - FaceDetailer (node 80) loads a SECOND full SDXL checkpoint (~7 GB,
#     CheckpointLoaderSimple 485) on top of the ~13 GB z-image stack -> the OS
#     OOM-kills ComfyUI (Errno 111 :8188). This was the dominant failure.
#   - UltimateSDUpscale (node 42) is a tiled SD-refine that runs ~750 s and
#     overruns the 840 s worker cap (and its captured graph is even missing the
#     required batch_size input, which drops the save chain at validation).
# We rewire the cheap post chain (sharpen/gamma/film-grain/vignette) onto the
# base decoded image, so both heavy branches become unreachable from the saver
# and are GC'd. The base Z-Image turbo output is a complete image; an upscale/
# detail pass can be re-run on favorites later.
_BASE_IMAGE_NODE = "506"    # easy cleanGpuUsed — passthrough of the base VAEDecode (84)
_COMPOSITE_NODE = "542"     # ImageCompositeMasked — dest/source were FaceDetailer
_RESOLUTION_NODE = "546"    # Get resolution [Crystools] — image was FaceDetailer


def _bypass_heavy_postprocessing(workflow: dict) -> None:
    """Rewire the post chain onto the base decoded image so the FaceDetailer +
    UltimateSDUpscale become unreachable from the saver and get GC'd. No-op on any
    node the template no longer has (defensive, like the VLM prune)."""
    base = [_BASE_IMAGE_NODE, 0]
    comp = workflow.get(_COMPOSITE_NODE)
    if isinstance(comp, dict):
        comp.setdefault("inputs", {})["destination"] = list(base)
        comp["inputs"]["source"] = list(base)
    res = workflow.get(_RESOLUTION_NODE)
    if isinstance(res, dict):
        res.setdefault("inputs", {})["image"] = list(base)
    _gc_unreachable(workflow, _SAVER_NODE)


# ---------------------------------------------------------------------------
# Per-model sampler overrides (extensible; empty = template defaults)
# ---------------------------------------------------------------------------
#
# The template runs node 490 KSampler at the Z-Image turbo defaults (steps via the
# mxSlider 489 = 11, cfg 1.5, euler/simple). A model that wants different settings
# maps its normalized name -> {steps?, cfg?, sampler?, scheduler?}. Keyed per-model
# so one model's recommendation doesn't change every Z-Image job. No-op for unlisted.
_ZIMAGE_MODEL_SAMPLER: dict[str, dict] = {}

_KSAMPLER_NODE = "490"
_STEPS_SLIDER_NODE = "489"  # mxSlider feeding KSampler.steps


def _apply_model_sampler(workflow: dict, model: str) -> None:
    over = _ZIMAGE_MODEL_SAMPLER.get(_normalize_model(model))
    if not over:
        return
    ks = workflow.get(_KSAMPLER_NODE)
    if isinstance(ks, dict):
        inp = ks.setdefault("inputs", {})
        for k in ("cfg", "sampler_name", "scheduler"):
            src = {"sampler": "sampler_name"}.get(k, k)
            if src in over:
                inp[k] = over[src]
    if "steps" in over:
        slider = workflow.get(_STEPS_SLIDER_NODE)
        if isinstance(slider, dict):
            slider.setdefault("inputs", {})["Xi"] = int(over["steps"])
            slider["inputs"]["Xf"] = float(over["steps"])


def build_zimage_workflow(
    model: str,
    positive: str,
    negative: str,
    input_image_name: Optional[str] = None,
    options: Optional[dict] = None,
) -> dict:
    """Return the Z-Image template with the user's model + prompts injected:
    model -> every ``UNETLoader.unet_name``; positive -> the ``easy positive``
    node; negative -> the ``easy negative`` node; plus a fresh valid seed.

    ``input_image_name`` (a filename the worker has staged into ComfyUI's
    ``input/``) routes that image into the VLM. When absent the VLM branch is
    pruned and it runs as txt2img. ``options`` is reserved (future detailer/upscale
    toggles); unused today.

    The prompt nodes are ALWAYS overwritten (even with an empty string) so the
    template's authored sample prompt can never leak into a user's job."""
    workflow = _load_template()

    _apply_model_sampler(workflow, model)

    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == _UNET_CLASS:
            node.setdefault("inputs", {})["unet_name"] = model

    pos_id = _first_node_id(workflow, _POSITIVE_CLASS)
    if pos_id:
        workflow[pos_id].setdefault("inputs", {})["positive"] = positive
    neg_id = _first_node_id(workflow, _NEGATIVE_CLASS)
    if neg_id:
        workflow[neg_id].setdefault("inputs", {})["negative"] = negative

    if input_image_name:
        _set_input_image(workflow, input_image_name)
    else:
        _prune_vlm_branch(workflow)

    # Bypass the OOM/timeout-inducing FaceDetailer + SD-upscale so the job runs
    # reliably on the 16 GB workers.
    _bypass_heavy_postprocessing(workflow)

    _inject_seed(workflow, random.randint(0, _SEED_MAX - 1))
    return workflow


def maybe_rewrite_to_zimage(
    workflow: dict,
    input_image_name: Optional[str] = None,
    options: Optional[dict] = None,
) -> Optional[dict]:
    """If ``workflow`` references a Z-Image model, return the Z-Image workflow
    carrying over the model + prompts (+ optional VLM input image); else ``None``
    (caller keeps the original). Never raises — on any failure returns ``None``.

    Like Anima: when a Z-Image model is detected we always substitute, even if
    prompt extraction comes back empty — the original is the exact graph that fails
    with a None-CLIP error, so substituting at least runs the correct architecture."""
    try:
        model = detect_zimage_model(workflow)
        if not model:
            return None
        positive, negative = extract_prompts(workflow)
        return build_zimage_workflow(model, positive, negative, input_image_name, options)
    except Exception:  # noqa: BLE001 — substitution must never break submission
        return None
