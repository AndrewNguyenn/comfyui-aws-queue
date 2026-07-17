"""
Krea 2 workflow substitution.

The "Krea 2" family (Krea) is a **Qwen-Image-like
architecture**, not SDXL. A Krea 2 model loads as three separate pieces —
``UNETLoader(<krea2>.safetensors)`` + ``CLIPLoader(qwen3vl_4b_bf16.safetensors,
type=krea2)`` + ``VAELoader(wan21_vae_fp32.safetensors)`` (Wan 2.1's VAE, shared
across several modern arches). Exactly like Anima (Qwen-arch) and Z-Image
(Lumina2-arch), a normal SDXL/checkpoint workflow that loads a Krea 2 model
through ``CheckpointLoaderSimple`` gets a ``None`` CLIP and dies — masked as an
840 s worker timeout because the worker only polls ``/history``.

So, mirroring ``zimage.py``/``anima.py``: when a submitted prompt references a
Krea 2 model we **discard the submitted graph** and run the official Krea 2
"simple gen" workflow instead, carrying over (a) the user's chosen Krea 2 model
and (b) their positive/negative prompt text.

We substitute the **Krea2Simple** template (``krea_templates/``): UNETLoader ->
three BAKED LoraLoaderModelOnly passes (turbo lora @0.6, filter-bypass lora
@1.0, amateur slider @1.5) -> two ClownsharKSampler_Beta passes (a 6-step base
pass + a 2-step low-denoise refiner pass) -> VAEDecode -> SaveImage. The turbo
+ filter-bypass loras are REQUIRED for these fast 6+2-step settings to produce
good output — that's why they're baked into the template rather than left to
the caller (a plain UNETLoader without them would need ~20-30 steps). The
amateur slider (civitai 2773343) counteracts the turbo pipeline's waxy
"plastic skin" tendency with amateur-photo noise/texture — the same role
igbaddie plays for Z-Image, but baked server-side. cfg is 1.4 (the published
workflow uses 1.0, where the negative prompt is inert; 1.4 re-enables the
negative at a modest speed cost — 2026-07-17 anti-plastic pass).

Detection is an **explicit allowlist** (``KREA_MODELS``) — deliberately NOT a
name heuristic, same rationale as Anima/Z-Image. DUPLICATED in
``comfytaggenerator/app.js`` (``KREA_MODELS``). Keep both in sync when adding a
model — the frontend can't import this.

This template is **txt2img only** — unlike Z-Image there is no VLM/reference-
image branch, so ``maybe_rewrite_to_krea`` accepts ``input_image_name`` only
for call-site symmetry with ``maybe_rewrite_to_zimage`` and ignores it.

The substitution never raises: any failure leaves the original workflow intact.
"""
from __future__ import annotations

import copy
import json
import os
import random
from typing import Optional

# Reuse Anima's prompt tracer + LoRA option validator verbatim — the frontend
# submits the same catpony graph (sampler.positive -> CLIPTextEncode.text ->
# String Literal) and the same {name, strength} lora option shape, so both are
# identical across families. Public functions; no private coupling.
from anima import extract_prompts, resolve_loras

# Match the other families' seed cap (rgthree's Seed node ceiling, kept for
# cross-family consistency; krea's seeds are plain sampler literals).
_SEED_MAX = 2 ** 50

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Explicit allowlist of Krea 2 model *filenames*, normalized (basename,
# extension stripped, lowercased). Add new Krea 2 models here as they are
# cataloged. A Krea 2 model must be cataloged as ``diffusion_models`` (and live
# under the diffusion_models/ S3 prefix) so the template's UNETLoader finds it.
# DUPLICATED in comfytaggenerator/app.js (KREA_MODELS). Keep both in sync.
KREA_MODELS: frozenset[str] = frozenset(
    {
        "krea2_raw_fp8_scaled",
        "krea2_raw_bf16",  # not yet cataloged — harmless until then, same convention as zimage's official variants
    }
)

# Node input keys that carry a model filename on a loader-style node. Duplicated
# from zimage.py/anima.py rather than imported — each family module keeps its
# own copy (see zimage.py's module docstring for the rationale).
_MODEL_INPUT_KEYS = ("ckpt_name", "unet_name", "model_name")
_MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".sft", ".gguf", ".bin")


def _normalize_model(name: str) -> str:
    base = os.path.basename(str(name)).strip()
    low = base.lower()
    for ext in _MODEL_EXTS:
        if low.endswith(ext):
            return low[: -len(ext)]
    return low


def detect_krea_model(workflow: dict) -> Optional[str]:
    """Return the original model filename if ``workflow`` references a Krea 2
    model, else ``None``. Scans every node's loader-style inputs."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in _MODEL_INPUT_KEYS:
            val = inputs.get(key)
            if isinstance(val, str) and _normalize_model(val) in KREA_MODELS:
                return val
    return None


# ---------------------------------------------------------------------------
# Template + injection
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "krea_templates", "Krea2Simple.api.json"
)
_template_cache: Optional[dict] = None

# Node ids in Krea2Simple (hand-authored, see krea_templates/Krea2Simple.api.json).
# Used with defensive .get guards; the template is version-controlled here so
# the ids are stable.
_UNET_CLASS = "UNETLoader"
_CLIP_CLASS = "CLIPLoader"
_POSITIVE_NODE = "5"        # CLIPTextEncode (Positive)
_NEGATIVE_NODE = "6"        # CLIPTextEncode (Negative)
_BAKED_TAIL_NODE = "13"     # LoraLoaderModelOnly (Amateur Slider) — baked chain tail


def _load_template() -> dict:
    global _template_cache
    if _template_cache is None:
        with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
            _template_cache = json.load(fh)
    return copy.deepcopy(_template_cache)


def _inject_seed(workflow: dict, seed: int) -> None:
    """Set a concrete seed on every literal seed widget (both
    ClownsharKSampler_Beta passes' ``seed`` int)."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), int):
                inputs[key] = seed


# ---------------------------------------------------------------------------
# LoRA stack injection
# ---------------------------------------------------------------------------
#
# ``krea_options.loras`` is an optional list of {"name": "<file>", "strength": f}
# entries. Validation (``resolve_loras``) is SHARED with anima.py/zimage.py —
# same options shape, same limits — imported rather than duplicated. Only the
# graph-splice strategy differs here: a chain of CORE ``LoraLoader`` nodes
# spliced between the BAKED CHAIN TAIL (node "3", the filter-bypass
# LoraLoaderModelOnly — i.e. user loras layer ON TOP of the required turbo +
# filter-bypass pair) / the CLIPLoader (node "4") and ALL of their direct
# consumers (both ClownsharKSampler_Beta passes' ``model``, both
# CLIPTextEncodes' ``clip``). Mirrors zimage._inject_loras's generic
# scan-and-rewire. LoRA files come from the catalog's ``lora`` type (mounted
# at models/loras/), value = the filename.

_LORA_CHAIN_BASE_ID = 9001   # chain node ids 9001.. (template ids are all <= 12)


def _inject_loras(wf: dict, loras: list[dict]) -> None:
    """Insert a chain of core LoraLoader nodes between the baked chain tail /
    CLIPLoader and ALL of their direct consumers, then rewire every other
    node's direct ``[<tail_id>, 0]`` model link / ``[<clip_id>, 0]`` clip link
    onto the chain tail (LoraLoader output 0 = MODEL, 1 = CLIP). The chain
    nodes themselves are exempt from rewiring. No-op (graph unchanged) when
    the list is empty or the anchor nodes aren't found."""
    if not loras:
        return
    tail_id = _BAKED_TAIL_NODE if isinstance(wf.get(_BAKED_TAIL_NODE), dict) else None
    clip_id = None
    for nid, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") == _CLIP_CLASS:
            clip_id = nid
            break
    if tail_id is None or clip_id is None:
        return

    prev_model, prev_clip = [tail_id, 0], [clip_id, 0]
    chain_ids: set[str] = set()
    for i, lora in enumerate(loras):
        nid = str(_LORA_CHAIN_BASE_ID + i)
        chain_ids.add(nid)
        wf[nid] = {
            "inputs": {
                "lora_name": lora["name"],
                "strength_model": lora["strength"],
                "strength_clip": lora["strength"],
                "model": list(prev_model),
                "clip": list(prev_clip),
            },
            "class_type": "LoraLoader",
            "_meta": {"title": f"LoRA {i + 1}: {lora['name']}"},
        }
        prev_model, prev_clip = [nid, 0], [nid, 1]

    for nid, node in wf.items():
        if nid in chain_ids or not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key, val in list(inputs.items()):
            if not (isinstance(val, list) and len(val) == 2):
                continue
            src, slot = str(val[0]), val[1]
            if src == str(tail_id) and slot == 0:
                inputs[key] = list(prev_model)
            elif src == str(clip_id) and slot == 0:
                inputs[key] = list(prev_clip)


def build_krea_workflow(
    model: str,
    positive: str,
    negative: str,
    options: Optional[dict] = None,
) -> dict:
    """Return the Krea 2 template with the user's model + prompts injected:
    model -> every ``UNETLoader.unet_name``; positive/negative -> the two
    CLIPTextEncode nodes; plus a fresh valid seed on both sampler passes.

    ``options["loras"]`` (see ``_inject_loras``) chains optional core
    LoraLoader nodes onto the baked chain tail / CLIPLoader; absent/empty
    leaves the workflow byte-identical (seed aside) to today's output.

    The prompt nodes are ALWAYS overwritten (even with an empty string) so the
    template's authored sample prompt (empty by default) can never leak stray
    user text across jobs."""
    workflow = _load_template()

    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == _UNET_CLASS:
            node.setdefault("inputs", {})["unet_name"] = model

    pos = workflow.get(_POSITIVE_NODE)
    if isinstance(pos, dict):
        pos.setdefault("inputs", {})["text"] = positive
    neg = workflow.get(_NEGATIVE_NODE)
    if isinstance(neg, dict):
        neg.setdefault("inputs", {})["text"] = negative

    # Optional LoRA stack (krea_options.loras) — chained core LoraLoader nodes
    # spliced between the baked chain tail/CLIPLoader and their consumers.
    # Never raises; sanitized + capped. No-op when no valid loras are given, so
    # the workflow stays unchanged from today's output.
    try:
        _inject_loras(workflow, resolve_loras(options))
    except Exception:  # noqa: BLE001 — a bad lora list must never break submission
        pass

    _inject_seed(workflow, random.randint(0, _SEED_MAX - 1))
    return workflow


def maybe_rewrite_to_krea(
    workflow: dict,
    input_image_name: Optional[str] = None,
    options: Optional[dict] = None,
) -> Optional[dict]:
    """If ``workflow`` references a Krea 2 model, return the Krea 2 workflow
    carrying over the model + prompts; else ``None`` (caller keeps the
    original). Never raises — on any failure returns ``None``.

    ``input_image_name`` is accepted only for call-site symmetry with
    ``maybe_rewrite_to_zimage`` and is IGNORED — the Krea2Simple template is
    txt2img only, with no VLM/reference-image branch.

    Like Anima/Z-Image: when a Krea 2 model is detected we always substitute,
    even if prompt extraction comes back empty — the original is the exact
    graph that fails with a None-CLIP error, so substituting at least runs the
    correct architecture."""
    del input_image_name  # unused — txt2img-only template, kept for symmetry
    try:
        model = detect_krea_model(workflow)
        if not model:
            return None
        positive, negative = extract_prompts(workflow)
        return build_krea_workflow(model, positive, negative, options)
    except Exception:  # noqa: BLE001 — substitution must never break submission
        return None
