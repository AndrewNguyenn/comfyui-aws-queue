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
pass + a 2-step low-denoise refiner pass; the base pass's step/cfg settings are
overridable per model via ``_KREA_MODEL_SAMPLER``) -> VAEDecode -> SaveImage. The turbo
+ filter-bypass loras are REQUIRED for these fast 6+2-step settings to produce
good output — that's why they're baked into the template rather than left to
the caller (a plain UNETLoader without them would need ~20-30 steps). The
amateur slider (civitai 2773343) counteracts the turbo pipeline's waxy
"plastic skin" tendency with amateur-photo noise/texture — the same role
igbaddie plays for Z-Image, but baked server-side. cfg is 1.4 (the published
workflow uses 1.0, where the negative prompt is inert; 1.4 re-enables the
negative at a modest speed cost — 2026-07-17 anti-plastic pass).

The turbo lora is the one baked node that is NOT universal: a turbo/lightning
*checkpoint* already has that distillation merged in, so applying it again
overcooks the image. Those models are listed in ``KREA_PREBAKED_TURBO`` and
have the turbo node spliced out of the chain at build time.

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
        "krea2turbofp8_krea2turbo",       # "Krea 2 TURBO" fp8 (civitai 2723583 v3060999)
        "redcraftminimaxh3redmix_30krea2",  # RedCraft RedMix 3.0 Lightning8 fp8 (civitai 958009 v3139241)
    }
)

# Krea 2 checkpoints that already have the step-distillation MERGED INTO THE
# WEIGHTS (a "turbo"/"lightning" release, as opposed to the raw base model).
# For these the template's baked ``krea2_turbo_lora`` is a SECOND helping of the
# same distillation and visibly overcooks the image (blown contrast, plastic
# skin, collapsed detail), so we drop that one node from the chain and run the
# checkpoint on its own acceleration. Everything else about the template —
# filter-bypass lora, amateur slider, the two sampler passes — is unchanged;
# per-model step/cfg differences go in ``_KREA_MODEL_SAMPLER`` below, not here.
#
# Membership is a property of the WEIGHTS, not of the filename, so this is an
# explicit allowlist for the same reason KREA_MODELS is. A model listed here
# must also be in KREA_MODELS; a KREA_MODELS entry absent here keeps the baked
# turbo lora (correct for the raw base models).
KREA_PREBAKED_TURBO: frozenset[str] = frozenset(
    {
        "krea2turbofp8_krea2turbo",
        "redcraftminimaxh3redmix_30krea2",
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
_TURBO_LORA_NODE = "2"      # LoraLoaderModelOnly (Turbo, baked) — dropped for KREA_PREBAKED_TURBO
_BASE_SAMPLER_NODE = "8"    # ClownsharKSampler_Beta (base pass, denoise 1.0)
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
# Per-model sampler overrides (extensible; unlisted = template defaults)
# ---------------------------------------------------------------------------
#
# Transplanted from ``zimage.py``'s ``_ZIMAGE_MODEL_SAMPLER``/``_apply_model_sampler``
# — same shape, same "keyed per-model so one model's recommendation doesn't
# change every job" rationale. Applies to the BASE pass only (node 8); the
# refiner is a fixed 2-step denoise-0.2 polish and is not per-model.
#
# Only settings we have a published source for go in here. In particular do NOT
# blanket-apply a model page's "CFG=1" — the template runs cfg 1.4 on purpose
# (at 1.0 the negative prompt is inert; see the module docstring), and that
# tuning outranks a generic model-card default.
#
# redcraftminimaxh3redmix_30krea2: its Civitai page publishes 8-12 steps. The
# template's base pass is 6 and the refiner runs at denoise 0.2, so it only
# touches the tail of a schedule — effective sampling is 6, under that floor.
# Raise the base pass to the published minimum; sampler/scheduler/cfg stay at
# the template's tuned values.
_KREA_MODEL_SAMPLER: dict[str, dict] = {
    "redcraftminimaxh3redmix_30krea2": {"steps": 8},
}


def _apply_model_sampler(workflow: dict, model: str) -> None:
    """Apply ``_KREA_MODEL_SAMPLER``'s base-pass overrides for ``model``.
    No-op for unlisted models or a missing/!dict base sampler node."""
    over = _KREA_MODEL_SAMPLER.get(_normalize_model(model))
    if not over:
        return
    node = workflow.get(_BASE_SAMPLER_NODE)
    if not isinstance(node, dict):
        return
    inputs = node.setdefault("inputs", {})
    if "steps" in over:
        inputs["steps"] = int(over["steps"])
    for key in ("cfg", "sampler_name", "scheduler"):
        if key in over:
            inputs[key] = over[key]


def _bypass_model_node(wf: dict, node_id: str) -> None:
    """Splice a single-input model node OUT of the chain: every consumer of
    ``[node_id, 0]`` is repointed at whatever fed that node's ``model`` input,
    then the node is deleted.

    Used to drop the baked turbo lora for checkpoints that already carry the
    distillation (``KREA_PREBAKED_TURBO``). No-op unless the node exists AND is
    fed by a ``model`` link to a node that is itself present, so a hand-edited
    template can never be left with a dangling link — worst case the turbo lora
    stays, which is today's behaviour. Only ``model`` inputs are rewired: this
    splices a node out of the MODEL chain and must not touch a same-id link on
    some other socket."""
    node = wf.get(node_id)
    if not isinstance(node, dict):
        return
    upstream = (node.get("inputs") or {}).get("model")
    if not (isinstance(upstream, list) and len(upstream) == 2):
        return
    if str(upstream[0]) not in wf:
        return
    for nid, other in wf.items():
        if nid == node_id or not isinstance(other, dict):
            continue
        inputs = other.get("inputs") or {}
        for key, val in list(inputs.items()):
            if key != "model":
                continue
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == str(node_id) and val[1] == 0:
                inputs[key] = list(upstream)
    del wf[node_id]


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

    Two per-model adjustments ride along: the baked turbo lora is spliced out
    for ``KREA_PREBAKED_TURBO`` checkpoints, and ``_KREA_MODEL_SAMPLER`` may
    override the base pass's sampler settings. Both are no-ops for unlisted
    models, leaving the output identical to the plain template.

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

    # Turbo/lightning checkpoints already carry the distillation — stacking the
    # template's turbo lora on top overcooks them. Drop that one node; the rest
    # of the baked chain (filter bypass, amateur slider) still applies. Runs
    # BEFORE _inject_loras so user loras still splice onto the same tail.
    if _normalize_model(model) in KREA_PREBAKED_TURBO:
        _bypass_model_node(workflow, _TURBO_LORA_NODE)

    # Per-model base-pass sampler settings (steps/cfg/sampler/scheduler).
    _apply_model_sampler(workflow, model)

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
