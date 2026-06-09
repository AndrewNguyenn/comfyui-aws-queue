"""
Anima workflow substitution.

The "Anima" model family (circlestone-labs) is a **Qwen-Image architecture**, not
SDXL. An Anima model loads as three separate pieces —
``UNETLoader(<anima>.safetensors)`` + ``CLIPLoader(qwen_3_06b_base.safetensors)``
+ ``VAELoader(qwen_image_vae.safetensors)``. A normal SDXL/checkpoint workflow
that loads an Anima model through ``CheckpointLoaderSimple`` gets a ``None`` CLIP
and dies with ``RuntimeError: ERROR: clip input is invalid: None`` — and because
the worker only polls ``/history`` for completion, that error is masked as a full
840 s timeout instead of failing fast.

So: when a submitted prompt references an Anima model, we **discard the submitted
graph** and run the official Anima workflow instead, carrying over only
(a) the user's chosen Anima model and (b) their positive/negative prompt text.

We substitute the **AnimaStandardDetailer** template: the AnimaStandard txt2img
graph (EmptyLatent → KSampler → VAEDecode → save-with-metadata) with the Hand /
Face / Eyes ADetailer passes and the hires-fix 2x upscale enabled (the stylistic
post-FX, NSFW detailer, and img2img path stay bypassed). Produced by un-bypassing
those groups in the official AnimaStandard UI workflow and re-converting via
ComfyUI's graphToPrompt; validated end-to-end on an A10G worker. (The shipped
AnimaDetailerV6 file is NOT used directly — its base sampler is bypassed, so it
can't generate from a prompt.)

Detection is an **explicit allowlist** (``ANIMA_MODELS``) — deliberately NOT a
name heuristic: "anime"-named SDXL checkpoints (e.g. ``novaAnimeXL_ilV190``,
Illustrious) are ordinary checkpoints that must NOT be rewritten.

The substitution never raises: any failure leaves the original workflow intact.
"""
from __future__ import annotations

import copy
import json
import os
import random
from typing import Any, Optional

# rgthree's Seed node validates seed <= 2**50; stay under it. The frontend
# normally resolves a Seed node's -1 to a random value before submit — we bypass
# the frontend, so we inject the seed ourselves.
_SEED_MAX = 2 ** 50

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Explicit allowlist of Anima (Qwen-arch) model *filenames*, normalized (basename,
# extension stripped, lowercased). Add new Anima models here as they are added to
# the catalog. NOTE: an Anima model must also be cataloged as ``diffusion_models``
# (and live under the diffusion_models/ S3 prefix) so the template's UNETLoader
# can find it — see project notes.
ANIMA_MODELS: frozenset[str] = frozenset(
    {
        "anima_basev10",
        "anima_preview3base",
        "animacattower_v10",
        "copycatanima_20260519",
        "miaomiaoharem_anima11",
        "terrarising_20terrarisinganima",
    }
)

# Node input keys that carry a model filename on a loader-style node. Covers the
# plain core loaders plus provider-wrapped ones (e.g. Image-Saver's "Checkpoint
# Loader with Name", whose input is still ``ckpt_name``).
_MODEL_INPUT_KEYS = ("ckpt_name", "unet_name", "model_name")

_MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".sft", ".gguf", ".bin")


def _normalize_model(name: str) -> str:
    base = os.path.basename(str(name)).strip()
    low = base.lower()
    for ext in _MODEL_EXTS:
        if low.endswith(ext):
            return low[: -len(ext)]
    return low


def detect_anima_model(workflow: dict) -> Optional[str]:
    """Return the original model filename if ``workflow`` references an Anima
    model, else ``None``. Scans every node's loader-style inputs."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in _MODEL_INPUT_KEYS:
            val = inputs.get(key)
            if isinstance(val, str) and _normalize_model(val) in ANIMA_MODELS:
                return val
    return None


# ---------------------------------------------------------------------------
# Prompt extraction (from the submitted workflow)
# ---------------------------------------------------------------------------

# Widget keys that hold literal prompt text on a text-bearing node, in priority
# order (String Literal -> .string, CLIPTextEncode -> .text, wildcard -> ...).
_TEXT_KEYS = ("string", "text", "populated_text", "wildcard_text", "value", "text_g")

# How deep to chase link -> node -> link chains while resolving text.
_RESOLVE_MAX_DEPTH = 8


def _resolve_text(workflow: dict, value: Any, depth: int = 0, seen: Optional[set] = None) -> str:
    """Resolve an input value — a literal string OR an ``[node_id, slot]`` link —
    to the prompt text behind it. Real graphs wire CLIPTextEncode.text from a
    ``String Literal`` provider node, so we follow links transitively."""
    if isinstance(value, str):
        return value
    if depth >= _RESOLVE_MAX_DEPTH:
        return ""
    if isinstance(value, list) and len(value) == 2:
        nid = str(value[0])
        seen = seen if seen is not None else set()
        if nid in seen:
            return ""
        seen.add(nid)
        node = workflow.get(nid)
        if not isinstance(node, dict):
            return ""
        inputs = node.get("inputs") or {}
        for key in _TEXT_KEYS:
            if key in inputs:
                text = _resolve_text(workflow, inputs[key], depth + 1, seen)
                if text.strip():
                    return text
    return ""


def _longest(values: list[str]) -> str:
    return max(values, key=len) if values else ""


def extract_prompts(workflow: dict) -> tuple[str, str]:
    """Best-effort ``(positive, negative)`` prompt text from a submitted workflow.

    Primary: trace every sampler's ``positive`` / ``negative`` conditioning back
    to the text feeding it. Fallback: scan text nodes by their title. For each
    role we keep the longest non-empty candidate (the real prompt, not an empty
    or detailer sub-prompt)."""
    positive: list[str] = []
    negative: list[str] = []

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if "sampler" not in str(node.get("class_type", "")).lower():
            continue
        inputs = node.get("inputs") or {}
        p = _resolve_text(workflow, inputs.get("positive"))
        n = _resolve_text(workflow, inputs.get("negative"))
        if p.strip():
            positive.append(p)
        if n.strip():
            negative.append(n)

    if not positive or not negative:
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            title = str((node.get("_meta") or {}).get("title") or "").lower()
            if "positive" not in title and "negative" not in title:
                continue
            inputs = node.get("inputs") or {}
            text = next((inputs[k] for k in _TEXT_KEYS if isinstance(inputs.get(k), str) and inputs[k].strip()), "")
            if not text:
                continue
            if "positive" in title:
                positive.append(text)
            elif "negative" in title:
                negative.append(text)

    return _longest(positive), _longest(negative)


# ---------------------------------------------------------------------------
# Template + injection
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "anima_templates", "AnimaStandardDetailer.api.json")
_template_cache: Optional[dict] = None

# Titles of the ImpactWildcardProcessor nodes that hold the prompt text in the
# Anima templates (set by the workflow author).
_POSITIVE_TITLE = "POSITIVE"
_NEGATIVE_TITLE = "NEGATIVE"


def _load_template() -> dict:
    """Load + cache the API-format Anima template (deep-copied per call so the
    caller can mutate freely)."""
    global _template_cache
    if _template_cache is None:
        with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
            _template_cache = json.load(fh)
    return copy.deepcopy(_template_cache)


def _node_by_title(workflow: dict, title: str) -> Optional[dict]:
    for node in workflow.values():
        if isinstance(node, dict) and (node.get("_meta") or {}).get("title") == title:
            return node
    return None


def _inject_seed(workflow: dict, seed: int) -> None:
    """Set a concrete seed wherever a node carries a *literal* seed widget
    (``seed``/``noise_seed`` as an int, not a ``[node, slot]`` link). The Anima
    template feeds one rgthree Seed node (shipped as -1) into the sampler + the
    wildcard nodes; without this every Anima job would submit a raw -1."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), int):
                inputs[key] = seed


def _neutralize_widget_to_string(workflow: dict, model: str) -> None:
    """KJNodes' ``WidgetToString`` reads ``extra_pnginfo["workflow"]`` to fetch a
    widget value off the UI graph — but the worker submits API-format prompts with
    NO extra_pnginfo, so the node crashes ('NoneType' is not subscriptable) before
    sampling. In the Anima template it only fetches the UNETLoader's ``unet_name``
    for the saved metadata, so resolve its output to the literal model name (or ''
    for any other widget) and drop the node."""
    for nid, node in list(workflow.items()):
        if not isinstance(node, dict) or node.get("class_type") != "WidgetToString":
            continue
        value = model if (node.get("inputs") or {}).get("widget_name") == "unet_name" else ""
        for other in workflow.values():
            inputs = other.get("inputs") if isinstance(other, dict) else None
            if not inputs:
                continue
            for key, val in list(inputs.items()):
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == str(nid):
                    inputs[key] = value
        del workflow[nid]


def build_anima_workflow(model: str, positive: str, negative: str) -> dict:
    """Return the Anima template with the user's model + prompts injected:
    model -> every ``UNETLoader.unet_name``; prompts -> the POSITIVE/NEGATIVE
    wildcard nodes; plus a fresh valid seed.

    The prompt nodes are ALWAYS overwritten (even with an empty string) so the
    template's authored sample prompt can never leak into a user's job, and the
    node ``mode`` is pinned to ``fixed`` so it uses our text verbatim instead of
    re-rolling wildcards from its own widget server-side."""
    workflow = _load_template()

    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "UNETLoader":
            node.setdefault("inputs", {})["unet_name"] = model

    for title, text in ((_POSITIVE_TITLE, positive), (_NEGATIVE_TITLE, negative)):
        node = _node_by_title(workflow, title)
        if node is not None:
            inputs = node.setdefault("inputs", {})
            inputs["wildcard_text"] = text
            inputs["populated_text"] = text
            inputs["mode"] = "fixed"

    _inject_seed(workflow, random.randint(0, _SEED_MAX - 1))
    _neutralize_widget_to_string(workflow, model)
    return workflow


def maybe_rewrite_to_anima(workflow: dict) -> Optional[dict]:
    """If ``workflow`` references an Anima model, return the Anima workflow
    carrying over the model + prompts; else ``None`` (caller keeps the original).
    Never raises — on any failure returns ``None``.

    Note: when an Anima model is detected we always substitute, even if prompt
    extraction comes back empty. Falling back to the original is not safer — the
    original is the exact graph that fails with a None-CLIP error (masked as an
    840s timeout); substituting at least runs the correct architecture, and the
    authored sample prompt is overwritten so it can't masquerade as the user's."""
    try:
        model = detect_anima_model(workflow)
        if not model:
            return None
        positive, negative = extract_prompts(workflow)
        return build_anima_workflow(model, positive, negative)
    except Exception:  # noqa: BLE001 — substitution must never break submission
        return None
