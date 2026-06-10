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

We substitute the **AnimaStandardFull** template: the AnimaStandard txt2img graph
(EmptyLatent → KSampler → VAEDecode → save-with-metadata) carrying ALL FOUR
ADetailer passes (Hand / NSFW / Face / Eyes) plus the hires-fix upscale, then
REDUCE it to the subset the submission asked for via ``anima_options`` (see
``_apply_anima_options``). With no options the reduction reproduces the old
AnimaStandardDetailer result exactly — hand/face/eyes on, nsfw off, upscale on at
2.0x — so today's behavior is preserved. Produced by un-bypassing those groups in
the official AnimaStandard UI workflow and re-converting via ComfyUI's
graphToPrompt; validated end-to-end on an A10G worker. (The shipped
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
# DUPLICATED in comfytaggenerator/index.html (cbNormalizeModel + ANIMA_MODELS, which
# gate the Anima toggle UI). Keep both in sync when adding a model — the frontend
# can't import this. (Follow-up: expose an is_anima flag on the models endpoint so
# the frontend derives it instead of hardcoding.)
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

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "anima_templates", "AnimaStandardFull.api.json")
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


# ---------------------------------------------------------------------------
# Detailer / upscale reduction (AnimaStandardFull -> the user's selected subset)
# ---------------------------------------------------------------------------
#
# The Full template ships ALL FOUR detailers (hand/nsfw/face/eyes) + the hires-fix
# upscale. A submission's optional ``anima_options`` turns each off; we splice the
# unwanted nodes out of the graph (the same passthrough a ComfyUI bypass does) and
# then garbage-collect anything no longer reachable from the saver. See
# `multi-view-prompt-recipe.md` / the spec for the node-id topology. Node ids below
# are AnimaStandardFull's; never the older Detailer template's.

# Per-detailer node-id triple: (FaceDetailerPipe, EditDetailerPipe, [detector ids]).
# The detector list is the EditDetailerPipe's exclusive feeders (nsfw's one detector
# feeds both bbox_detector + segm_detector slots, so it's still a single node).
_DETAILERS: dict[str, dict[str, Any]] = {
    "hand": {"fdp": "27", "edp": "14", "detectors": ["9"]},
    "nsfw": {"fdp": "28", "edp": "15", "detectors": ["10"]},
    "face": {"fdp": "29", "edp": "16", "detectors": ["11"]},
    "eyes": {"fdp": "30", "edp": "17", "detectors": ["12"]},
}

# Per-key defaults: today's AnimaStandardDetailer behavior — hand/face/eyes on,
# nsfw off, upscale on at 2.0x.
_DETAILER_DEFAULTS: dict[str, bool] = {"hand": True, "nsfw": False, "face": True, "eyes": True}
_UPSCALE_DEFAULT = True
_UPSCALE_FACTOR_DEFAULT = 2.0

# easy hiresFix node + its widget knobs. The net upscale is model_scale(4) *
# percent/100; the only user-facing factor is ``percent`` (= factor * 25).
_UPSCALE_NODE = "60"
_UPSCALE_FACTOR_MIN = 1.0  # percent 25 (the 4x model can't down-scale below this knob)
_UPSCALE_FACTOR_MAX = 4.0  # percent 100 (the 4x model's ceiling — don't exceed it)

# The terminal output node we run reachability from (and the saver invariant).
_SAVER_NODE = "13"


def _find_consumer(wf: dict, src_id: str) -> Optional[tuple[str, str]]:
    """Return ``(consumer_node_id, input_key)`` of the single node reading
    ``[src_id, slot]`` on its ``image``/``images`` chain — i.e. the next node in
    the linear image chain. Returns ``None`` if nothing consumes it.

    We restrict to the image-chain keys (``image``/``images``) so we follow the
    FaceDetailerPipe linked-list, not the detailer-pipe side links."""
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in ("image", "images"):
            val = inputs.get(key)
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == str(src_id):
                return nid, key
    return None


def _remove_detailer(wf: dict, which: str) -> None:
    """Splice one detailer out of the live graph: rewire the consumer of its
    FaceDetailerPipe's output to the FDP's own image source (bypass passthrough),
    then delete the FDP, its EditDetailerPipe, and the detector node(s).

    No-op (and never raises) if the named detailer's nodes aren't present — so it
    composes safely across arbitrary subsets and repeat calls."""
    spec = _DETAILERS.get(which)
    if spec is None:
        return
    fdp_id, edp_id = spec["fdp"], spec["edp"]
    fdp = wf.get(fdp_id)
    if not isinstance(fdp, dict):
        return
    # STEP 1: read the FDP's current image source S.
    src = (fdp.get("inputs") or {}).get("image")
    # STEP 2: rewire the unique downstream image-chain consumer to S.
    consumer = _find_consumer(wf, fdp_id)
    if consumer is not None and isinstance(src, list):
        cid, ckey = consumer
        wf[cid].setdefault("inputs", {})[ckey] = list(src)
    # STEP 3: delete the orphaned detailer-side feeders (EDP + detector(s)).
    for det_id in spec["detectors"]:
        wf.pop(det_id, None)
    wf.pop(edp_id, None)
    # STEP 4: delete the FDP itself.
    wf.pop(fdp_id, None)


def _remove_upscale(wf: dict) -> None:
    """Splice the hires-fix upscale (node 60) out: rewire the saver's ``images``
    to node 60's current image source, then delete node 60. Resolves the source
    dynamically so it works whether or not detailers were already removed."""
    node = wf.get(_UPSCALE_NODE)
    if not isinstance(node, dict):
        return
    src = (node.get("inputs") or {}).get("image")
    consumer = _find_consumer(wf, _UPSCALE_NODE)
    if consumer is not None and isinstance(src, list):
        cid, ckey = consumer
        wf[cid].setdefault("inputs", {})[ckey] = list(src)
    wf.pop(_UPSCALE_NODE, None)


def _set_upscale_factor(wf: dict, factor: float) -> None:
    """Keep node 60 and encode the user's net upscale factor as ``percent`` (=
    factor * 25; the 4x model output is rescaled by that percentage). factor is
    clamped to [1.0, 4.0] so percent stays within the model's 4x ceiling."""
    node = wf.get(_UPSCALE_NODE)
    if not isinstance(node, dict):
        return
    f = max(_UPSCALE_FACTOR_MIN, min(_UPSCALE_FACTOR_MAX, float(factor)))
    node.setdefault("inputs", {})["percent"] = f * 25.0


def _reachable_from(wf: dict, root: str) -> set[str]:
    """Set of node ids reachable from ``root`` by walking every ``[node, slot]``
    input link transitively (the node's input dependency tree)."""
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
    """Drop every node not reachable from ``root`` — garbage-collects the now-
    orphaned base-pipe infra (ToDetailerPipe 19, detector 7, SAMLoader 8) once the
    last detailer that consumed them is gone. Safest way to keep the graph valid."""
    keep = _reachable_from(wf, root)
    for nid in [n for n in wf if n not in keep]:
        del wf[nid]


def _coerce_bool(value: Any, default: bool) -> bool:
    """Treat only real bools as meaningful; anything else falls back to default
    (the substitution must never raise on a malformed option)."""
    return bool(value) if isinstance(value, bool) else default


def _resolve_options(options: Optional[dict]) -> dict[str, Any]:
    """Merge a (possibly partial / malformed) ``anima_options`` over the per-key
    defaults. Never raises; bad values fall back to that key's default.

    Returns ``{"detailers": {hand,nsfw,face,eyes: bool}, "upscale": bool,
    "upscale_factor": float}``."""
    opts = options if isinstance(options, dict) else {}

    raw_detailers = opts.get("detailers")
    if not isinstance(raw_detailers, dict):
        raw_detailers = {}
    detailers = {
        name: _coerce_bool(raw_detailers.get(name), default)
        for name, default in _DETAILER_DEFAULTS.items()
    }

    upscale = _coerce_bool(opts.get("upscale"), _UPSCALE_DEFAULT)

    factor = _UPSCALE_FACTOR_DEFAULT
    raw_factor = opts.get("upscale_factor")
    if isinstance(raw_factor, bool):
        pass  # bool is not a valid factor; keep default
    elif isinstance(raw_factor, (int, float)) and raw_factor > 0:
        factor = float(raw_factor)

    return {"detailers": detailers, "upscale": upscale, "upscale_factor": factor}


# ---------------------------------------------------------------------------
# LoRA stack injection
# ---------------------------------------------------------------------------
#
# ``anima_options.loras`` is an optional list of {"name": "<file>", "strength": f}
# entries. We inject a chain of CORE ``LoraLoader`` nodes between the base loaders
# (UNETLoader 1 + CLIPLoader 18) and the "Lora Loader (LoraManager)" node 5 —
# node 5 stays as an empty passthrough, and core-node semantics avoid any
# dependency on LoraManager's custom cache/name resolution. LoRA files come from
# the catalog's ``lora`` type (mounted at models/loras/), value = the filename.

_LORA_TARGET_NODE = "5"      # the LoraManager passthrough whose model/clip we feed
_LORA_CHAIN_BASE_ID = 9001   # chain node ids 9001.. (template ids are all <= ~102)
_LORA_MAX_COUNT = 8
_LORA_STRENGTH_LIMIT = 4.0   # |strength| clamp; core LoraLoader allows negatives


def _resolve_loras(options: Optional[dict]) -> list[dict]:
    """Sanitized ``[{"name": str, "strength": float}]`` from anima_options.
    Malformed entries are dropped (never raises); the list is capped."""
    opts = options if isinstance(options, dict) else {}
    raw = opts.get("loras")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for entry in raw[: _LORA_MAX_COUNT * 2]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if not name.lower().endswith(_MODEL_EXTS):
            name += ".safetensors"
        strength = entry.get("strength")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            strength = 1.0
        strength = max(-_LORA_STRENGTH_LIMIT, min(_LORA_STRENGTH_LIMIT, float(strength)))
        out.append({"name": name, "strength": strength})
        if len(out) >= _LORA_MAX_COUNT:
            break
    return out


def _inject_loras(wf: dict, loras: list[dict]) -> None:
    """Insert a chain of core LoraLoader nodes feeding ``_LORA_TARGET_NODE``.
    Reads the target's current model/clip links so it composes with whatever
    the template wires there; no-op when the list is empty."""
    target = wf.get(_LORA_TARGET_NODE)
    if not loras or not isinstance(target, dict):
        return
    inputs = target.setdefault("inputs", {})
    model_src, clip_src = inputs.get("model"), inputs.get("clip")
    if not (isinstance(model_src, list) and isinstance(clip_src, list)):
        return
    prev_model, prev_clip = model_src, clip_src
    for i, lora in enumerate(loras):
        nid = str(_LORA_CHAIN_BASE_ID + i)
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
    inputs["model"] = prev_model
    inputs["clip"] = prev_clip


def _saver_reachable(wf: dict) -> bool:
    """Cheap post-reduction invariant: the saver must still exist and its image
    link must resolve to a node still in the graph. A broken saver would otherwise
    ship as a silent 0-output job (the failure mode the project notes warn about),
    so the caller checks this and falls back to the full template if it fails."""
    saver = wf.get(_SAVER_NODE)
    if not isinstance(saver, dict):
        return False
    img = (saver.get("inputs") or {}).get("images")
    return isinstance(img, list) and len(img) == 2 and str(img[0]) in wf


def _apply_anima_options(wf: dict, options: Optional[dict]) -> None:
    """Reduce the Full template to the user's selected detailer/upscale subset, in
    place. Runs AFTER model/prompt/seed injection + WidgetToString neutralization.

    Order: remove unwanted detailers first (each reading live links so chained
    splices compose), then handle upscale against the now-current chain, then a
    reachability sweep from the saver GCs the orphaned base-pipe infra. MAY raise on
    a malformed graph — ``build_anima_workflow`` runs this on a COPY and falls back
    to the full (valid) template if it raises or breaks the saver invariant."""
    resolved = _resolve_options(options)
    for name, keep in resolved["detailers"].items():
        if not keep:
            _remove_detailer(wf, name)
    if resolved["upscale"]:
        _set_upscale_factor(wf, resolved["upscale_factor"])
    else:
        _remove_upscale(wf)
    # GC orphaned base-pipe nodes (19/7/8) and anything else now unreachable from
    # the saver. This also drops any dangling [deleted, slot] link's source.
    _gc_unreachable(wf, _SAVER_NODE)


def build_anima_workflow(
    model: str, positive: str, negative: str, options: Optional[dict] = None
) -> dict:
    """Return the Anima template with the user's model + prompts injected:
    model -> every ``UNETLoader.unet_name``; prompts -> the POSITIVE/NEGATIVE
    wildcard nodes; plus a fresh valid seed.

    The prompt nodes are ALWAYS overwritten (even with an empty string) so the
    template's authored sample prompt can never leak into a user's job, and the
    node ``mode`` is pinned to ``fixed`` so it uses our text verbatim instead of
    re-rolling wildcards from its own widget server-side.

    ``options`` (the submission's ``anima_options``) selects which detailers and
    the upscale survive; absent/partial options fall back to today's behavior
    (hand/face/eyes on, nsfw off, upscale on @2.0x). The reduction runs LAST so
    seed injection + WidgetToString neutralization apply to surviving nodes."""
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

    # Optional LoRA stack (anima_options.loras) — chained core LoraLoader nodes
    # feeding the LoraManager passthrough. Never raises; sanitized + capped. The
    # chain sits upstream of the KSampler so it survives every reduction subset.
    try:
        _inject_loras(workflow, _resolve_loras(options))
    except Exception:  # noqa: BLE001 — a bad lora list must never break submission
        pass

    # Reduce to the requested detailer/upscale subset on a COPY. If the reduction
    # raises or disconnects the saver, keep the full (valid) template rather than
    # ship a structurally-broken job — a safe degradation that still renders.
    try:
        reduced = copy.deepcopy(workflow)
        _apply_anima_options(reduced, options)
        if _saver_reachable(reduced):
            workflow = reduced
    except Exception:  # noqa: BLE001 — keep the full template on any reduction failure
        pass
    return workflow


def maybe_rewrite_to_anima(workflow: dict, options: Optional[dict] = None) -> Optional[dict]:
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
        return build_anima_workflow(model, positive, negative, options)
    except Exception:  # noqa: BLE001 — substitution must never break submission
        return None
