"""
MiniMax H3 reference-to-video+audio.

MiniMax H3 (https://www.minimax.io/blog/minimax-h3) is an omni-modal generator:
one pass produces video AND audio, from a prompt plus reference images. It is
native to ComfyUI from v0.30.0 (``comfy_extras/nodes_minimax_h3.py``) — no
custom node pack — and is a different architecture from everything else we run,
which is why it gets its own family rather than a mode on ``scail.py``.

The graph is the official ``video_minimax_h3_r2v`` template with two
substitutions: ``int8_convrot`` weights for both the transformer and the
Qwen3-VL-32B text encoder, because the template ships an ``nvfp4_awq`` CLIP and
NVFP4 is a Blackwell format that cannot execute on this fleet's A10G (sm_86) or
L40S (sm_89).

Three invariants differ from every other family here and are enforced below,
because each one silently produces garbage rather than an error:

  1. **Frame counts live on a 17k+5 grid at 24 fps** — not Wan's 4n+1 at 16 fps.
     The node snaps internally, but snapping here keeps the reported duration
     honest instead of quietly lengthening the clip.
  2. **The canvas is area-capped**, not just rounded: 768*1344 pixels max, each
     axis a multiple of 32.
  3. **Reference slots are 0-indexed while the prompt's labels are 1-indexed.**
     ``ref_image_0`` is what the prompt calls ``<Picture 1>``. Getting this
     backwards misattributes every reference in a multi-shot script.

Note on the reference inputs: ``ref_images`` is an Autogrow input. Its slot
names (``ref_image_0`` .. ``ref_image_8``) are what the SCHEMA declares, for the
editor and for validation — but ``execute()`` takes the group as a single
``ref_images`` dict, so that is what an API-format prompt must send:

    "ref_images": {"ref_image_0": ["10", 0]}

Sending the slots flat on the node instead produces
``execute() got an unexpected keyword argument 'ref_image_0'`` — ComfyUI does
not regroup them.

The build never raises: any failure returns None and leaves the caller's
workflow intact.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from typing import Optional

from krea import KREA_MODELS, KREA_PREBAKED_TURBO, _PREBAKED_SLIDER_STRENGTH
from zimage import ZIMAGE_MODELS, _normalize_model

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "minimax_templates")

# MiniMax H3 ships as two separately-trained checkpoints, and the difference is
# what the supplied image MEANS:
#
#   ref  (Ref2VA) — MiniMaxH3ReferenceToVideo. The image is a REFERENCE whose
#                   latent rides alongside every sampling step. The clip does
#                   not start from it; the model composes a fresh scene that the
#                   reference only nudges. Free to re-frame and re-light across
#                   shots, but weak at holding a specific face — measured: an
#                   East Asian reference produced a Western-featured subject on
#                   two runs, at both ref_image_size settings.
#
#   fl2v (FL2VA)  — MiniMaxH3ImageToVideo. The image becomes frame 0 outright
#                   (the node files it as resolved_frame_index 0), so identity,
#                   wardrobe and room start correct by construction and the clip
#                   animates forward from there. The trade is that everything
#                   downstream inherits that opening frame.
#
# Different checkpoints, so switching modes changes the UNet as well as the
# conditioning node.
_TEMPLATES = {
    "ref": "MiniMaxH3Ref2VA.api.json",
    "fl2v": "MiniMaxH3FL2VA.api.json",
}
_DEFAULT_MODE = "ref"

_SEED_MAX = 2 ** 50

FPS = 24
# From the node's own tooltip: 124 frames (~5.2 s) is the default and the
# trained range is ~124-362 (~5.2-15.1 s). We allow a little beyond the top of
# that band but refuse to pretend anything past it is supported.
_MIN_FRAMES = 124
_MAX_FRAMES = 362

# comfy_extras/nodes_minimax_h3.py: BASE_SHORT_EDGE 768, MAX_PIXELS 768*1344,
# CANVAS_MULTIPLE 32.
_CANVAS_MULTIPLE = 32
_MAX_PIXELS = 768 * 1344
_MIN_DIM = 32

_N_UNET = "1"
_N_CLIP = "2"
_N_REF_LOAD = "10"
_N_REF2V = "20"
_REF_GROUP = "ref_images"        # Autogrow group; slots ref_image_0..8 live inside it (ref mode)
_N_REF_RESIZE = "11"             # ImageScale cover-crop ahead of first_frame (fl2v mode)
_N_NOISE = "30"
_N_SCHED = "32"
_N_GUIDER = "33"
_N_VIDEO = "50"

# Step-distilled LoRA (lightx2v). Two things make this the 8-step and not the
# advertised 4-step build:
#
#   * Community testing converged on 8 — "the number in the name is not the
#     number most people should use". 6-8 steps largely removes the motion
#     smear 4 steps introduces.
#   * It is distilled at 12/3 video/audio sigma shifts, which is exactly
#     ComfyUI's MiniMaxH3SigmaShift default. This template has no shift node,
#     so it inherits those and the LoRA drops in unchanged. The 4-step 768p
#     build wants 6/3 instead — using it here would be a silent mismatch, not
#     an error, because nothing validates shifts against the weights.
#
# Distillation is documented to hurt "the quietest and most sustained vocal
# registers", and several of our sets are built on whispered close-mic lines,
# so this is opt-in per job rather than always-on.
# The sex LoRA. Trained on MiniMax H3 for missionary, doggy, cowgirl, handjob,
# blowjob and insertions; the author's own note is "use it at strength 0.5 or
# below", so that is the default and 1.0 is the hard cap. It is spliced the same
# way as the turbo LoRA and composes with it.
_NSFW_LORA = "HMNSFW_AIO_V2.safetensors"
_NSFW_STRENGTH = 0.5
_N_NSFW_LORA = "9002"

# What counts as a clip this LoRA is for. Every word here is one our own writer
# actually produces — it is told to name anatomy directly, so "cock" and "cum"
# are reliable and coy substitutes never appear. Deliberately NOT "facial":
# "extreme facial close-up" is a framing in the house camera vocabulary and
# would light this up on clips with nothing in them.
_SEX_RE = re.compile(
    r"\b(cock|cocks|blowjob|handjob|deepthroat|missionary|doggy|cowgirl|"
    r"fuck|fucks|fucked|fucking|penetrate[sd]?|penetration|penetrating|"
    r"insertion|insertions|cum|cumming|creampie|riding him|"
    # Insertion is one of the six acts it was trained on, and the solo version
    # of it is the one the act words above miss: a clip can be a toy or her own
    # fingers with no partner in it anywhere.
    r"dildo|vibrator|fingering)\b"
    r"|\bfingers?\s+(?:\w+\s+){0,2}(?:into|inside)\s+her\b"
    r"|\b(?:into|inside)\s+her\s+(?:pussy|arse|ass)\b",
    re.IGNORECASE,
)

_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
_TURBO_STRENGTH = 0.7   # the publisher's own recommendation
_TURBO_STEPS = 8
_N_TURBO_LORA = "9001"


def snap_frames(seconds: float) -> int:
    """Seconds -> a frame count on MiniMax's 17k+5 grid at 24 fps.

    This is the node's own arithmetic (the official template computes it with a
    math expression node): round to frames, then walk UP to the next value
    satisfying ``n % 17 == 5``. 5 s -> 120 -> 124.
    """
    n = max(5, round(float(seconds) * FPS))
    n += (5 - (n % 17)) % 17
    return n


def clamp_frames(n: int) -> int:
    """Clamp to the trained band, staying on the grid."""
    n = max(_MIN_FRAMES, min(_MAX_FRAMES, int(n)))
    n += (5 - (n % 17)) % 17
    if n > _MAX_FRAMES:          # the walk-up may overshoot the ceiling
        n -= 17
    return n


def snap_canvas(width: int, height: int) -> tuple[int, int]:
    """Round each axis to 32 and scale down to the model's pixel cap.

    The cap is on AREA, so a legal-looking 1920x1080 (both /32) is still out of
    range; it has to be scaled, not just rounded.
    """
    w = max(_MIN_DIM, int(width))
    h = max(_MIN_DIM, int(height))
    if w * h > _MAX_PIXELS:
        s = math.sqrt(_MAX_PIXELS / (w * h))
        w, h = w * s, h * s
    rw = max(_CANVAS_MULTIPLE, int(round(w / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)
    rh = max(_CANVAS_MULTIPLE, int(round(h / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)
    # Rounding up can push a borderline canvas back over the cap.
    while rw * rh > _MAX_PIXELS:
        if rw >= rh:
            rw -= _CANVAS_MULTIPLE
        else:
            rh -= _CANVAS_MULTIPLE
    return rw, rh


def _load_template(mode: str) -> Optional[dict]:
    try:
        with open(os.path.join(_TEMPLATE_DIR, _TEMPLATES[mode]), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        print(f"minimax: template load failed for mode {mode!r}: {e!r}")
        return None


def maybe_build_minimax(
    options: Optional[dict],
    ref_image_name: Optional[str] = None,
) -> Optional[dict]:
    """Build a MiniMax H3 ref2va workflow, or None if this isn't one.

    ``options`` is the request's ``minimax_options``. An image is normally
    required — it is either the reference or frame 0, and the prompt's
    ``<Picture 1>`` has to resolve to something. The one exception is
    ``options["autoframe"]``: a request that carries no upload but does ask for
    an auto first frame gets a Z-Image still generated inside the same graph
    (see ``splice_autoframe``), so ``<Picture 1>`` still resolves.
    """
    if not isinstance(options, dict) or not options:
        return None

    mode = str(options.get("mode") or _DEFAULT_MODE).lower()
    if mode not in _TEMPLATES:
        print(f"minimax: unknown mode {mode!r}; not building")
        return None
    # Both modes need an image — it is either the reference or frame 0. Without
    # an upload we can still build, but only if the caller opted into having one
    # generated; silently falling through to text-to-video would quietly discard
    # the character the prompt was composed around.
    autoframe = options.get("autoframe")
    autoframe = autoframe if isinstance(autoframe, dict) else None
    if not ref_image_name and not autoframe:
        print("minimax: missing reference image; not building")
        return None

    wf = _load_template(mode)
    if wf is None:
        return None

    try:
        wf = copy.deepcopy(wf)

        if options.get("frames") is not None:
            frames = clamp_frames(options["frames"])
        else:
            frames = clamp_frames(snap_frames(options.get("seconds", 5)))

        width, height = snap_canvas(
            options.get("width", 1344), options.get("height", 768)
        )

        if ref_image_name:
            wf[_N_REF_LOAD]["inputs"]["image"] = ref_image_name

        r2v = wf[_N_REF2V]["inputs"]
        r2v["prompt"] = str(options.get("prompt", "") or "")
        r2v["width"] = width
        r2v["height"] = height
        r2v["length"] = frames
        # FL2VA cover-crops the first frame to the canvas before the node's own
        # plain stretch; that resize must track the canvas or frame 0 distorts.
        if mode == "fl2v" and ref_image_name:
            wf[_N_REF_RESIZE]["inputs"]["width"] = width
            wf[_N_REF_RESIZE]["inputs"]["height"] = height
        # ref_image_size exists only on the Ref2VA node — FL2VA cover-crops to
        # the canvas via the ImageScale ahead of it instead.
        if mode == "ref" and options.get("ref_image_size") in ("match", "max"):
            r2v["ref_image_size"] = options["ref_image_size"]

        seed = options.get("seed")
        seed = int(seed) % _SEED_MAX if seed is not None else random.randint(0, _SEED_MAX - 1)
        wf[_N_NOISE]["inputs"]["noise_seed"] = seed

        # The sex LoRA first, so that with turbo on the chain is
        # UNet -> nsfw -> turbo -> sampler and the distilled schedule stays
        # nearest the consumers.
        nsfw = nsfw_lora_strength(options)
        if nsfw:
            splice_nsfw_lora(wf, nsfw)

        # Turbo before the explicit steps override, so a caller can still pin a
        # step count on top of it (e.g. to A/B 8 vs 6 on the same LoRA).
        if options.get("turbo"):
            splice_turbo_lora(wf)

        if options.get("steps") is not None:
            wf[_N_SCHED]["inputs"]["steps"] = max(1, min(60, int(options["steps"])))

        # No upload: render the opening frame in-graph. Done here, after the
        # canvas is snapped, so the still is generated at exactly the video's
        # dimensions and nothing has to be cropped into place afterwards.
        if not ref_image_name:
            splice_autoframe(wf, mode, autoframe, width, height, seed)

        # Keep the container's frame rate tied to the model's: MiniMax H3 is a
        # 24 fps model, and muxing its frames at any other rate changes the
        # playback speed rather than the frame count.
        wf[_N_VIDEO]["inputs"]["fps"] = float(FPS)

        return wf
    except Exception as e:  # noqa: BLE001
        print(f"minimax: build failed, leaving workflow untouched: {e!r}")
        return None


# ---------------------------------------------------------------------------
# Auto first frame
# ---------------------------------------------------------------------------
# Both MiniMax nodes take their image as an OPTIONAL input, so a clip with no
# upload is legal — it just becomes text-to-video, and the model invents a face
# from scratch. That is worse than it sounds here: this app already has a
# curated character pool, a natural-prompt library and per-family house LoRA
# stacks, and none of it reaches a T2V clip.
#
# So instead of dropping the image, we GENERATE it: a small text-to-image graph
# is spliced into the same workflow, and its decode feeds ``first_frame``. One
# job, one queue, one GPU — ComfyUI unloads the still model before MiniMax's
# 20 GiB transformer loads, so the peak is unchanged and the cost is a handful
# of seconds against a multi-minute sample.
#
# Two families are supported, because the image fleet runs two architectures
# and they share nothing: different text encoder, different CLIP type, different
# VAE, different sampler settings, different baked LoRA stack. The family is
# derived from the model rather than passed in, so the caller cannot pair a
# Krea checkpoint with Z-Image's text encoder.
#
# Krea samples with RES4LYF's ClownsharKSampler_Beta, the same node the image
# fleet uses, so frame 0 matches the gallery by construction. This started out
# on core res_multistep/deis instead, because RES4LYF was not baked into the
# video worker image — and standing in for res_2s/deis_3m silently dropped
# `eta 0.5` (ancestral noise re-injected each step) and `bongmath`. The missing
# stochasticity read directly as waxy, overbaked skin, which is the failure
# this whole recipe exists to avoid. RES4LYF is now baked (workers/video/
# baked_nodes.txt) and the settings below are lifted from
# krea_templates/Krea2Simple.api.json verbatim rather than approximated.
#
# Z-Image stays on core nodes: its graph never used RES4LYF.
_AUTOFRAME_FAMILIES = {
    # Krea 2 — Qwen-Image-like. Note the VAE: Krea 2 decodes through the Wan 2.1
    # VAE, which this fleet already holds for SCAIL, so it costs no new weights.
    "krea": {
        "clip": "qwen3vl_4b_bf16.safetensors",
        "clip_type": "krea2",
        "clip_layer": None,
        "vae": "wan21_vae_fp32.safetensors",
        "steps": 8,          # _KREA_MODEL_SAMPLER's redcraft override
        "cfg": 1.4,          # not the published 1.0: at 1.0 the negative is inert
        "sampler": "exponential/res_2s",
        "scheduler": "beta",
        "sampler_class": "ClownsharKSampler_Beta",
        "eta": 0.5,
        # The image fleet's second pass: 2 steps at 0.2 denoise, a polish rather
        # than a generation. Cheap, and the look is calibrated with it present.
        "refine": {"steps": 2, "denoise": 0.2,
                   "sampler": "multistep/deis_3m", "scheduler": "bong_tangent"},
        "lora_class": "LoraLoaderModelOnly",
        # Chain order matters and mirrors krea_templates/Krea2Simple.api.json.
        # `role` is what makes an entry conditional; see KREA_PREBAKED_TURBO.
        "baked": (
            {"name": "krea2_turbo_lora_rank_64_bf16.safetensors", "strength": 0.6, "role": "turbo"},
            {"name": "krea2filterbypass.safetensors", "strength": 1.0, "role": None},
            {"name": "AmateurSlider-KREA2_v1.safetensors", "strength": 1.5, "role": "slider"},
        ),
        "trigger": "",
    },
    # Z-Image — Lumina2-like, and the only family whose house LoRA has a CLIP
    # half, hence the full LoraLoader rather than the model-only variant.
    "zimage": {
        "clip": "qwen_3_4b_fp8_mixed.safetensors",
        "clip_type": "lumina2",
        "clip_layer": -2,
        "vae": "ae_zimgturbo.safetensors",
        "steps": 11,
        "cfg": 1.5,
        "sampler": "euler",
        "scheduler": "simple",
        "sampler_class": "KSampler",
        "refine": None,
        "lora_class": "LoraLoader",
        "baked": ({"name": "zimage-igbaddie_pruned.safetensors", "strength": 0.6, "role": None},),
        "trigger": "igbaddie",
    },
}
# RedCraft RedMix 3.0 (Krea 2), the checkpoint the image side is currently on.
# It is in KREA_PREBAKED_TURBO, so its turbo pass is dropped and the amateur
# slider drops with it — see krea.py for why those two are one fact.
_AUTOFRAME_DEFAULT_MODEL = "redcraftMinimaxH3REDMIX_30Krea2.safetensors"
# Short on purpose: both families sample at cfg <= 1.5, where a long negative
# costs more than it corrects.
_AUTOFRAME_NEGATIVE = (
    "blurry, low quality, worst quality, watermark, text, signature, "
    "deformed hands, extra fingers, extra limbs, mutated"
)

# 9100+ so these never collide with the templates' 1..51.
_AF_UNET = "9101"
_AF_CLIP = "9102"
_AF_CLIPSET = "9103"
_AF_VAE = "9104"
_AF_POS = "9105"
_AF_NEG = "9106"
_AF_LATENT = "9107"
_AF_KSAMPLER = "9108"
_AF_DECODE = "9109"
_AF_REFINE = "9110"
_AF_BAKED_BASE = 9111
_AF_LORA_BASE = 9120
_AF_MAX_LORAS = 8


def autoframe_family(model: str) -> Optional[str]:
    """Which text-to-image recipe a checkpoint belongs to, or None if neither.

    Doubles as the allowlist: this name goes straight into a UNETLoader and the
    whole model catalog is mounted on this fleet, so anything not recognised as
    one of our two image families is refused rather than loaded.
    """
    n = _normalize_model(model or "")
    if n in KREA_MODELS:
        return "krea"
    if n in ZIMAGE_MODELS:
        return "zimage"
    return None


def _autoframe_loras(options: dict) -> list:
    """Validated ``[{name, strength}]``, capped like every other family's."""
    out = []
    for entry in options.get("loras") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        # Same containment rule as the reference-image name: a LoRA filename is
        # a leaf under models/loras, never a path.
        if not name or "/" in name or "\\" in name or ".." in name:
            continue
        try:
            strength = float(entry.get("strength", 1.0))
        except (TypeError, ValueError):
            continue
        if not -4.0 <= strength <= 4.0:
            continue
        out.append({"name": name, "strength": strength})
        if len(out) >= _AF_MAX_LORAS:
            break
    return out


def _baked_stack(fam: dict, model: str) -> list:
    """The family's house LoRAs for this checkpoint.

    A turbo/lightning checkpoint already has that distillation merged in, so
    applying the turbo LoRA again overcooks it — and the amateur slider's 1.5 is
    calibrated to cancel plasticity that is then no longer being introduced,
    landing as grain instead. Both adjustments come from the same fact, which is
    why krea.py keeps them welded and this reads them from there.
    """
    prebaked = _normalize_model(model) in KREA_PREBAKED_TURBO
    out = []
    for entry in fam["baked"]:
        if prebaked and entry["role"] == "turbo":
            continue
        strength = entry["strength"]
        if prebaked and entry["role"] == "slider":
            strength = _PREBAKED_SLIDER_STRENGTH
        out.append({"name": entry["name"], "strength": strength})
    return out


def splice_autoframe(wf: dict, mode: str, options: dict,
                     width: int, height: int, seed: int) -> None:
    """Add a text-to-image subgraph and hand its image to the MiniMax node.

    Replaces the template's LoadImage (and, in fl2v, the ImageScale that
    cover-crops for it): the still is rendered AT the video canvas, so there is
    nothing left to crop.
    """
    model = options.get("model")
    family = autoframe_family(model) if isinstance(model, str) else None
    if model and family is None:
        print(f"minimax: autoframe model {model!r} is not a known image "
              f"checkpoint; using {_AUTOFRAME_DEFAULT_MODEL}")
    if family is None:
        model = _AUTOFRAME_DEFAULT_MODEL
        family = autoframe_family(model)
    model = model.strip()
    fam = _AUTOFRAME_FAMILIES[family]

    steps = fam["steps"]
    cfg = fam["cfg"]
    if options.get("steps") is not None:
        steps = max(1, min(50, int(options["steps"])))
    if options.get("cfg") is not None:
        cfg = max(0.0, min(20.0, float(options["cfg"])))

    positive = str(options.get("prompt", "") or "").strip()
    # The trigger token has to appear in the prompt or the house LoRA is weights
    # loaded for nothing. Owned here rather than by the caller, so a prompt
    # written for one family never arrives carrying the other's token.
    trigger = fam["trigger"]
    if trigger and trigger.lower() not in re.split(r"[\s,]+", positive.lower()):
        positive = f"{trigger}, {positive}" if positive else trigger
    negative = str(options.get("negative", "") or "").strip() or _AUTOFRAME_NEGATIVE

    wf[_AF_UNET] = {"class_type": "UNETLoader",
                    "inputs": {"unet_name": model, "weight_dtype": "default"}}
    wf[_AF_CLIP] = {"class_type": "CLIPLoader",
                    "inputs": {"clip_name": fam["clip"], "type": fam["clip_type"],
                               "device": "default"}}
    clip_src = [_AF_CLIP, 0]
    if fam["clip_layer"] is not None:
        wf[_AF_CLIPSET] = {"class_type": "CLIPSetLastLayer",
                           "inputs": {"stop_at_clip_layer": int(fam["clip_layer"]),
                                      "clip": [_AF_CLIP, 0]}}
        clip_src = [_AF_CLIPSET, 0]
    wf[_AF_VAE] = {"class_type": "VAELoader", "inputs": {"vae_name": fam["vae"]}}

    model_src = [_AF_UNET, 0]
    lora_class = fam["lora_class"]
    for i, lora in enumerate(_baked_stack(fam, model)):
        nid = str(_AF_BAKED_BASE + i)
        inputs = {"lora_name": lora["name"], "strength_model": lora["strength"],
                  "model": model_src}
        if lora_class == "LoraLoader":
            inputs["strength_clip"] = lora["strength"]
            inputs["clip"] = clip_src
        wf[nid] = {"class_type": lora_class, "inputs": inputs}
        model_src = [nid, 0]
        if lora_class == "LoraLoader":
            clip_src = [nid, 1]

    # Caller LoRAs layer ON TOP of the house stack, and always ride the CLIP too
    # — a model-only chain silently drops the half of a character LoRA that
    # lives in the text encoder, so the trained token then means nothing.
    for i, lora in enumerate(_autoframe_loras(options)):
        nid = str(_AF_LORA_BASE + i)
        wf[nid] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": lora["name"],
            "strength_model": lora["strength"],
            "strength_clip": lora["strength"],
            "model": model_src, "clip": clip_src,
        }}
        model_src, clip_src = [nid, 0], [nid, 1]

    wf[_AF_POS] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": positive, "clip": clip_src}}
    wf[_AF_NEG] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": negative, "clip": clip_src}}
    wf[_AF_LATENT] = {"class_type": "EmptySD3LatentImage",
                      "inputs": {"width": width, "height": height, "batch_size": 1}}
    def _sampler(nid, sampler_name, scheduler, n_steps, denoise, latent):
        """One node, whichever class the family samples with.

        ClownsharKSampler_Beta's extra widgets are not decoration: `eta` is the
        ancestral noise that keeps skin from going waxy, and `bongmath` is its
        numerical correction. Dropping either is what made frame 0 look baked.
        """
        inputs = {
            "seed": seed, "steps": n_steps, "cfg": cfg,
            "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise,
            "model": model_src, "positive": [_AF_POS, 0], "negative": [_AF_NEG, 0],
            "latent_image": latent,
        }
        if fam["sampler_class"] == "ClownsharKSampler_Beta":
            inputs.update({"eta": fam["eta"], "steps_to_run": -1,
                           "sampler_mode": "standard", "bongmath": True})
        wf[nid] = {"class_type": fam["sampler_class"], "inputs": inputs}
        return [nid, 0]

    sampled = _sampler(_AF_KSAMPLER, fam["sampler"], fam["scheduler"],
                       steps, 1.0, [_AF_LATENT, 0])
    refine = fam["refine"]
    if refine:
        sampled = _sampler(_AF_REFINE, refine["sampler"], refine["scheduler"],
                           refine["steps"], refine["denoise"], sampled)
    wf[_AF_DECODE] = {"class_type": "VAEDecode",
                      "inputs": {"samples": sampled, "vae": [_AF_VAE, 0]}}

    # Hand the generated still to the MiniMax node the same way an upload
    # would arrive, then drop the now-unreachable loader nodes.
    if mode == "fl2v":
        wf[_N_REF2V]["inputs"]["first_frame"] = [_AF_DECODE, 0]
        wf.pop(_N_REF_RESIZE, None)
    else:
        wf[_N_REF2V]["inputs"][_REF_GROUP] = {"ref_image_0": [_AF_DECODE, 0]}
    wf.pop(_N_REF_LOAD, None)


def _splice_model_lora(wf: dict, nid: str, lora_name: str, strength: float) -> None:
    """Insert a model-only LoRA between the model and everything reading it.

    Both consumers have to be rewired, not just the sampler: BasicScheduler
    derives the sigma schedule from the model, and a distilled model's schedule
    is the whole point. Leaving node 32 on the raw UNet would run 8 steps of an
    undistilled sigma curve — fast and wrong, with no error to show for it.

    Reads whatever the guider currently points at rather than assuming the raw
    UNet, so two of these compose: spliced one after another they chain, where
    a hardcoded [_N_UNET, 0] would leave the second one loaded and connected to
    nothing.
    """
    guider = wf.get(_N_GUIDER)
    if not isinstance(guider, dict):
        return
    src = guider.get("inputs", {}).get("model")
    if not src:
        return
    wf[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "lora_name": lora_name,
        "strength_model": strength,
        "model": src,
    }}
    for consumer in (_N_SCHED, _N_GUIDER):
        node = wf.get(consumer)
        if isinstance(node, dict) and node.get("inputs", {}).get("model") == src:
            node["inputs"]["model"] = [nid, 0]


def splice_turbo_lora(wf: dict) -> None:
    """The step-distilled LoRA, plus the step count that is the point of it."""
    _splice_model_lora(wf, _N_TURBO_LORA, _TURBO_LORA, _TURBO_STRENGTH)
    wf[_N_SCHED]["inputs"]["steps"] = _TURBO_STEPS


def splice_nsfw_lora(wf: dict, strength: float = _NSFW_STRENGTH) -> None:
    """The sex LoRA. Weights only — no step count, no schedule of its own."""
    _splice_model_lora(wf, _N_NSFW_LORA, _NSFW_LORA, strength)


def nsfw_lora_strength(options: dict) -> float:
    """How strongly to apply the sex LoRA to this job — 0.0 for not at all.

    Decided from the prompt rather than asked of the caller, because every
    caller would have to answer it and the prompt already says. `nsfw_lora` in
    the options overrides either way: False to keep it off a clip that reads
    explicit, a number to pin the strength.
    """
    want = options.get("nsfw_lora")
    if want is False:
        return 0.0
    if isinstance(want, bool):          # True — on, at the default
        return _NSFW_STRENGTH
    if isinstance(want, (int, float)):
        return max(0.0, min(1.0, float(want)))
    return _NSFW_STRENGTH if _SEX_RE.search(str(options.get("prompt") or "")) else 0.0
