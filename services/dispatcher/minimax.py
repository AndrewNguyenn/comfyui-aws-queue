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
from typing import Optional

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
_N_VIDEO = "50"


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
# curated character pool, a natural-prompt library and an always-on house LoRA,
# and none of it reaches a T2V clip.
#
# So instead of dropping the image, we GENERATE it: a small Z-Image text-to-
# image graph is spliced into the same workflow, and its decode feeds
# ``first_frame``. One job, one queue, one GPU — ComfyUI unloads Z-Image before
# MiniMax's 20 GiB transformer loads, so the peak is unchanged and the cost is
# a handful of seconds against a multi-minute sample.
#
# Every node here is core ComfyUI (no custom pack), which matters because this
# runs on the VIDEO worker image, whose baked node set is deliberately small.
_AUTOFRAME_DEFAULTS = {
    "model": "Z Image.safetensors",
    "clip": "qwen_3_4b_fp8_mixed.safetensors",
    "clip_type": "lumina2",
    "clip_layer": -2,
    "vae": "ae_zimgturbo.safetensors",
    "steps": 11,
    "cfg": 1.5,
    "sampler": "euler",
    "scheduler": "simple",
}
# Mirrors the image fleet's Z-Image negative. Short on purpose: Z-Image runs at
# cfg 1.5, where a long negative costs more than it corrects.
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
_AF_LORA_BASE = 9120
_AF_MAX_LORAS = 8


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


def splice_autoframe(wf: dict, mode: str, options: dict,
                     width: int, height: int, seed: int) -> None:
    """Add a Z-Image T2I subgraph and hand its image to the MiniMax node.

    Replaces the template's LoadImage (and, in fl2v, the ImageScale that
    cover-crops for it): the still is rendered AT the video canvas, so there is
    nothing left to crop.
    """
    o = dict(_AUTOFRAME_DEFAULTS)
    for k in ("model", "clip", "clip_type", "vae", "sampler", "scheduler"):
        v = options.get(k)
        if isinstance(v, str) and v.strip():
            o[k] = v.strip()
    if options.get("steps") is not None:
        o["steps"] = max(1, min(50, int(options["steps"])))
    if options.get("cfg") is not None:
        o["cfg"] = max(0.0, min(20.0, float(options["cfg"])))

    positive = str(options.get("prompt", "") or "").strip()
    negative = str(options.get("negative", "") or "").strip() or _AUTOFRAME_NEGATIVE

    wf[_AF_UNET] = {"class_type": "UNETLoader",
                    "inputs": {"unet_name": o["model"], "weight_dtype": "default"}}
    wf[_AF_CLIP] = {"class_type": "CLIPLoader",
                    "inputs": {"clip_name": o["clip"], "type": o["clip_type"],
                               "device": "default"}}
    wf[_AF_CLIPSET] = {"class_type": "CLIPSetLastLayer",
                       "inputs": {"stop_at_clip_layer": int(o["clip_layer"]),
                                  "clip": [_AF_CLIP, 0]}}
    wf[_AF_VAE] = {"class_type": "VAELoader", "inputs": {"vae_name": o["vae"]}}

    # LoRAs ride between the loaders and every consumer, so the text encoders
    # see the trained tokens too — a model-only chain silently drops the half
    # of a character LoRA that lives in the CLIP.
    model_src, clip_src = [_AF_UNET, 0], [_AF_CLIPSET, 0]
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
    wf[_AF_KSAMPLER] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": o["steps"], "cfg": o["cfg"],
        "sampler_name": o["sampler"], "scheduler": o["scheduler"], "denoise": 1.0,
        "model": model_src, "positive": [_AF_POS, 0], "negative": [_AF_NEG, 0],
        "latent_image": [_AF_LATENT, 0],
    }}
    wf[_AF_DECODE] = {"class_type": "VAEDecode",
                      "inputs": {"samples": [_AF_KSAMPLER, 0], "vae": [_AF_VAE, 0]}}

    # Hand the generated still to the MiniMax node the same way an upload
    # would arrive, then drop the now-unreachable loader nodes.
    if mode == "fl2v":
        wf[_N_REF2V]["inputs"]["first_frame"] = [_AF_DECODE, 0]
        wf.pop(_N_REF_RESIZE, None)
    else:
        wf[_N_REF2V]["inputs"][_REF_GROUP] = {"ref_image_0": [_AF_DECODE, 0]}
    wf.pop(_N_REF_LOAD, None)
