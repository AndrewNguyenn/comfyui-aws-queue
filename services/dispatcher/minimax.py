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

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "minimax_templates", "MiniMaxH3Ref2VA.api.json"
)

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
_REF_GROUP = "ref_images"        # Autogrow group; slots ref_image_0..8 live inside it
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


def _load_template() -> Optional[dict]:
    try:
        with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        print(f"minimax: template load failed: {e!r}")
        return None


def maybe_build_minimax(
    options: Optional[dict],
    ref_image_name: Optional[str] = None,
) -> Optional[dict]:
    """Build a MiniMax H3 ref2va workflow, or None if this isn't one.

    ``options`` is the request's ``minimax_options``. A reference image is
    required: this is the reference-to-video graph, and the prompt's
    ``<Picture 1>`` has to resolve to something.
    """
    if not isinstance(options, dict) or not options:
        return None
    if not ref_image_name:
        print("minimax: missing reference image; not building")
        return None

    wf = _load_template()
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

        wf[_N_REF_LOAD]["inputs"]["image"] = ref_image_name

        r2v = wf[_N_REF2V]["inputs"]
        r2v["prompt"] = str(options.get("prompt", "") or "")
        r2v["width"] = width
        r2v["height"] = height
        r2v["length"] = frames
        if options.get("ref_image_size") in ("match", "max"):
            r2v["ref_image_size"] = options["ref_image_size"]

        seed = options.get("seed")
        wf[_N_NOISE]["inputs"]["noise_seed"] = (
            int(seed) % _SEED_MAX if seed is not None else random.randint(0, _SEED_MAX - 1)
        )

        if options.get("steps") is not None:
            wf[_N_SCHED]["inputs"]["steps"] = max(1, min(60, int(options["steps"])))

        # Keep the container's frame rate tied to the model's: MiniMax H3 is a
        # 24 fps model, and muxing its frames at any other rate changes the
        # playback speed rather than the frame count.
        wf[_N_VIDEO]["inputs"]["fps"] = float(FPS)

        return wf
    except Exception as e:  # noqa: BLE001
        print(f"minimax: build failed, leaving workflow untouched: {e!r}")
        return None
