"""
SCAIL-2 video motion transfer.

SCAIL (https://github.com/zai-org/SCAIL) is a Wan 2.1 14B derivative that takes
a **reference character image** plus a **driving pose sequence** and generates
video in which the reference character performs the driving motion. Unlike a
plain I2V model it conditions on the reference latent for the whole denoise
(``WanVideoAddSCAILReferenceEmbeds``), which is what keeps the face from
drifting across a long clip.

This family differs from anima/zimage/krea in kind. Those are **rewriters**: the
frontend submits a real SDXL-shaped graph and we substitute an equivalent one
when we spot a model we know needs different loaders. SCAIL has no SDXL-shaped
equivalent — the inputs are a reference image, a driving video and a duration,
none of which fit a txt2img graph. So this module is a **builder**: given
``scail_options`` it discards whatever placeholder the caller sent and emits the
``Scail2PoseControl`` template patched with the caller's parameters. Detection is
therefore the presence of ``scail_options``, not a model allowlist.

The template mirrors kijai's reference workflow
(``wanvideo_2_1_14B_SCAIL_pose_control_example_01.json``) and runs on the video
fleet: the router sees ``WanVideo*`` class_types and queues to video-jobs.

Three invariants are enforced here rather than left to the caller, because each
one fails *silently* — producing garbage frames instead of an error:

  1. **Pose resolution is exactly half the generation resolution.** SCAIL
     consumes downsampled poses; feeding full-resolution poses misaligns every
     joint and the character flails.
  2. **Frame count is 4n+1.** Wan's VAE has a temporal stride of 4 with one
     extra head frame. An off-stride count silently truncates the tail.
  3. **Dimensions divisible by 32.** Wan patchifies 2x2 over an 8x-downsampled
     latent; anything else is cropped without warning.

The build never raises: any failure returns None and leaves the caller's
workflow intact.
"""
from __future__ import annotations

import copy
import json
import os
import random
from typing import Optional

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "scail_templates", "Scail2PoseControl.api.json"
)

# Match the other families' seed cap for cross-family consistency.
_SEED_MAX = 2 ** 50

# Wan 2.1 is trained on 81-frame clips at 16 fps. WanVideoContextOptions slides
# an 81-frame window across a longer request, so length is bounded by patience
# and the 840 s worker timeout rather than by the model. 1001 frames is ~62 s of
# video and already far past what a single job should attempt on an A10G; it's a
# guard against a fat-fingered request, not a model limit.
_MIN_FRAMES = 81
_MAX_FRAMES = 1001
_NATIVE_FPS = 16

# Generation resolution bounds. The A10G has 24 GB and the template block-swaps
# 25 blocks; 720p portrait is about the ceiling before it thrashes.
_MIN_DIM = 256
_MAX_DIM = 1280

# Node ids in the template we patch. Kept as named constants so a template
# renumbering is a one-line fix rather than a scavenger hunt.
_N_LORA = "2"
_N_TEXT = "5"
_N_REF_LOAD = "10"
_N_REF_RESIZE = "11"
_N_DRIVE_LOAD = "20"
_N_DRIVE_RESIZE = "21"
_N_POSE_RENDER = "35"
_N_EMPTY = "40"
_N_CONTEXT = "50"
_N_SCHED = "52"
_N_SAMPLER = "53"
_N_SAVE = "55"

_DEFAULT_NEGATIVE = (
    "bright colors, overexposed, static, blurred details, subtitles, style, "
    "artwork, painting, picture, still, overall gray, worst quality, low quality, "
    "JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, malformed limbs, fused fingers, cluttered background, "
    "three legs, walking backwards"
)


def _snap_dim(v: int) -> int:
    """Round a dimension to the nearest multiple of 32, within bounds."""
    v = max(_MIN_DIM, min(_MAX_DIM, int(v)))
    return max(_MIN_DIM, int(round(v / 32.0)) * 32)


def _snap_frames(v: int) -> int:
    """Round a frame count to the nearest 4n+1, within bounds.

    Wan's VAE compresses time 4x plus a head frame, so valid lengths are
    1, 5, 9, ... 81, 85. Rounding rather than rejecting keeps a 'give me 10
    seconds' request working instead of erroring on arithmetic the caller
    shouldn't have to know.
    """
    v = max(_MIN_FRAMES, min(_MAX_FRAMES, int(v)))
    return ((v - 1) // 4) * 4 + 1


def _resolve_frames(options: dict) -> int:
    """Frame count from an explicit `frames`, else `seconds` at native fps."""
    if options.get("frames") is not None:
        return _snap_frames(options["frames"])
    if options.get("seconds") is not None:
        return _snap_frames(float(options["seconds"]) * _NATIVE_FPS)
    return _MIN_FRAMES


def _load_template() -> Optional[dict]:
    try:
        with open(_TEMPLATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        print(f"scail: template load failed: {e!r}")
        return None


def maybe_build_scail(
    options: Optional[dict],
    ref_image_name: Optional[str] = None,
    drive_video_name: Optional[str] = None,
) -> Optional[dict]:
    """Build a SCAIL-2 workflow, or None if this isn't a SCAIL job.

    ``options`` is the request's ``scail_options``. A reference image is
    mandatory — SCAIL's whole purpose is transferring motion onto *that*
    character, so without it there is nothing to build. A driving video is also
    mandatory: the pose branch has no other source.
    """
    if not isinstance(options, dict) or not options:
        return None
    if not ref_image_name or not drive_video_name:
        print("scail: missing reference image or driving video; not building")
        return None

    wf = _load_template()
    if wf is None:
        return None

    try:
        wf = copy.deepcopy(wf)

        width = _snap_dim(options.get("width", 512))
        height = _snap_dim(options.get("height", 896))
        frames = _resolve_frames(options)

        # Invariant 1: poses are consumed at exactly half the generation size.
        pose_w, pose_h = width // 2, height // 2

        wf[_N_REF_LOAD]["inputs"]["image"] = ref_image_name
        wf[_N_DRIVE_LOAD]["inputs"]["video"] = drive_video_name

        for nid in (_N_REF_RESIZE, _N_DRIVE_RESIZE):
            wf[nid]["inputs"]["width"] = width
            wf[nid]["inputs"]["height"] = height

        wf[_N_POSE_RENDER]["inputs"]["width"] = pose_w
        wf[_N_POSE_RENDER]["inputs"]["height"] = pose_h

        wf[_N_EMPTY]["inputs"]["width"] = width
        wf[_N_EMPTY]["inputs"]["height"] = height
        wf[_N_EMPTY]["inputs"]["num_frames"] = frames

        # The driving video must supply exactly as many frames as we generate:
        # fewer and the pose tensor is shorter than the latent (the sampler
        # errors); more and the tail is silently discarded.
        wf[_N_DRIVE_LOAD]["inputs"]["frame_load_cap"] = frames
        if options.get("select_every_nth") is not None:
            wf[_N_DRIVE_LOAD]["inputs"]["select_every_nth"] = max(
                1, int(options["select_every_nth"])
            )

        wf[_N_TEXT]["inputs"]["positive_prompt"] = str(options.get("prompt", "") or "")
        wf[_N_TEXT]["inputs"]["negative_prompt"] = str(
            options.get("negative_prompt") or _DEFAULT_NEGATIVE
        )

        seed = options.get("seed")
        wf[_N_SAMPLER]["inputs"]["seed"] = (
            int(seed) % _SEED_MAX if seed is not None else random.randint(0, _SEED_MAX - 1)
        )

        if options.get("steps") is not None:
            wf[_N_SCHED]["inputs"]["steps"] = max(1, min(60, int(options["steps"])))
        if options.get("shift") is not None:
            wf[_N_SCHED]["inputs"]["shift"] = float(options["shift"])
        # cfg stays at 1.0 unless asked: the baked lightx2v lora is a
        # cfg-step-distill, and any cfg > 1 with it produces washed-out frames.
        if options.get("cfg") is not None:
            wf[_N_SAMPLER]["inputs"]["cfg"] = float(options["cfg"])
        if options.get("lora_strength") is not None:
            wf[_N_LORA]["inputs"]["strength"] = float(options["lora_strength"])

        # Context window must not exceed what we're generating — a window wider
        # than the clip degenerates to a single pass with a misleading overlap.
        ctx = wf[_N_CONTEXT]["inputs"]
        ctx["context_frames"] = min(int(ctx["context_frames"]), frames)
        if ctx["context_overlap"] >= ctx["context_frames"]:
            ctx["context_overlap"] = max(0, ctx["context_frames"] // 2)

        return wf
    except Exception as e:  # noqa: BLE001
        print(f"scail: build failed, leaving workflow untouched: {e!r}")
        return None
