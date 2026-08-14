"""Tests for SCAIL-2 build / invariants / parameter clamping."""
import json
import os

import scail
from workflow_router import classify_workflow

HERE = os.path.dirname(__file__)
TEMPLATE = os.path.join(HERE, "scail_templates", "Scail2PoseControl.api.json")

REF = "ref.png"
DRIVE = "drive.mp4"


def _build(**opts):
    return scail.maybe_build_scail({"prompt": "a woman dancing", **opts}, REF, DRIVE)


# --- detection -------------------------------------------------------------

def test_no_options_is_not_a_scail_job():
    assert scail.maybe_build_scail(None, REF, DRIVE) is None
    assert scail.maybe_build_scail({}, REF, DRIVE) is None
    assert scail.maybe_build_scail("nope", REF, DRIVE) is None


def test_requires_both_reference_image_and_driving_video():
    opts = {"prompt": "x"}
    assert scail.maybe_build_scail(opts, None, DRIVE) is None
    assert scail.maybe_build_scail(opts, REF, None) is None
    assert scail.maybe_build_scail(opts, REF, DRIVE) is not None


# --- structure -------------------------------------------------------------

def test_builds_and_routes_to_the_video_fleet():
    wf = _build()
    assert wf is not None
    # The router is what puts this on video-jobs rather than image-jobs.
    assert classify_workflow(wf) == "video"


def test_every_link_resolves_and_nothing_is_orphaned():
    wf = _build()
    for nid, node in wf.items():
        for key, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in wf, f"{nid}.{key} -> missing node {val[0]}"

    out = [i for i, x in wf.items() if x["class_type"] == "VHS_VideoCombine"]
    assert len(out) == 1
    seen = set()

    def walk(i):
        if i in seen:
            return
        seen.add(i)
        for val in wf[i]["inputs"].values():
            if isinstance(val, list) and len(val) == 2 and val[0] in wf:
                walk(val[0])

    walk(out[0])
    assert seen == set(wf), f"unreachable nodes: {sorted(set(wf) - seen)}"


def test_reference_and_driving_inputs_are_wired():
    wf = _build()
    assert wf[scail._N_REF_LOAD]["inputs"]["image"] == REF
    assert wf[scail._N_DRIVE_LOAD]["inputs"]["video"] == DRIVE


# --- invariant 1: pose is half the generation resolution -------------------

def test_pose_resolution_is_exactly_half_generation_resolution():
    for w, h in ((512, 896), (640, 640), (768, 1280)):
        wf = _build(width=w, height=h)
        gen_w = wf[scail._N_EMPTY]["inputs"]["width"]
        gen_h = wf[scail._N_EMPTY]["inputs"]["height"]
        pose = wf[scail._N_POSE_RENDER]["inputs"]
        assert pose["width"] * 2 == gen_w, (w, h, pose["width"], gen_w)
        assert pose["height"] * 2 == gen_h, (w, h, pose["height"], gen_h)


def test_resize_nodes_match_generation_resolution():
    wf = _build(width=640, height=640)
    for nid in (scail._N_REF_RESIZE, scail._N_DRIVE_RESIZE):
        assert wf[nid]["inputs"]["width"] == 640
        assert wf[nid]["inputs"]["height"] == 640


# --- invariant 2: frame count is 4n+1 --------------------------------------

def test_frame_count_snaps_to_4n_plus_1():
    for req in (81, 82, 83, 84, 85, 100, 161, 200):
        wf = _build(frames=req)
        got = wf[scail._N_EMPTY]["inputs"]["num_frames"]
        assert got % 4 == 1, (req, got)
        assert got <= max(req, scail._MIN_FRAMES)


def test_seconds_convert_to_frames_at_native_fps():
    wf = _build(seconds=10)
    got = wf[scail._N_EMPTY]["inputs"]["num_frames"]
    assert got % 4 == 1
    # 10 s * 16 fps = 160 -> nearest valid 4n+1 at or below
    assert got == 157


def test_frame_count_is_clamped_to_bounds():
    assert _build(frames=1)[scail._N_EMPTY]["inputs"]["num_frames"] == scail._MIN_FRAMES
    hi = _build(frames=99999)[scail._N_EMPTY]["inputs"]["num_frames"]
    assert hi <= scail._MAX_FRAMES and hi % 4 == 1


def test_driving_video_supplies_exactly_the_generated_frame_count():
    for req in (81, 161, 200):
        wf = _build(frames=req)
        assert (
            wf[scail._N_DRIVE_LOAD]["inputs"]["frame_load_cap"]
            == wf[scail._N_EMPTY]["inputs"]["num_frames"]
        )


# --- invariant 3: dimensions divisible by 32 -------------------------------

def test_dimensions_snap_to_multiples_of_32():
    wf = _build(width=500, height=900)
    assert wf[scail._N_EMPTY]["inputs"]["width"] % 32 == 0
    assert wf[scail._N_EMPTY]["inputs"]["height"] % 32 == 0


def test_dimensions_are_clamped_to_bounds():
    lo = _build(width=16, height=16)
    assert lo[scail._N_EMPTY]["inputs"]["width"] >= scail._MIN_DIM
    hi = _build(width=99999, height=99999)
    assert hi[scail._N_EMPTY]["inputs"]["height"] <= scail._MAX_DIM


# --- context windows -------------------------------------------------------

def test_context_window_never_exceeds_the_clip_length():
    wf = _build(frames=81)
    ctx = wf[scail._N_CONTEXT]["inputs"]
    assert ctx["context_frames"] <= wf[scail._N_EMPTY]["inputs"]["num_frames"]
    assert ctx["context_overlap"] < ctx["context_frames"]


def test_long_clips_keep_the_native_window():
    wf = _build(frames=321)
    assert wf[scail._N_CONTEXT]["inputs"]["context_frames"] == 81


# --- parameters ------------------------------------------------------------

def test_prompts_are_carried_through():
    wf = _build(prompt="a knight running", negative_prompt="blurry")
    assert wf[scail._N_TEXT]["inputs"]["positive_prompt"] == "a knight running"
    assert wf[scail._N_TEXT]["inputs"]["negative_prompt"] == "blurry"


def test_default_negative_is_applied_when_absent():
    wf = _build()
    assert wf[scail._N_TEXT]["inputs"]["negative_prompt"] == scail._DEFAULT_NEGATIVE


def test_seed_is_honoured_and_bounded():
    assert _build(seed=42)[scail._N_SAMPLER]["inputs"]["seed"] == 42
    assert 0 <= _build()[scail._N_SAMPLER]["inputs"]["seed"] < scail._SEED_MAX


def test_cfg_defaults_to_1_for_the_distill_lora():
    # The baked lightx2v cfg-step-distill lora washes out at cfg > 1, so the
    # default must stay 1.0 unless the caller explicitly overrides.
    assert _build()[scail._N_SAMPLER]["inputs"]["cfg"] == 1.0
    assert _build(cfg=3.5)[scail._N_SAMPLER]["inputs"]["cfg"] == 3.5


def test_steps_and_lora_strength_overrides():
    wf = _build(steps=8, lora_strength=0.7)
    assert wf[scail._N_SCHED]["inputs"]["steps"] == 8
    assert wf[scail._N_LORA]["inputs"]["strength"] == 0.7


# --- failure handling ------------------------------------------------------

def test_build_never_raises_on_bad_input():
    assert scail.maybe_build_scail({"width": "wide"}, REF, DRIVE) is None


def test_template_is_not_mutated_between_builds():
    original = json.load(open(TEMPLATE))
    _build(frames=161, width=768, seed=7)
    assert json.load(open(TEMPLATE)) == original


# --- handler: driving-video filename sanitising ----------------------------
# Imported the same way test_source.py does (stubbed boto3, dummy env) so the
# module-level AWS clients don't need real credentials.

def _handler():
    import importlib.util
    import sys
    import types

    class _Inert:
        def __getattr__(self, _):
            return lambda **kw: None

    sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *a, **k: _Inert()))
    _bc = types.ModuleType("botocore")
    _exc = types.ModuleType("botocore.exceptions")
    _exc.ClientError = type("ClientError", (Exception,), {})
    _bc.exceptions = _exc
    sys.modules.setdefault("botocore", _bc)
    sys.modules.setdefault("botocore.exceptions", _exc)
    for k in ("JOBS_TABLE", "MODELS_TABLE", "OBJECT_INFO_TABLE",
              "IMAGE_QUEUE_URL", "VIDEO_QUEUE_URL"):
        os.environ.setdefault(k, "x")
    sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location(
        "dispatcher_handler_scail", os.path.join(HERE, "handler.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_video_name_is_sanitised_and_traversal_is_stripped():
    h = _handler()
    assert h._safe_input_video_name("dance.mp4") == "dance.mp4"
    assert h._safe_input_video_name("../../etc/passwd") == "passwd.mp4"
    assert "/" not in h._safe_input_video_name("a/b/c.mov")
    assert h._safe_input_video_name("") == ""
    assert h._safe_input_video_name("..") == ""


def test_video_name_gains_an_extension_vhs_accepts():
    h = _handler()
    assert h._safe_input_video_name("clip").endswith(".mp4")
    # already-valid container extensions are preserved
    for ext in (".mp4", ".webm", ".mov", ".mkv"):
        assert h._safe_input_video_name("clip" + ext) == "clip" + ext


def test_unbuildable_scail_request_is_rejected_not_queued_as_an_image_job():
    # Without this the empty placeholder graph falls through to the image fleet
    # and returns 0 outputs instead of an error.
    h = _handler()
    resp = h._post_prompt({"body": json.dumps({
        "prompt": {},
        "scail_options": {"prompt": "a woman dancing"},
        # no input_image / input_video
    })})
    assert resp["statusCode"] == 400
    assert "reference image" in json.loads(resp["body"])["error"]
