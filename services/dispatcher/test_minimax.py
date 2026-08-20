"""Tests for MiniMax H3 build + its three architecture-specific invariants."""
import json
import os

import minimax
from workflow_router import classify_workflow

HERE = os.path.dirname(__file__)
TEMPLATE = os.path.join(HERE, "minimax_templates", "MiniMaxH3Ref2VA.api.json")
REF = "ref.png"


def _build(**opts):
    return minimax.maybe_build_minimax({"prompt": "a woman dancing", **opts}, REF)


# --- detection -------------------------------------------------------------

def test_no_options_is_not_a_minimax_job():
    assert minimax.maybe_build_minimax(None, REF) is None
    assert minimax.maybe_build_minimax({}, REF) is None


def test_reference_image_is_required():
    assert minimax.maybe_build_minimax({"prompt": "x"}, None) is None


def test_builds_and_routes_to_video():
    wf = _build()
    assert wf is not None
    # None of the router's Wan/LTX patterns match a MiniMax graph, so this
    # asserts the MiniMaxH3/CreateVideo patterns were actually added.
    assert classify_workflow(wf) == "video"


def _assert_fully_reachable(wf):
    """Every link resolves, and every node is reachable from the SaveVideo."""
    for nid, node in wf.items():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in wf, f"{nid}.{k} -> {v[0]}"
    out = [i for i, x in wf.items() if x["class_type"] == "SaveVideo"]
    assert len(out) == 1
    seen = set()

    def walk(i):
        if i in seen:
            return
        seen.add(i)
        def links(val):
            # links can be nested inside an Autogrow group dict, not only at
            # the top level of inputs — missing that made this test report a
            # false orphan and sent me "fixing" a correct template.
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                yield val[0]
            elif isinstance(val, dict):
                for sub in val.values():
                    yield from links(sub)

        for v in wf[i]["inputs"].values():
            for dep in links(v):
                if dep in wf:
                    walk(dep)

    walk(out[0])
    assert seen == set(wf), f"unreachable: {sorted(set(wf) - seen)}"


def test_links_resolve_and_nothing_is_orphaned():
    _assert_fully_reachable(_build())


# --- invariant 1: 17k+5 frame grid at 24 fps -------------------------------

def test_snap_frames_matches_the_models_grid():
    # every snapped value must satisfy n % 17 == 5
    for s in (1, 2, 5, 5.2, 7, 10, 12.5, 15):
        n = minimax.snap_frames(s)
        assert n % 17 == 5, (s, n)


def test_five_seconds_is_the_documented_124_frames():
    assert minimax.snap_frames(5) == 124


def test_built_length_is_always_on_the_grid():
    for s in (1, 5, 8, 12, 20, 60):
        wf = _build(seconds=s)
        assert wf[minimax._N_REF2V]["inputs"]["length"] % 17 == 5


def test_frames_are_clamped_to_the_trained_band():
    assert _build(seconds=0.1)[minimax._N_REF2V]["inputs"]["length"] == minimax._MIN_FRAMES
    hi = _build(seconds=600)[minimax._N_REF2V]["inputs"]["length"]
    assert hi <= minimax._MAX_FRAMES and hi % 17 == 5


def test_clamp_does_not_overshoot_the_ceiling_when_walking_up():
    # walking up to the next 17k+5 must not push past _MAX_FRAMES
    for n in range(minimax._MAX_FRAMES - 20, minimax._MAX_FRAMES + 20):
        c = minimax.clamp_frames(n)
        assert c % 17 == 5 and c <= minimax._MAX_FRAMES, (n, c)


# --- invariant 2: area-capped canvas ---------------------------------------

def test_canvas_axes_are_multiples_of_32():
    for w, h in ((1344, 768), (1000, 700), (833, 481), (37, 37)):
        rw, rh = minimax.snap_canvas(w, h)
        assert rw % 32 == 0 and rh % 32 == 0, (w, h, rw, rh)


def test_canvas_respects_the_pixel_cap_not_just_rounding():
    # 1920x1080 is legal per-axis but far over the area cap — must be scaled.
    rw, rh = minimax.snap_canvas(1920, 1080)
    assert rw * rh <= minimax._MAX_PIXELS, (rw, rh)
    assert rw % 32 == 0 and rh % 32 == 0


def test_canvas_cap_holds_for_a_range_of_requests():
    for w, h in ((4000, 4000), (3000, 800), (800, 3000), (1344, 768), (1345, 769)):
        rw, rh = minimax.snap_canvas(w, h)
        assert rw * rh <= minimax._MAX_PIXELS, (w, h, rw, rh)


def test_default_canvas_survives_unchanged():
    # the node's own default is exactly at the cap and must not be shrunk
    assert minimax.snap_canvas(1344, 768) == (1344, 768)


# --- invariant 3: reference slot indexing ----------------------------------

def test_picture_1_is_reference_slot_zero():
    # ref_image_0 is what the prompt calls <Picture 1>; inverting this
    # misattributes every reference in a multi-shot script.
    wf = _build()
    ins = wf[minimax._N_REF2V]["inputs"]
    # execute() takes the Autogrow group as one ref_images dict; sending the
    # slots flat on the node raises "unexpected keyword argument 'ref_image_0'".
    assert "ref_image_0" not in ins
    assert ins[minimax._REF_GROUP] == {"ref_image_0": [minimax._N_REF_LOAD, 0]}
    assert wf[minimax._N_REF_LOAD]["inputs"]["image"] == REF


# --- parameters ------------------------------------------------------------

def test_prompt_carries_the_whole_multishot_document():
    doc = ("For the target video, at 0.00 seconds, <Picture 1> (from [Shot 1]) "
           "is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] ...")
    assert _build(prompt=doc)[minimax._N_REF2V]["inputs"]["prompt"] == doc


def test_seed_and_steps_overrides():
    wf = _build(seed=7, steps=12)
    assert wf[minimax._N_NOISE]["inputs"]["noise_seed"] == 7
    assert wf[minimax._N_SCHED]["inputs"]["steps"] == 12


def test_container_fps_is_pinned_to_24():
    # muxing at any other rate changes playback speed, not frame count
    assert _build()[minimax._N_VIDEO]["inputs"]["fps"] == 24.0


def test_uses_ada_compatible_weights_not_nvfp4():
    wf = _build()
    assert "nvfp4" not in wf[minimax._N_CLIP]["inputs"]["clip_name"]
    assert wf[minimax._N_CLIP]["inputs"]["type"] == "minimax"


def test_build_never_raises_on_bad_input():
    assert minimax.maybe_build_minimax({"width": "wide"}, REF) is None


def test_template_is_not_mutated():
    original = json.load(open(TEMPLATE))
    _build(seconds=12, width=900, seed=3)
    assert json.load(open(TEMPLATE)) == original


# --- handler routing -------------------------------------------------------

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
        "dispatcher_handler_minimax", os.path.join(HERE, "handler.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_minimax_without_a_reference_is_rejected_not_queued():
    h = _handler()
    resp = h._post_prompt({"body": json.dumps({
        "prompt": {},
        "minimax_options": {"prompt": "a woman dancing"},
    })})
    assert resp["statusCode"] == 400
    assert "reference image" in json.loads(resp["body"])["error"]


def test_empty_prompt_is_still_rejected_without_either_video_family():
    h = _handler()
    resp = h._post_prompt({"body": json.dumps({"prompt": {}})})
    assert resp["statusCode"] == 400
    assert "workflow JSON" in json.loads(resp["body"])["error"]


def test_minimax_jobs_go_to_the_minimax_queue():
    # A MiniMax job's type is "video" (classify_workflow says so), but it must
    # be queued to the MiniMax fleet — a video worker cannot hold its weights.
    os.environ["MINIMAX_QUEUE_URL"] = "https://sqs/minimax"
    os.environ["VIDEO_QUEUE_URL"] = "https://sqs/video"
    os.environ["IMAGE_QUEUE_URL"] = "https://sqs/image"
    import importlib.util
    import sys
    import types

    sent = {}

    class _Rec:
        def put_item(self, **kw): pass
        def update_item(self, **kw): pass
        def send_message(self, **kw): sent.update(kw)
        def __getattr__(self, _): return lambda **kw: {}

    rec = _Rec()
    sys.modules["boto3"] = types.SimpleNamespace(client=lambda *a, **k: rec)
    _bc = types.ModuleType("botocore")
    _exc = types.ModuleType("botocore.exceptions")
    _exc.ClientError = type("ClientError", (Exception,), {})
    _bc.exceptions = _exc
    sys.modules["botocore"] = _bc
    sys.modules["botocore.exceptions"] = _exc
    for k in ("JOBS_TABLE", "MODELS_TABLE", "OBJECT_INFO_TABLE"):
        os.environ.setdefault(k, "x")
    sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location(
        "dispatcher_handler_route", os.path.join(HERE, "handler.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    resp = mod._post_prompt({"body": json.dumps({
        "prompt": {},
        "minimax_options": {"prompt": "a woman dancing", "seconds": 5},
        "input_image": {"key": "uploads/2026/01/01/abc", "name": "ref.png"},
    })})
    assert resp["statusCode"] == 200, resp
    assert sent.get("QueueUrl") == "https://sqs/minimax", sent.get("QueueUrl")


# --- fl2v mode -------------------------------------------------------------
# FL2VA makes the image frame 0 rather than a conditioning reference.

def _fl2v(**opts):
    return minimax.maybe_build_minimax({"prompt": "a woman turning", "mode": "fl2v", **opts}, REF)


def test_fl2v_builds_and_routes_to_video():
    wf = _fl2v()
    assert wf is not None
    assert classify_workflow(wf) == "video"


def test_fl2v_uses_the_fl2va_checkpoint_not_ref2va():
    # Different training, not just a different node — the UNet must switch too.
    assert "fl2va" in _fl2v()[minimax._N_UNET]["inputs"]["unet_name"]
    assert "ref2va" in _build()[minimax._N_UNET]["inputs"]["unet_name"]


def test_fl2v_wires_the_image_as_frame_zero():
    wf = _fl2v()
    ins = wf[minimax._N_REF2V]["inputs"]
    assert wf[minimax._N_REF2V]["class_type"] == "MiniMaxH3ImageToVideo"
    assert ins["first_frame"] == [minimax._N_REF_RESIZE, 0]
    # ref-mode inputs must not leak into fl2v
    assert "ref_images" not in ins and "ref_image_size" not in ins
    assert wf[minimax._N_REF_LOAD]["inputs"]["image"] == REF


def test_fl2v_cover_crops_before_the_nodes_plain_stretch():
    # MiniMaxH3ImageToVideo resizes first_frame with crop "disabled" (a plain
    # stretch), so a mismatched aspect would distort frame 0 and propagate.
    wf = _fl2v(width=640, height=640)
    rs = wf[minimax._N_REF_RESIZE]["inputs"]
    assert rs["crop"] == "center"
    e = wf[minimax._N_REF2V]["inputs"]
    assert (rs["width"], rs["height"]) == (e["width"], e["height"])


def test_ref_mode_has_no_pre_crop_node():
    # The pre-crop exists only to defuse FL2VA's stretch; ref mode scales
    # references itself via ref_image_size.
    assert minimax._N_REF_RESIZE not in _build()


def test_fl2v_respects_frame_and_canvas_invariants():
    wf = _fl2v(seconds=12, width=900, height=500)
    e = wf[minimax._N_REF2V]["inputs"]
    assert e["length"] % 17 == 5
    assert e["width"] % 32 == 0 and e["height"] % 32 == 0
    assert e["width"] * e["height"] <= minimax._MAX_PIXELS


def test_fl2v_still_decodes_audio():
    # MiniMaxH3ImageToVideo takes no audio_vae, but the latent it emits is still
    # a NestedTensor (video, audio) — the audio VAE must still be loaded+decoded.
    wf = _fl2v()
    assert any(n["class_type"] == "VAEDecodeAudio" for n in wf.values())
    assert any(n["class_type"] == "VAELoader" and "audio" in n["inputs"]["vae_name"]
               for n in wf.values())


def test_unknown_minimax_mode_is_refused():
    assert minimax.maybe_build_minimax({"prompt": "x", "mode": "nope"}, REF) is None


def test_both_modes_carry_the_prompt_and_seed():
    for build in (_build, _fl2v):
        wf = build(prompt="a specific document", seed=5)
        assert wf[minimax._N_REF2V]["inputs"]["prompt"] == "a specific document"
        assert wf[minimax._N_NOISE]["inputs"]["noise_seed"] == 5


# --- auto first frame ------------------------------------------------------
# No upload used to be a hard 400. It can now mean "generate the opening frame
# for me": a Z-Image T2I subgraph is spliced into the same workflow and its
# decode feeds the MiniMax node, so the character comes from this app's own
# image stack instead of MiniMax's idea of the words.

AF = {"prompt": "a photo of a woman kneeling on a bed"}


def _autoframed(mode="fl2v", **af):
    return minimax.maybe_build_minimax(
        {"prompt": "a woman dancing", "mode": mode, "autoframe": {**AF, **af}}, None
    )


def test_autoframe_builds_without_any_upload():
    for mode in ("ref", "fl2v"):
        wf = _autoframed(mode)
        assert wf is not None, mode
        assert classify_workflow(wf) == "video", mode


def test_autoframe_graph_is_fully_connected():
    for mode in ("ref", "fl2v"):
        _assert_fully_reachable(_autoframed(mode))


def test_autoframe_drops_the_loader_nodes_it_replaces():
    # A LoadImage left behind would fail the job outright: its widget still
    # names a file that was never staged into ComfyUI's input/ dir.
    for mode in ("ref", "fl2v"):
        wf = _autoframed(mode)
        assert not [n for n in wf.values() if n["class_type"] == "LoadImage"], mode
    # fl2v's ImageScale exists only to cover-crop an upload to the canvas; the
    # still is rendered AT the canvas, so there is nothing to crop.
    assert not [n for n in _autoframed("fl2v").values()
                if n["class_type"] == "ImageScale"]


def test_autoframe_feeds_frame_zero_in_fl2v():
    wf = _autoframed("fl2v")
    src = wf["20"]["inputs"]["first_frame"]
    assert wf[src[0]]["class_type"] == "VAEDecode"


def test_autoframe_feeds_picture_one_in_ref_mode():
    wf = _autoframed("ref")
    src = wf["20"]["inputs"]["ref_images"]["ref_image_0"]
    assert wf[src[0]]["class_type"] == "VAEDecode"


def test_autoframe_renders_at_the_video_canvas():
    # Any mismatch here re-introduces the crop this splice exists to avoid.
    wf = _autoframed("fl2v")
    r2v = wf["20"]["inputs"]
    latent = wf[minimax._AF_LATENT]["inputs"]
    assert (latent["width"], latent["height"]) == (r2v["width"], r2v["height"])


def test_autoframe_canvas_follows_the_snapped_size_not_the_request():
    wf = minimax.maybe_build_minimax(
        {"prompt": "x", "mode": "fl2v", "width": 1920, "height": 1080,
         "autoframe": AF}, None)
    latent = wf[minimax._AF_LATENT]["inputs"]
    assert (latent["width"], latent["height"]) == minimax.snap_canvas(1920, 1080)


def test_autoframe_shares_the_clips_seed():
    # One seed for the pair, so re-queueing a job reproduces both the opening
    # frame and the motion rather than only half of it.
    wf = minimax.maybe_build_minimax(
        {"prompt": "x", "mode": "fl2v", "seed": 4242, "autoframe": AF}, None)
    assert wf[minimax._AF_KSAMPLER]["inputs"]["seed"] == 4242
    assert wf["30"]["inputs"]["noise_seed"] == 4242


def test_autoframe_uses_only_core_comfyui_nodes():
    # This runs on the VIDEO worker image, whose baked custom-node set is three
    # packs wide. A rgthree/Impact node here would fail at graph validation.
    core = {"UNETLoader", "CLIPLoader", "CLIPSetLastLayer", "VAELoader",
            "CLIPTextEncode", "EmptySD3LatentImage", "KSampler", "VAEDecode",
            "LoraLoader"}
    wf = _autoframed("fl2v", loras=[{"name": "x.safetensors", "strength": 0.6}])
    for nid, node in wf.items():
        if int(nid) >= 9100:
            assert node["class_type"] in core, f"{nid} {node['class_type']}"


def test_autoframe_loras_ride_the_clip_as_well_as_the_model():
    # A model-only chain silently drops the half of a character LoRA that lives
    # in the text encoder — the trained token then means nothing.
    wf = _autoframed("fl2v", loras=[{"name": "igbaddie.safetensors", "strength": 0.6}])
    lora = str(minimax._AF_LORA_BASE)
    assert wf[lora]["inputs"]["lora_name"] == "igbaddie.safetensors"
    assert wf[minimax._AF_KSAMPLER]["inputs"]["model"] == [lora, 0]
    assert wf[minimax._AF_POS]["inputs"]["clip"] == [lora, 1]
    assert wf[minimax._AF_NEG]["inputs"]["clip"] == [lora, 1]


def test_autoframe_lora_chain_stays_in_order():
    wf = _autoframed("fl2v", loras=[
        {"name": "a.safetensors", "strength": 1.0},
        {"name": "b.safetensors", "strength": 0.5},
    ])
    a, b = str(minimax._AF_LORA_BASE), str(minimax._AF_LORA_BASE + 1)
    assert wf[b]["inputs"]["model"] == [a, 0]
    assert wf[b]["inputs"]["clip"] == [a, 1]
    assert wf[minimax._AF_KSAMPLER]["inputs"]["model"] == [b, 0]


def test_autoframe_rejects_lora_paths_and_absurd_strengths():
    wf = _autoframed("fl2v", loras=[
        {"name": "../../etc/passwd", "strength": 1.0},
        {"name": "sub/dir.safetensors", "strength": 1.0},
        {"name": "loud.safetensors", "strength": 99},
        {"name": "", "strength": 1.0},
    ])
    assert not [n for n in wf.values() if n["class_type"] == "LoraLoader"]


def test_autoframe_caps_the_lora_chain():
    wf = _autoframed("fl2v", loras=[
        {"name": f"l{i}.safetensors", "strength": 0.5} for i in range(20)])
    assert len([n for n in wf.values() if n["class_type"] == "LoraLoader"]) == 8


def test_an_upload_still_wins_over_autoframe():
    # Explicit beats implicit: if the user actually uploaded a frame, use it.
    wf = minimax.maybe_build_minimax(
        {"prompt": "x", "mode": "fl2v", "autoframe": AF}, REF)
    assert wf["10"]["inputs"]["image"] == REF
    assert minimax._AF_KSAMPLER not in wf


def test_autoframe_carries_its_own_prompt_not_the_clips():
    wf = _autoframed("fl2v")
    assert wf[minimax._AF_POS]["inputs"]["text"] == AF["prompt"]
    assert wf[minimax._AF_POS]["inputs"]["text"] != wf["20"]["inputs"]["prompt"]


def test_autoframe_has_a_negative_by_default_and_accepts_an_override():
    assert wf_neg(_autoframed("fl2v")) == minimax._AUTOFRAME_NEGATIVE
    assert wf_neg(_autoframed("fl2v", negative="hands")) == "hands"


def wf_neg(wf):
    return wf[minimax._AF_NEG]["inputs"]["text"]


def test_autoframe_must_be_a_dict_not_a_truthy_flag():
    # `autoframe: true` from a sloppy client must not build a graph whose
    # subject prompt is empty — that is a blank-faced clip, not an error.
    assert minimax.maybe_build_minimax(
        {"prompt": "x", "mode": "fl2v", "autoframe": True}, None) is None


def test_template_is_not_mutated_by_autoframe():
    before = json.load(open(TEMPLATE))
    _autoframed("ref")
    assert json.load(open(TEMPLATE)) == before


def test_handler_accepts_autoframe_without_an_upload():
    h = _handler()
    resp = h._post_prompt({"body": json.dumps({
        "prompt": {},
        "minimax_options": {"prompt": "a woman dancing", "mode": "fl2v",
                            "autoframe": {"prompt": "a photo of a woman"}},
    })})
    assert resp["statusCode"] != 400, resp["body"]


def test_autoframe_default_model_is_a_real_zimage_checkpoint():
    # A UNETLoader pointed at a name the catalog does not have fails the job at
    # load time, after MiniMax's weights are already resident.
    import zimage
    assert zimage._normalize_model(
        minimax._AUTOFRAME_DEFAULTS["model"]) in zimage.ZIMAGE_MODELS


def test_autoframe_model_override_is_allowlisted():
    # The whole model catalog is mounted on this fleet, so a free-form model
    # name here would let a caller load anything into the UNETLoader.
    wf = _autoframed("fl2v", model="krea2_raw_fp8_scaled.safetensors")
    assert wf[minimax._AF_UNET]["inputs"]["unet_name"] == \
        minimax._AUTOFRAME_DEFAULTS["model"]
    wf = _autoframed("fl2v", model="zit_v12.safetensors")
    assert wf[minimax._AF_UNET]["inputs"]["unet_name"] == "zit_v12.safetensors"
