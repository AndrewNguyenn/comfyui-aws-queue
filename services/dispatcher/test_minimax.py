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


def test_autoframe_uses_only_nodes_the_video_image_bakes():
    # Every node here has to exist on the VIDEO worker, whose custom-node set is
    # deliberately small. ClownsharKSampler_Beta is allowed because RES4LYF is
    # baked for exactly this (workers/video/baked_nodes.txt) — anything else
    # from a pack the image does not carry fails graph validation on every clip.
    allowed = {"UNETLoader", "CLIPLoader", "CLIPSetLastLayer", "VAELoader",
               "CLIPTextEncode", "EmptySD3LatentImage", "KSampler", "VAEDecode",
               "LoraLoader", "LoraLoaderModelOnly", "ClownsharKSampler_Beta"}
    for model in ("zit_v12.safetensors", "krea2_raw_fp8_scaled.safetensors"):
        wf = _autoframed("fl2v", model=model,
                         loras=[{"name": "x.safetensors", "strength": 0.6}])
        for nid, node in wf.items():
            if int(nid) >= 9100:
                assert node["class_type"] in allowed, f"{model} {nid} {node['class_type']}"


def test_res4lyf_is_actually_baked_into_the_video_worker():
    # The dispatcher and the image have to agree. Shipping the sampler switch
    # ahead of the rebuild fails every autoframed job at graph validation.
    here = os.path.dirname(os.path.abspath(__file__))
    baked = os.path.join(here, "..", "..", "workers", "video", "baked_nodes.txt")
    with open(baked, encoding="utf-8") as fh:
        packs = [l.split()[0] for l in fh if l.strip() and not l.startswith("#")]
    assert "RES4LYF" in packs, packs


def test_krea_frame_zero_samples_exactly_like_the_image_fleet():
    # The whole point of baking RES4LYF: frame 0 must not be an approximation of
    # the gallery look. Compare against the template rather than restating it,
    # so a change on the image side surfaces here instead of drifting silently.
    tpl = json.load(open(os.path.join(
        HERE, "krea_templates", "Krea2Simple.api.json")))
    wf = _autoframed("fl2v")
    for ours, theirs, denoise in ((minimax._AF_KSAMPLER, "8", 1.0),
                                  (minimax._AF_REFINE, "9", 0.2)):
        a, b = wf[ours]["inputs"], tpl[theirs]["inputs"]
        assert wf[ours]["class_type"] == tpl[theirs]["class_type"]
        for k in ("sampler_name", "scheduler", "eta", "cfg", "bongmath"):
            assert a[k] == b[k], f"{ours}.{k}: {a[k]!r} != {b[k]!r}"
        assert a["denoise"] == denoise
    # Steps differ from the raw template on purpose: _KREA_MODEL_SAMPLER
    # overrides the base pass to 8 for this checkpoint.
    import krea
    assert wf[minimax._AF_KSAMPLER]["inputs"]["steps"] == \
        krea._KREA_MODEL_SAMPLER["redcraftminimaxh3redmix_30krea2"]["steps"]
    assert wf[minimax._AF_REFINE]["inputs"]["steps"] == tpl["9"]["inputs"]["steps"]


def test_zimage_frame_zero_stays_on_the_core_sampler():
    # Z-Image's graph never used RES4LYF; baking it must not drag Z-Image along.
    wf = _autoframed("fl2v", model="zit_v12.safetensors")
    assert wf[minimax._AF_KSAMPLER]["class_type"] == "KSampler"
    assert "eta" not in wf[minimax._AF_KSAMPLER]["inputs"]


def test_autoframe_trigger_token_is_owned_by_the_family():
    # A prompt written for one family must never arrive carrying the other's
    # token, so the dispatcher prepends it rather than the caller.
    z = _autoframed("fl2v", model="zit_v12.safetensors")
    assert z[minimax._AF_POS]["inputs"]["text"].startswith("igbaddie,")
    # Krea's house stack has no trigger — one would be a junk token.
    assert "igbaddie" not in _autoframed("fl2v")[minimax._AF_POS]["inputs"]["text"]
    # And it is not doubled when the prompt already has it.
    once = _autoframed("fl2v", model="zit_v12.safetensors",
                       prompt="igbaddie, a photo")[minimax._AF_POS]["inputs"]["text"]
    assert once.count("igbaddie") == 1, once


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


def test_autoframe_default_model_is_a_real_catalogued_checkpoint():
    # A UNETLoader pointed at a name the catalog does not have fails the job at
    # load time, after MiniMax's weights are already resident.
    assert minimax.autoframe_family(minimax._AUTOFRAME_DEFAULT_MODEL) is not None


def test_autoframe_model_override_is_allowlisted():
    # The whole model catalog is mounted on this fleet, so a free-form model
    # name here would let a caller load anything into the UNETLoader.
    wf = _autoframed("fl2v", model="../../etc/passwd")
    assert wf[minimax._AF_UNET]["inputs"]["unet_name"] == \
        minimax._AUTOFRAME_DEFAULT_MODEL
    wf = _autoframed("fl2v", model="minimax_h3_fl2va_pruned_fp8_scaled.safetensors")
    assert wf[minimax._AF_UNET]["inputs"]["unet_name"] == \
        minimax._AUTOFRAME_DEFAULT_MODEL
    for ok in ("zit_v12.safetensors", "krea2_raw_fp8_scaled.safetensors"):
        wf = _autoframed("fl2v", model=ok)
        assert wf[minimax._AF_UNET]["inputs"]["unet_name"] == ok


def test_autoframe_family_follows_the_model_not_the_caller():
    # Pairing a Krea checkpoint with Z-Image's text encoder produces a graph
    # that loads and then generates noise, so the family is derived, never given.
    krea = _autoframed("fl2v", model="krea2_raw_fp8_scaled.safetensors")
    zimg = _autoframed("fl2v", model="zit_v12.safetensors")
    assert krea[minimax._AF_CLIP]["inputs"]["type"] == "krea2"
    assert zimg[minimax._AF_CLIP]["inputs"]["type"] == "lumina2"
    assert krea[minimax._AF_VAE]["inputs"]["vae_name"] != \
        zimg[minimax._AF_VAE]["inputs"]["vae_name"]
    # Z-Image clamps the CLIP layer; Krea does not have that node at all.
    assert minimax._AF_CLIPSET in zimg and minimax._AF_CLIPSET not in krea


def test_krea_autoframe_drops_turbo_for_a_prebaked_checkpoint():
    # RedCraft RedMix already has the distillation merged in; applying the turbo
    # LoRA again overcooks it, and the amateur slider's 1.5 then reads as grain.
    import krea
    assert krea._normalize_model(minimax._AUTOFRAME_DEFAULT_MODEL) \
        in krea.KREA_PREBAKED_TURBO
    wf = _autoframed("fl2v")
    baked = [n["inputs"] for i, n in sorted(wf.items())
             if n["class_type"] == "LoraLoaderModelOnly"]
    names = [b["lora_name"] for b in baked]
    assert not any("turbo" in n.lower() for n in names), names
    slider = [b for b in baked if "AmateurSlider" in b["lora_name"]][0]
    assert slider["strength_model"] == krea._PREBAKED_SLIDER_STRENGTH


def test_krea_autoframe_keeps_the_full_stack_on_a_raw_checkpoint():
    wf = _autoframed("fl2v", model="krea2_raw_fp8_scaled.safetensors")
    baked = [n["inputs"] for i, n in sorted(wf.items())
             if n["class_type"] == "LoraLoaderModelOnly"]
    names = [b["lora_name"] for b in baked]
    assert any("turbo" in n.lower() for n in names), names
    slider = [b for b in baked if "AmateurSlider" in b["lora_name"]][0]
    assert slider["strength_model"] == 1.5


def test_krea_autoframe_polishes_with_the_second_pass():
    # The image fleet's look is calibrated with a 2-step 0.2-denoise refiner.
    wf = _autoframed("fl2v")
    refine = wf[minimax._AF_REFINE]["inputs"]
    assert refine["denoise"] == 0.2
    assert refine["latent_image"] == [minimax._AF_KSAMPLER, 0]
    assert wf[minimax._AF_DECODE]["inputs"]["samples"] == [minimax._AF_REFINE, 0]
    # Z-Image is a single pass — no stray refiner node.
    assert minimax._AF_REFINE not in _autoframed("fl2v", model="zit_v12.safetensors")


def test_autoframe_does_not_hijack_the_viewers_prompt():
    # The extractor reads samplers in workflow order and shows the first one's
    # prompt as "Positive". The spliced KSampler must therefore land AFTER
    # MiniMax's guider, or the viewer captions every clip with the still's
    # prompt instead of the shot script — the exact "no prompt recorded" class
    # of bug the video-prompt extraction work fixed.
    import extract
    wf = minimax.maybe_build_minimax(
        {"prompt": "SHOT SCRIPT", "mode": "fl2v",
         "autoframe": {"prompt": "STILL PROMPT"}}, None)
    parsed = extract._parse_workflow(json.dumps(wf))
    prompts = extract._extract_prompts(parsed)
    assert prompts[0]["text"] == "SHOT SCRIPT", prompts
    # The still is still recorded, just second — it is genuinely useful to see
    # what painted frame 0.
    assert any(p["text"] == "STILL PROMPT" for p in prompts[1:]), prompts
    # And the job is still attributed to the video model, not Z-Image.
    assert "minimax" in extract._extract_model(parsed)


# --- turbo (step-distilled) LoRA -------------------------------------------

def test_turbo_is_off_unless_asked_for():
    wf = _build()
    assert not [n for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert wf["32"]["inputs"]["steps"] == 20


def test_turbo_rewires_BOTH_model_consumers():
    # The sampler is the obvious one; BasicScheduler is the one that bites.
    # It derives the sigma schedule FROM THE MODEL, and a distilled model's
    # schedule is the entire point — leaving it on the raw UNet runs 8 steps of
    # an undistilled curve, which is fast, wrong, and silent.
    wf = _build(turbo=True, mode="fl2v")
    lora, shift = minimax._N_TURBO_LORA, minimax._N_SIGMA_SHIFT
    assert wf[lora]["inputs"]["model"] == [minimax._N_UNET, 0]
    assert wf[shift]["inputs"]["model"] == [lora, 0]
    assert wf["32"]["inputs"]["model"] == [shift, 0], "BasicScheduler left on the raw UNet"
    assert wf["33"]["inputs"]["model"] == [shift, 0], "BasicGuider left on the raw UNet"


def test_turbo_sets_the_distilled_step_count():
    wf = _build(turbo=True, mode="fl2v")
    assert wf["32"]["inputs"]["steps"] == minimax._TURBO_BUILDS[("fl2v", 8)]["steps"] == 8


def test_a_bare_turbo_flag_is_still_the_8_step_build():
    # `turbo: true` is what every job queued while this was a checkbox carries,
    # and 8 steps is what it rendered. A step count is now a choice; the
    # absence of one is not a licence to change what those jobs mean.
    wf = _build(turbo=True, mode="fl2v")
    assert "8step" in wf[minimax._N_TURBO_LORA]["inputs"]["lora_name"]
    assert "fl2v" in wf[minimax._N_TURBO_LORA]["inputs"]["lora_name"]


def test_the_four_step_build_brings_its_own_shifts():
    """The mismatch this table exists to prevent.

    The 4-step 768p build is distilled at 6/3 video/audio shifts; the 8-step at
    12/3, which is also MiniMaxH3SigmaShift's default. Splice the 4-step LoRA
    and inherit the default and it renders — wrong, quietly, with nothing in
    the graph to say so.
    """
    wf = _build(turbo=4, mode="fl2v")
    assert "4step" in wf[minimax._N_TURBO_LORA]["inputs"]["lora_name"]
    assert "768p" in wf[minimax._N_TURBO_LORA]["inputs"]["lora_name"]
    assert wf["32"]["inputs"]["steps"] == 4
    shift = wf[minimax._N_SIGMA_SHIFT]["inputs"]
    assert (shift["shift_video"], shift["shift_audio"]) == (6.0, 3.0), shift
    _assert_fully_reachable(wf)


def test_the_eight_step_build_writes_its_shifts_explicitly():
    # They happen to equal the node's defaults. Writing them anyway is what
    # keeps the 4-step from ever inheriting them.
    wf = _build(turbo=8, mode="fl2v")
    shift = wf[minimax._N_SIGMA_SHIFT]["inputs"]
    assert (shift["shift_video"], shift["shift_audio"]) == (12.0, 3.0), shift


def test_the_step_count_can_be_written_out():
    # The panel sends what the operator picked; "4step" reads better in a log
    # than 4 does, and both have to mean the same build.
    for want in (4, "4", "4step", "4 step"):
        wf = _build(turbo=want, mode="fl2v")
        assert wf["32"]["inputs"]["steps"] == 4, want


def test_a_step_count_with_no_build_is_refused():
    # Not rounded to the nearest build, not silently ignored: there is no
    # 6-step distillation, and pretending otherwise is how you get a
    # twelve-minute render of the wrong schedule.
    assert _build(turbo=6, mode="fl2v") is None
    assert _build(turbo="fast", mode="fl2v") is None


def test_an_explicit_step_count_still_wins_over_turbo():
    # So 8 vs 6 can be A/B'd on the same LoRA.
    wf = _build(turbo=True, steps=6, mode="fl2v")
    assert wf["32"]["inputs"]["steps"] == 6
    assert wf["32"]["inputs"]["model"] == [minimax._N_SIGMA_SHIFT, 0]


def test_turbo_graph_is_still_fully_connected():
    _assert_fully_reachable(_build(turbo=True, mode="fl2v"))


def test_turbo_works_alongside_an_autoframed_first_frame():
    wf = minimax.maybe_build_minimax(
        {"prompt": "x", "mode": "fl2v", "turbo": True,
         "autoframe": {"prompt": "Photo. A woman."}}, None)
    _assert_fully_reachable(wf)
    # The video LoRA must not leak onto the frame-0 sampler, which is a
    # different architecture entirely.
    assert wf[minimax._AF_KSAMPLER]["inputs"]["model"] != [minimax._N_TURBO_LORA, 0]


# ---------------------------------------------------------------------------
# The sex LoRA. Decided from the prompt, because every caller would otherwise
# have to answer a question the prompt already answers.
# ---------------------------------------------------------------------------

_EXPLICIT = ("At 00:03.000, close-up: his cock pulled out of her mouth and her "
             "hand pumping it against her cheek.")
_TAME = ("At 00:03.000, close-up: her hand dragging down over her hip in an "
         "extreme facial close-up as she turns.")


def test_an_explicit_prompt_attaches_the_sex_lora():
    wf = _build(prompt=_EXPLICIT)
    node = wf[minimax._N_NSFW_LORA]
    assert node["inputs"]["lora_name"] == "HMNSFW_AIO_V2.safetensors", node
    # The author's own ceiling is 0.5; anything above it is a bug, not a taste.
    assert node["inputs"]["strength_model"] <= 0.5, node


def test_a_tame_prompt_does_not():
    wf = _build(prompt=_TAME)
    assert minimax._N_NSFW_LORA not in wf, "a clip with nothing in it got the sex LoRA"


def test_a_facial_close_up_is_a_framing_not_an_act():
    # "extreme facial close-up" is in the house camera vocabulary and appears in
    # clips with nothing sexual in them at all.
    assert minimax.nsfw_lora_strength({"prompt": "an extreme facial close-up of her face"}) == 0.0


def test_solo_insertion_counts_too():
    # "insertions" is one of the six acts it was trained on, and a solo clip has
    # none of the partner words that catch the other five.
    on = "she slides two fingers into her pussy while she keeps walking"
    assert minimax.nsfw_lora_strength({"prompt": on}) == 0.5, on
    # Level 8 is bare and touching herself with nothing inside her — the line
    # this has to draw.
    off = "she works herself over with a flat palm, breathing through her teeth"
    assert minimax.nsfw_lora_strength({"prompt": off}) == 0.0, off


def test_the_caller_can_veto_and_can_pin():
    assert minimax.nsfw_lora_strength({"prompt": _EXPLICIT, "nsfw_lora": False}) == 0.0
    assert minimax.nsfw_lora_strength({"prompt": _TAME, "nsfw_lora": True}) == 0.5
    assert minimax.nsfw_lora_strength({"prompt": _TAME, "nsfw_lora": 0.3}) == 0.3
    assert minimax.nsfw_lora_strength({"prompt": _TAME, "nsfw_lora": 9}) == 1.0


def test_the_sex_lora_graph_is_still_fully_connected():
    _assert_fully_reachable(_build(prompt=_EXPLICIT))


def test_it_chains_with_turbo_instead_of_dangling():
    wf = _build(prompt=_EXPLICIT, turbo=True, mode="fl2v")
    _assert_fully_reachable(wf)
    # UNet -> nsfw -> turbo -> sampler: both LoRAs actually in the path, and the
    # distilled schedule nearest the consumers.
    assert wf[minimax._N_NSFW_LORA]["inputs"]["model"] == [minimax._N_UNET, 0]
    assert wf[minimax._N_TURBO_LORA]["inputs"]["model"] == [minimax._N_NSFW_LORA, 0]
    assert wf[minimax._N_SIGMA_SHIFT]["inputs"]["model"] == [minimax._N_TURBO_LORA, 0]
    for nid in (minimax._N_SCHED, minimax._N_GUIDER):
        assert wf[nid]["inputs"]["model"] == [minimax._N_SIGMA_SHIFT, 0], nid
    assert wf[minimax._N_SCHED]["inputs"]["steps"] == 8


def test_it_does_not_leak_onto_the_first_frame_sampler():
    wf = minimax.maybe_build_minimax(
        {"prompt": _EXPLICIT, "mode": "fl2v",
         "autoframe": {"prompt": "Photo. A woman."}}, None)
    _assert_fully_reachable(wf)
    assert wf[minimax._AF_KSAMPLER]["inputs"]["model"] != [minimax._N_NSFW_LORA, 0]


# ---------------------------------------------------------------------------
# The finish LoRA, and the trap the register switch set for the detector.
# ---------------------------------------------------------------------------

_CLINICAL = ("hmmotion, blowjob, pov, fast, close-up. Her hand flattens the shaft against "
             "her lips, the glans darker pink than the taut shaft, the corona ridge dragging.")
_FINISH = ("hmmotion, handjob, pov, fast, close-up. CUMSH0T. Continuous short pulses of long "
           "ropes of white semen across her cheekbone, the last running off her chin.")
# The trigger with a zero in it, and nothing else a detector could catch on: the
# writer emits it for the finish LoRA and for nothing else.
_TRIGGER_ONLY = "CUMSH0T."


def test_the_clinical_register_still_attaches_the_sex_lora():
    # The whole point of the register switch is that the writer stops saying
    # "cock". If the detector only knew the blunt words, the switch would have
    # turned the LoRA OFF for exactly the clips it exists for.
    assert "cock" not in _CLINICAL.lower()
    assert minimax.nsfw_lora_strength({"prompt": _CLINICAL}) == 0.5, _CLINICAL


def test_the_trigger_token_alone_is_enough():
    assert minimax.nsfw_lora_strength({"prompt": "hmmotion, side, slow, wide shot."}) == 0.5


def test_the_sex_lora_runs_at_its_authors_ceiling():
    # "Use it at strength 0.5 or below." We sat at 0.35 under that until the
    # finish LoRA landed at 1.0 on top and outweighed it three to one.
    wf = _build(prompt=_CLINICAL)
    assert wf[minimax._N_NSFW_LORA]["inputs"]["strength_model"] == 0.5
    assert minimax._NSFW_STRENGTH <= 0.5, "0.5 is the author's stated ceiling"


def test_a_finish_attaches_the_cumshot_lora_too():
    wf = _build(prompt=_FINISH)
    node = wf[minimax._N_CUMSHOT_LORA]
    assert node["inputs"]["lora_name"] == "epic_cumshots-MiniMaxH3-ALPHA-CUMSH0T.safetensors", node
    # NOT its author's 1.0. At full strength it stacks on the anatomy LoRA for
    # the whole clip and the genitalia came back malformed and sometimes
    # doubled; a controlled A/B on one prompt put 0.8 ahead of 1.0 and the
    # operator settled on 0.75. His warning about rolling back to a paint-bucket
    # finish under 1.0 did not hold here.
    assert node["inputs"]["strength_model"] == minimax._CUMSHOT_STRENGTH == 0.75, node
    _assert_fully_reachable(wf)


def test_the_finish_trigger_alone_attaches_both():
    # The zero is the trap: `cumsh0t` does not match a detector written for the
    # word cumshot, and a clip carrying only the trigger would have rendered
    # with neither LoRA on it.
    assert minimax.cumshot_lora_strength({"prompt": _TRIGGER_ONLY}) == minimax._CUMSHOT_STRENGTH
    assert minimax.nsfw_lora_strength({"prompt": _TRIGGER_ONLY}) == 0.5


def test_semen_counts_as_a_finish():
    # The new LoRA was captioned in that word, so the writer now uses it.
    assert minimax.cumshot_lora_strength(
        {"prompt": "long ropes of white semen across her mouth"}) == minimax._CUMSHOT_STRENGTH


def test_the_house_number_is_not_the_author_s_and_a_caller_can_still_pin_either():
    # 0.75 is ours, measured. His 1.0 is still one option away, and so is off.
    assert minimax._CUMSHOT_STRENGTH == 0.75
    assert minimax.cumshot_lora_strength({"prompt": _FINISH, "cumshot_lora": 1.0}) == 1.0
    assert minimax.cumshot_lora_strength({"prompt": _FINISH, "cumshot_lora": False}) == 0.0
    wf = _build(prompt=_FINISH, cumshot_lora=0.9)
    assert wf[minimax._N_CUMSHOT_LORA]["inputs"]["strength_model"] == 0.9


def test_a_clip_with_no_finish_does_not_get_it():
    wf = _build(prompt=_CLINICAL)
    assert minimax._N_CUMSHOT_LORA not in wf


def test_all_three_chain_in_order():
    wf = _build(prompt=_FINISH, turbo=True, mode="fl2v")
    _assert_fully_reachable(wf)
    assert wf[minimax._N_NSFW_LORA]["inputs"]["model"] == [minimax._N_UNET, 0]
    assert wf[minimax._N_CUMSHOT_LORA]["inputs"]["model"] == [minimax._N_NSFW_LORA, 0]
    assert wf[minimax._N_TURBO_LORA]["inputs"]["model"] == [minimax._N_CUMSHOT_LORA, 0]
    # ...and the shift last of all, because it IS the sigma curve the scheduler
    # reads off the model.
    assert wf[minimax._N_SIGMA_SHIFT]["inputs"]["model"] == [minimax._N_TURBO_LORA, 0]
    for nid in (minimax._N_SCHED, minimax._N_GUIDER):
        assert wf[nid]["inputs"]["model"] == [minimax._N_SIGMA_SHIFT, 0], nid


def test_turbo_runs_at_its_own_strength_either_way():
    # HMCumshot's author ran the 8-step turbo at 0.20 alongside his LoRA, and we
    # followed him. Epic Cumshots says nothing about turbo, so there is no
    # number to justify overriding the author's own 1.0 — which is what every
    # one of lightx2v's published graphs sets it to.
    hot = _build(prompt=_FINISH, turbo=True, mode="fl2v")
    assert hot[minimax._N_TURBO_LORA]["inputs"]["strength_model"] == 1.0
    plain = _build(prompt=_CLINICAL, turbo=True, mode="fl2v")
    assert plain[minimax._N_TURBO_LORA]["inputs"]["strength_model"] == 1.0


def test_the_caller_can_veto_the_finish_lora():
    assert minimax.cumshot_lora_strength({"prompt": _FINISH, "cumshot_lora": False}) == 0.0
    assert minimax.cumshot_lora_strength({"prompt": _CLINICAL, "cumshot_lora": 0.5}) == 0.5


def test_weakening_the_distillation_without_steps_is_refused():
    """The pairing that cost four renders: 8 steps of a 20%-distilled model.

    It rendered, it took twelve minutes, and it came back with malformed faces
    and bodies. Nothing in the graph said anything was wrong, because nothing
    was checking.
    """
    wf = _build(prompt=_FINISH, turbo=False)
    build = minimax._TURBO_BUILDS[("fl2v", 8)]
    try:
        minimax.splice_turbo_lora(wf, build, 0.2)
    except ValueError as e:
        assert "steps" in str(e), e
    else:
        raise AssertionError("turbo at 0.2 on an 8-step schedule was allowed")


def test_lowering_the_strength_is_fine_when_the_steps_come_with_it():
    wf = _build(prompt=_FINISH, turbo=False)
    minimax.splice_turbo_lora(wf, minimax._TURBO_BUILDS[("fl2v", 8)], 0.2, steps=24)
    assert wf[minimax._N_TURBO_LORA]["inputs"]["strength_model"] == 0.2
    assert wf[minimax._N_SCHED]["inputs"]["steps"] == 24
    _assert_fully_reachable(wf)


def test_the_default_pairing_is_the_distilled_one():
    build = minimax._TURBO_BUILDS[("fl2v", 8)]
    wf = _build(prompt=_FINISH, turbo=True, mode="fl2v")
    assert wf[minimax._N_TURBO_LORA]["inputs"]["strength_model"] == build["strength"]
    assert wf[minimax._N_SCHED]["inputs"]["steps"] == build["steps"]


def test_the_fl2va_lora_is_never_spliced_onto_a_ref2va_graph():
    """The FL2VA LoRA is published for FL2VA and this build ships ref2va too.

    Same file size, matching keys: it would load clean and render wrong, which
    is the failure this module has now had twice. Ref mode is distilled by its
    OWN build, not by making an exception for that one.
    """
    for want in (True, 4):
        wf = _build(prompt=_FINISH, turbo=want, mode="ref")
        name = wf[minimax._N_TURBO_LORA]["inputs"]["lora_name"]
        assert "ref2v" in name, name
        assert "fl2v" not in name, name


def test_ref_mode_gets_the_ref2va_distillation():
    wf = _build(prompt=_FINISH, turbo=True, mode="ref")
    assert wf[minimax._N_SCHED]["inputs"]["steps"] == 4
    shift = wf[minimax._N_SIGMA_SHIFT]["inputs"]
    assert (shift["shift_video"], shift["shift_audio"]) == (12.0, 3.0), shift
    _assert_fully_reachable(wf)


def test_there_is_no_eight_step_ref_build_and_asking_says_so():
    # An explicit 8 in ref mode is a specific request for weights that do not
    # exist. It fails at submit rather than quietly rendering the 4-step.
    assert _build(prompt=_FINISH, turbo=8, mode="ref") is None
    try:
        minimax.turbo_build({"turbo": 8}, "ref")
    except ValueError as e:
        assert "8-step" in str(e) and "[4]" in str(e), e
    else:
        raise AssertionError("an 8-step ref build was invented")


def test_every_build_carries_weights_for_its_own_task():
    # The whole table in one line: a build whose file does not name its mode is
    # the mismatch that has cost this module two rounds of bad renders.
    for (mode, steps), build in minimax._TURBO_BUILDS.items():
        tag = "fl2v" if mode == "fl2v" else "ref2v"
        assert tag in build["lora"], (mode, build["lora"])
        assert f"{steps}step" in build["lora"], (steps, build["lora"])


def test_turbo_still_applies_in_fl2v():
    build = minimax._TURBO_BUILDS[("fl2v", 8)]
    wf = _build(prompt=_FINISH, turbo=True, mode="fl2v")
    assert wf[minimax._N_TURBO_LORA]["inputs"]["strength_model"] == build["strength"]
    assert wf[minimax._N_SCHED]["inputs"]["steps"] == build["steps"]


def test_a_ref_job_without_turbo_is_unchanged():
    wf = _build(prompt=_FINISH, mode="ref")
    assert wf[minimax._N_SCHED]["inputs"]["steps"] == 20
