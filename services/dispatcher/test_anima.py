"""Tests for Anima detection / extraction / substitution.

Includes a real submitted workflow fixture (testdata/anima_submitted_real.json,
a captured `animaCatTower_v10` job) because real graphs wire prompt text through
provider nodes (CLIPTextEncode.text <- "String Literal" node), which synthetic
fixtures tend to miss.
"""
import json
import os

import anima

HERE = os.path.dirname(__file__)
REAL_FIXTURE = os.path.join(HERE, "testdata", "anima_submitted_real.json")


def _real() -> dict:
    with open(REAL_FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


# --- detection -------------------------------------------------------------

def test_detect_checkpoint_loader_anima():
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "animaCatTower_v10.safetensors"}}}
    assert anima.detect_anima_model(wf) == "animaCatTower_v10.safetensors"


def test_detect_unet_loader_anima():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima_baseV10.safetensors"}}}
    assert anima.detect_anima_model(wf) == "anima_baseV10.safetensors"


def test_detect_provider_wrapped_loader():
    # Image-Saver's "Checkpoint Loader with Name" still uses ckpt_name.
    wf = {"9": {"class_type": "Checkpoint Loader with Name (Image Saver)",
                "inputs": {"ckpt_name": "copycatAnima_20260519.safetensors"}}}
    assert anima.detect_anima_model(wf) == "copycatAnima_20260519.safetensors"


def test_anime_named_sdxl_is_NOT_anima():
    # The whole point of the allowlist: novaAnimeXL (Illustrious/SDXL) must pass through.
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "novaAnimeXL_ilV190.safetensors"}}}
    assert anima.detect_anima_model(wf) is None


def test_detect_on_real_fixture():
    assert anima.detect_anima_model(_real()) == "animaCatTower_v10.safetensors"


# --- extraction ------------------------------------------------------------

def test_extract_traces_sampler_through_provider_text():
    # sampler.positive -> CLIPTextEncode.text -> String Literal.string
    wf = {
        "5": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["8", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ["9", 0]}},
        "8": {"class_type": "String Literal", "inputs": {"string": "a tall woman, masterpiece"}},
        "9": {"class_type": "String Literal", "inputs": {"string": "worst quality, blurry"}},
    }
    pos, neg = anima.extract_prompts(wf)
    assert pos == "a tall woman, masterpiece"
    assert neg == "worst quality, blurry"


def test_extract_inline_clip_text():
    wf = {
        "5": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, solo"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres"}},
    }
    assert anima.extract_prompts(wf) == ("1girl, solo", "lowres")


def test_extract_prefers_longest_when_multiple():
    wf = {
        "5": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},  # empty detailer prompt
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad anatomy"}},
        "55": {"class_type": "KSampler", "inputs": {"positive": ["56", 0], "negative": ["57", 0]}},
        "56": {"class_type": "CLIPTextEncode", "inputs": {"text": "the real, much longer positive prompt"}},
        "57": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad anatomy"}},
    }
    pos, neg = anima.extract_prompts(wf)
    assert pos == "the real, much longer positive prompt"
    assert neg == "bad anatomy"


def test_extract_on_real_fixture():
    pos, neg = anima.extract_prompts(_real())
    assert pos.startswith("bianca_fontaine"), pos[:40]
    assert neg.startswith("score_6"), neg[:40]


# --- build / injection -----------------------------------------------------

def test_build_injects_model_and_prompts():
    wf = anima.build_anima_workflow("animaCatTower_v10.safetensors", "POS TEXT", "NEG TEXT")
    unet = [n for n in wf.values() if n.get("class_type") == "UNETLoader"]
    assert unet and unet[0]["inputs"]["unet_name"] == "animaCatTower_v10.safetensors"
    pos = anima._node_by_title(wf, "POSITIVE")
    neg = anima._node_by_title(wf, "NEGATIVE")
    assert pos["inputs"]["wildcard_text"] == "POS TEXT"
    assert pos["inputs"]["populated_text"] == "POS TEXT"
    assert neg["inputs"]["wildcard_text"] == "NEG TEXT"
    # mode pinned to 'fixed' so the node uses our text verbatim (no re-roll).
    assert pos["inputs"]["mode"] == "fixed"
    assert neg["inputs"]["mode"] == "fixed"


def test_build_overwrites_sample_prompt_even_when_empty():
    # The template ships an authored sample prompt; an empty extraction must NOT
    # leak it into the user's job.
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "", "")
    pos = anima._node_by_title(wf, "POSITIVE")
    assert pos["inputs"]["wildcard_text"] == ""
    assert "masterpiece" not in pos["inputs"]["populated_text"]


def test_build_removes_widget_to_string():
    # WidgetToString reads extra_pnginfo (absent in headless submissions) and would
    # crash before sampling; it must be gone, with its model-name output inlined.
    wf = anima.build_anima_workflow("anima_preview3Base.safetensors", "p", "n")
    assert not any(n.get("class_type") == "WidgetToString" for n in wf.values())
    # the metadata node that consumed it now carries the literal model name
    meta = [n for n in wf.values() if n.get("class_type") == "Image Saver Metadata"]
    assert meta and meta[0]["inputs"]["modelname"] == "anima_preview3Base.safetensors"
    # no dangling references to the removed node remain
    for n in wf.values():
        for v in (n.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                assert str(v[0]) in wf, f"dangling ref to {v[0]}"


def test_build_injects_valid_seed():
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n")
    seed_nodes = [n for n in wf.values() if "Seed" in n.get("class_type", "") and isinstance(n.get("inputs", {}).get("seed"), int)]
    assert seed_nodes, "expected a literal-seed node in the template"
    for n in seed_nodes:
        s = n["inputs"]["seed"]
        assert 0 <= s < 2 ** 50 and s != -1


def test_build_template_is_complete_api_format():
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n")
    # Every node must have a class_type (no broken/unserialized nodes).
    assert all(isinstance(n, dict) and n.get("class_type") for n in wf.values())
    # Qwen architecture wiring is present.
    cts = {n["class_type"] for n in wf.values()}
    assert {"UNETLoader", "CLIPLoader", "VAELoader"} <= cts


def test_build_does_not_mutate_cached_template():
    a = anima.build_anima_workflow("anima_baseV10.safetensors", "x", "y")
    b = anima.build_anima_workflow("anima_preview3Base.safetensors", "z", "w")
    a_unet = [n for n in a.values() if n.get("class_type") == "UNETLoader"][0]
    assert a_unet["inputs"]["unet_name"] == "anima_baseV10.safetensors"  # not clobbered by b


# --- end to end ------------------------------------------------------------

def test_rewrite_on_real_fixture():
    out = anima.maybe_rewrite_to_anima(_real())
    assert out is not None
    unet = [n for n in out.values() if n.get("class_type") == "UNETLoader"][0]
    assert unet["inputs"]["unet_name"] == "animaCatTower_v10.safetensors"
    assert anima._node_by_title(out, "POSITIVE")["inputs"]["wildcard_text"].startswith("bianca_fontaine")


def test_rewrite_passthrough_for_non_anima():
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "novaAnimeXL_ilV190.safetensors"}}}
    assert anima.maybe_rewrite_to_anima(wf) is None


# --- detailer / upscale reduction -----------------------------------------
#
# Node ids reference AnimaStandardFull.api.json. The detailer triples are:
#   hand: FDP 27 / EDP 14 / det 9
#   nsfw: FDP 28 / EDP 15 / det 10
#   face: FDP 29 / EDP 16 / det 11
#   eyes: FDP 30 / EDP 17 / det 12
# Image chain (when all present): VAEDecode 43 -> 27 -> 28 -> 29 -> 30 ->
# hiresFix 60 -> Image Saver Simple 13.

_DET_NODES = {
    "hand": ("27", "14", ["9"]),
    "nsfw": ("28", "15", ["10"]),
    "face": ("29", "16", ["11"]),
    "eyes": ("30", "17", ["12"]),
}
_ALL_ON = {"detailers": {"hand": True, "nsfw": True, "face": True, "eyes": True}}


def _build(options=None):
    return anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n", options)


def _assert_no_dangling(wf):
    """Every [id, slot] input link must point to a node still in the graph."""
    for nid, node in wf.items():
        for key, val in (node.get("inputs") or {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], (str, int)):
                # second element is the slot index (int); first is the node id.
                assert str(val[0]) in wf, f"dangling ref {val} on node {nid}.{key}"


def _image_src(node):
    inp = node.get("inputs") or {}
    return inp.get("image") if "image" in inp else inp.get("images")


def _assert_image_chain_intact(wf):
    """Walk the linear image chain from the saver's images input back through every
    FaceDetailerPipe / hiresFix to VAEDecode 43 — each hop must reference a node
    that exists, and the chain must terminate at VAEDecode 43 (the image root)."""
    saver = wf["13"]
    src = saver["inputs"]["images"]
    assert isinstance(src, list), "saver.images must be a link"
    seen = set()
    cur = str(src[0])
    while True:
        assert cur in wf, f"image chain references missing node {cur}"
        assert cur not in seen, "image chain has a cycle"
        seen.add(cur)
        node = wf[cur]
        if node.get("class_type") == "VAEDecode":
            return  # reached the image root
        nxt = _image_src(node)
        assert isinstance(nxt, list), f"node {cur} has no image source — chain broken"
        cur = str(nxt[0])


def _assert_pipeline_path(wf):
    """UNETLoader -> ... -> KSampler -> VAEDecode -> ... -> an Image Saver exists.
    Asserts the structural backbone survives any reduction: a UNETLoader, a
    sampler reading a model, a VAEDecode reading the sampler, and an image saver."""
    cts = {n.get("class_type") for n in wf.values()}
    assert "UNETLoader" in cts, "UNETLoader missing"
    # KSampler reads a model (transitively from the UNETLoader) and feeds VAEDecode.
    samplers = [nid for nid, n in wf.items() if n.get("class_type") == "KSampler"]
    assert samplers, "KSampler missing"
    vaedecodes = [nid for nid, n in wf.items() if n.get("class_type") == "VAEDecode"]
    assert vaedecodes, "VAEDecode missing"
    # VAEDecode reads its latent from a sampler.
    vd = wf[vaedecodes[0]]
    samples = (vd.get("inputs") or {}).get("samples")
    assert isinstance(samples, list) and str(samples[0]) in samplers, "VAEDecode not fed by sampler"
    # An image saver exists and has a valid image input.
    savers = [n for n in wf.values() if "Image Saver" in str(n.get("class_type", ""))]
    assert savers, "Image Saver missing"
    assert any("Simple" in str(n.get("class_type", "")) for n in savers)


def _assert_invariants(wf):
    _assert_no_dangling(wf)
    _assert_image_chain_intact(wf)
    _assert_pipeline_path(wf)


def test_default_keeps_three_detailers_and_upscale():
    # No options reproduces today's AnimaStandardDetailer: hand/face/eyes on,
    # nsfw off, upscale on @2.0x (percent 50).
    wf = _build(None)
    for name in ("hand", "face", "eyes"):
        fdp, edp, dets = _DET_NODES[name]
        assert fdp in wf and edp in wf and all(d in wf for d in dets), f"{name} dropped"
    nfdp, nedp, ndet = _DET_NODES["nsfw"]
    assert nfdp not in wf and nedp not in wf and ndet[0] not in wf, "nsfw should default off"
    assert "60" in wf and wf["60"]["inputs"]["percent"] == 50.0
    # chain heals: VAEDecode 43 -> hand 27 -> face 29 -> eyes 30 -> 60 -> 13
    assert wf["27"]["inputs"]["image"] == ["43", 0]
    assert wf["29"]["inputs"]["image"] == ["27", 0]
    assert wf["30"]["inputs"]["image"] == ["29", 0]
    assert wf["60"]["inputs"]["image"] == ["30", 0]
    assert wf["13"]["inputs"]["images"] == ["60", 1]
    _assert_invariants(wf)


def test_remove_first_detailer_hand():
    # All-on except hand removed: VAEDecode 43 should feed nsfw 28 directly.
    opts = {"detailers": {"hand": False, "nsfw": True, "face": True, "eyes": True}}
    wf = _build(opts)
    assert "27" not in wf and "14" not in wf and "9" not in wf
    assert wf["28"]["inputs"]["image"] == ["43", 0]
    _assert_invariants(wf)


def test_remove_middle_detailer_face():
    # face removed (all else on): chain heals nsfw 28 -> eyes 30.
    opts = {"detailers": {"hand": True, "nsfw": True, "face": False, "eyes": True}}
    wf = _build(opts)
    assert "29" not in wf and "16" not in wf and "11" not in wf
    assert wf["30"]["inputs"]["image"] == ["28", 0]
    _assert_invariants(wf)


def test_remove_last_detailer_eyes():
    # eyes removed (all else on): hiresFix 60 should read face 29's output.
    opts = {"detailers": {"hand": True, "nsfw": True, "face": True, "eyes": False}}
    wf = _build(opts)
    assert "30" not in wf and "17" not in wf and "12" not in wf
    assert wf["60"]["inputs"]["image"] == ["29", 0]
    _assert_invariants(wf)


def test_each_detailer_individually_removable():
    # Remove exactly one detailer at a time (others all on); the graph stays valid
    # and the removed triple is gone.
    for name in ("hand", "nsfw", "face", "eyes"):
        dets = {"hand": True, "nsfw": True, "face": True, "eyes": True}
        dets[name] = False
        wf = _build({"detailers": dets})
        fdp, edp, det = _DET_NODES[name]
        assert fdp not in wf, f"{name} FDP {fdp} not removed"
        assert edp not in wf, f"{name} EDP {edp} not removed"
        assert all(d not in wf for d in det), f"{name} detector not removed"
        # ToDetailerPipe 19 survives — three detailers still consume it.
        assert "19" in wf
        _assert_invariants(wf)


def test_nsfw_on_keeps_nsfw_triple():
    # nsfw is the only detailer ON by request (not default) — its triple stays
    # and the chain runs hand 27 -> nsfw 28 -> face 29 -> eyes 30.
    wf = _build(_ALL_ON)
    for name in ("hand", "nsfw", "face", "eyes"):
        fdp, edp, dets = _DET_NODES[name]
        assert fdp in wf and edp in wf and all(d in wf for d in dets), f"{name} missing"
    assert wf["28"]["inputs"]["image"] == ["27", 0]
    assert wf["29"]["inputs"]["image"] == ["28", 0]
    assert wf["30"]["inputs"]["image"] == ["29", 0]
    _assert_invariants(wf)


def test_all_detailers_off_collapses_to_vaedecode():
    # Every detailer off: hiresFix 60 reads VAEDecode 43 directly; all FDPs/EDPs/
    # detectors AND the orphaned base pipe (19/7/8) are GC'd.
    opts = {"detailers": {"hand": False, "nsfw": False, "face": False, "eyes": False}}
    wf = _build(opts)
    for fdp, edp, dets in _DET_NODES.values():
        assert fdp not in wf and edp not in wf and all(d not in wf for d in dets)
    for orphan in ("19", "7", "8"):
        assert orphan not in wf, f"orphaned base-pipe node {orphan} not GC'd"
    assert "60" in wf
    assert wf["60"]["inputs"]["image"] == ["43", 0]
    _assert_invariants(wf)


def test_upscale_off_rewires_saver_to_last_detailer():
    # Upscale off (detailers default): node 60 gone, saver reads eyes 30's output.
    wf = _build({"upscale": False})
    assert "60" not in wf
    assert wf["13"]["inputs"]["images"] == ["30", 0]
    _assert_invariants(wf)


def test_all_off_both_collapses_to_vaedecode_then_saver():
    # All detailers off AND upscale off: saver 13 reads VAEDecode 43 directly.
    opts = {"detailers": {"hand": False, "nsfw": False, "face": False, "eyes": False}, "upscale": False}
    wf = _build(opts)
    assert "60" not in wf
    for fdp, edp, dets in _DET_NODES.values():
        assert fdp not in wf and edp not in wf and all(d not in wf for d in dets)
    for orphan in ("19", "7", "8"):
        assert orphan not in wf
    assert wf["13"]["inputs"]["images"] == ["43", 0]
    _assert_invariants(wf)


def test_upscale_off_with_all_detailers_off_saver_to_vaedecode():
    # Same as above but expressed via the per-detailer toggles only; the saver must
    # still land on VAEDecode 43.
    opts = {"detailers": {"hand": False, "nsfw": False, "face": False, "eyes": False}, "upscale": False}
    wf = _build(opts)
    src = wf["13"]["inputs"]["images"]
    assert src == ["43", 0]
    assert wf[str(src[0])]["class_type"] == "VAEDecode"
    _assert_invariants(wf)


def test_upscale_factor_change():
    # factor encodes as percent = factor * 25.
    for factor, expected in ((1.0, 25.0), (1.5, 37.5), (2.0, 50.0), (3.0, 75.0), (4.0, 100.0)):
        wf = _build({"upscale": True, "upscale_factor": factor})
        assert "60" in wf
        assert wf["60"]["inputs"]["percent"] == expected, (factor, wf["60"]["inputs"]["percent"])
        # model_name stays the fixed 4x model; rescale mode untouched.
        assert wf["60"]["inputs"]["model_name"] == "4x_foolhardy_Remacri.pth"
        assert wf["60"]["inputs"]["rescale"] == "by percentage"
        _assert_invariants(wf)


def test_upscale_factor_clamped():
    # Out-of-range factors clamp to [1.0, 4.0] (percent in [25, 100]); never raise.
    assert _build({"upscale": True, "upscale_factor": 99})["60"]["inputs"]["percent"] == 100.0
    assert _build({"upscale": True, "upscale_factor": 0.1})["60"]["inputs"]["percent"] == 25.0


def test_invalid_options_fall_back_to_defaults():
    # Malformed values must be ignored (never raise) and fall back to per-key
    # defaults: hand/face/eyes on, nsfw off, upscale on @2.0x.
    for bad in (
        {"detailers": "not a dict"},
        {"detailers": {"hand": "yes", "nsfw": 1}},
        {"upscale": "true"},
        {"upscale_factor": "big"},
        {"upscale_factor": -5},
        {"upscale_factor": 0},
        {"upscale_factor": True},
        {"unknown_key": 123},
        "not a dict at all",
        None,
    ):
        wf = _build(bad)
        # default detailer set: hand/face/eyes on, nsfw off
        assert "27" in wf and "29" in wf and "30" in wf, bad
        assert "28" not in wf, bad
        # upscale default on @2.0x
        assert "60" in wf and wf["60"]["inputs"]["percent"] == 50.0, bad
        _assert_invariants(wf)


def test_partial_detailers_merge_over_defaults():
    # Only one sub-key given; the rest fall back to per-detailer defaults
    # (face/hand/eyes on, nsfw off). Here we turn hand off and say nothing else —
    # face/eyes stay on, nsfw stays off.
    wf = _build({"detailers": {"hand": False}})
    assert "27" not in wf, "hand should be removed"
    assert "29" in wf and "30" in wf, "face/eyes should stay (default on)"
    assert "28" not in wf, "nsfw should stay off (default)"
    _assert_invariants(wf)


def test_partial_key_missing_uses_default_upscale():
    # detailers given but upscale/factor omitted -> upscale on @2.0x.
    wf = _build({"detailers": {"nsfw": True}})
    assert "28" in wf, "nsfw on by request"
    assert "60" in wf and wf["60"]["inputs"]["percent"] == 50.0
    _assert_invariants(wf)


def test_reduction_preserves_model_and_prompts():
    # The reduction runs AFTER injection — model + prompts must survive on the
    # nodes that remain, even with a heavily reduced graph.
    opts = {"detailers": {"hand": False, "nsfw": False, "face": False, "eyes": False}, "upscale": False}
    wf = anima.build_anima_workflow("animaCatTower_v10.safetensors", "POS", "NEG", opts)
    unet = [n for n in wf.values() if n.get("class_type") == "UNETLoader"]
    assert unet and unet[0]["inputs"]["unet_name"] == "animaCatTower_v10.safetensors"
    assert anima._node_by_title(wf, "POSITIVE")["inputs"]["wildcard_text"] == "POS"
    assert anima._node_by_title(wf, "NEGATIVE")["inputs"]["wildcard_text"] == "NEG"
    # no WidgetToString survived (it was neutralized before reduction).
    assert not any(n.get("class_type") == "WidgetToString" for n in wf.values())
    _assert_invariants(wf)


def test_maybe_rewrite_threads_options():
    # End-to-end: options passed through maybe_rewrite_to_anima reach the reduction.
    opts = {"detailers": {"hand": True, "nsfw": True, "face": True, "eyes": True}, "upscale_factor": 3.0}
    out = anima.maybe_rewrite_to_anima(_real(), opts)
    assert out is not None
    assert "28" in out, "nsfw should be kept when requested"
    assert out["60"]["inputs"]["percent"] == 75.0
    _assert_invariants(out)


def test_seed_injected_into_surviving_detailers():
    # Seed injection ran before reduction; surviving FaceDetailerPipes carry a
    # concrete (non -1) seed where the template had a literal one. (In the Full
    # template the detailer seeds are links to the rgthree node, so we assert the
    # rgthree Seed node itself got a valid seed.)
    wf = _build(None)
    seed_nodes = [n for n in wf.values()
                  if "Seed" in n.get("class_type", "") and isinstance(n.get("inputs", {}).get("seed"), int)]
    assert seed_nodes
    for n in seed_nodes:
        s = n["inputs"]["seed"]
        assert 0 <= s < 2 ** 50 and s != -1


# --- LoRA stack injection --------------------------------------------------

def test_loras_absent_is_noop():
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n", {})
    assert not any(n.get("class_type") == "LoraLoader" for n in wf.values())


def test_loras_chain_wiring():
    opts = {"loras": [{"name": "a.safetensors", "strength": 0.8},
                      {"name": "b", "strength": 1.2}]}
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n", opts)
    l1, l2 = wf["9001"], wf["9002"]
    assert l1["class_type"] == "LoraLoader" and l1["inputs"]["lora_name"] == "a.safetensors"
    assert l1["inputs"]["model"] == ["1", 0] and l1["inputs"]["clip"] == ["18", 0]
    assert l1["inputs"]["strength_model"] == 0.8
    # bare name gets the extension appended
    assert l2["inputs"]["lora_name"] == "b.safetensors"
    assert l2["inputs"]["model"] == ["9001", 0] and l2["inputs"]["clip"] == ["9001", 1]
    # the LoraManager passthrough now feeds from the chain end
    assert wf["5"]["inputs"]["model"] == ["9002", 0]
    assert wf["5"]["inputs"]["clip"] == ["9002", 1]


def test_loras_malformed_entries_dropped():
    opts = {"loras": ["junk", {"strength": 1}, {"name": ""},
                      {"name": "ok.safetensors", "strength": "high"},
                      {"name": "clamp.safetensors", "strength": 99}]}
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n", opts)
    chain = [n for n in wf.values() if n.get("class_type") == "LoraLoader"]
    assert len(chain) == 2
    assert chain[0]["inputs"]["strength_model"] == 1.0      # bad strength -> default
    assert chain[1]["inputs"]["strength_model"] == 4.0      # clamped


def test_loras_capped_at_max():
    opts = {"loras": [{"name": f"l{i}.safetensors"} for i in range(20)]}
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n", opts)
    assert sum(1 for n in wf.values() if n.get("class_type") == "LoraLoader") == 8


def test_loras_survive_full_reduction():
    # The chain feeds the KSampler (via node 5), which survives every reduction —
    # even all-detailers-off + no-upscale must keep the loras and stay dangling-free.
    opts = {"loras": [{"name": "x.safetensors", "strength": 1.0}],
            "detailers": {"face": False, "hand": False, "eyes": False, "nsfw": False},
            "upscale": False}
    wf = anima.build_anima_workflow("anima_baseV10.safetensors", "p", "n", opts)
    assert any(n.get("class_type") == "LoraLoader" for n in wf.values())
    for n in wf.values():
        for v in (n.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                assert str(v[0]) in wf, f"dangling ref to {v[0]}"


def _params_node(wf):
    return next(n["inputs"] for n in wf.values()
               if n.get("class_type") == "Input Parameters (Image Saver)")


def test_per_model_sampler_override_applies_to_listed_model():
    wf = anima.build_anima_workflow("miaomiaoharem_anima11.safetensors", "p", "n", None)
    p = _params_node(wf)
    assert p["cfg"] == 5.5 and p["sampler"] == "euler" and p["scheduler"] == "normal"
    assert p["steps"] == 30


def test_per_model_sampler_noop_for_unlisted_model():
    # other Anima models keep the template defaults (er_sde / simple / cfg 4)
    wf = anima.build_anima_workflow("anima_basev10.safetensors", "p", "n", None)
    p = _params_node(wf)
    assert p["cfg"] == 4 and p["sampler"] == "er_sde" and p["scheduler"] == "simple"


def test_per_model_sampler_copycatanima():
    wf = anima.build_anima_workflow("copycatanima_20260519.safetensors", "p", "n", None)
    p = _params_node(wf)
    assert p["steps"] == 32 and p["cfg"] == 4 and p["sampler"] == "er_sde"
    # scheduler untouched (template default 'simple')
    assert p["scheduler"] == "simple"
