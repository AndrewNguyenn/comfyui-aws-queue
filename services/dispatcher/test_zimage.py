"""Tests for Z-Image detection / substitution / VLM-input handling."""
import copy
import json
import os

import zimage

HERE = os.path.dirname(__file__)
TEMPLATE = os.path.join(HERE, "zimage_templates", "ZImageStandardFull.api.json")


def _catpony_like(model: str) -> dict:
    """A minimal submitted graph shaped like the frontend's catpony submission:
    a CheckpointLoaderSimple with the chosen model + a sampler whose positive/
    negative trace back to String Literal nodes (real graphs wire prompts so)."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "5": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["8", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ["9", 0]}},
        "8": {"class_type": "String Literal", "inputs": {"string": "a tall woman on a beach, masterpiece"}},
        "9": {"class_type": "String Literal", "inputs": {"string": "worst quality, blurry"}},
    }


# --- detection -------------------------------------------------------------

def test_detect_community_zimage_checkpoint():
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "moodyProMix_zitV13.safetensors"}}}
    assert zimage.detect_zimage_model(wf) == "moodyProMix_zitV13.safetensors"


def test_detect_via_unet_loader():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "divingZImageTurbo_v60Fp16.safetensors"}}}
    assert zimage.detect_zimage_model(wf) == "divingZImageTurbo_v60Fp16.safetensors"


def test_non_zimage_is_passthrough():
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "cyberrealisticPony_v180Coreshift.safetensors"}}}
    assert zimage.detect_zimage_model(wf) is None


def test_anima_is_not_zimage():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima_basev10.safetensors"}}}
    assert zimage.detect_zimage_model(wf) is None


# --- template integrity ----------------------------------------------------

def test_template_loads_and_has_no_dangling_links():
    wf = json.load(open(TEMPLATE))
    ids = set(wf)
    for nid, node in wf.items():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                assert str(v[0]) in ids, f"node {nid} links to missing {v[0]}"


def test_template_florence2_is_base_promptgen():
    wf = json.load(open(TEMPLATE))
    f2 = next(n for n in wf.values() if n.get("class_type") == "DownloadAndLoadFlorence2Model")
    assert f2["inputs"]["model"] == "MiaoshouAI/Florence-2-base-PromptGen-v2.0"


def test_template_has_no_unbaked_sampler_selector():
    wf = json.load(open(TEMPLATE))
    assert all(n.get("class_type") != "Sampler Selector" for n in wf.values())


# --- substitution: with VLM image -----------------------------------------

def test_build_with_input_image_keeps_vlm_and_sets_loadimage():
    out = zimage.build_zimage_workflow(
        "moodyProMix_zitV13.safetensors", "a girl", "bad", input_image_name="abc.png"
    )
    # VLM branch retained
    assert any(n.get("class_type") == "Florence2Run" for n in out.values())
    li = [n for n in out.values() if n.get("class_type") == "LoadImage"]
    assert li and li[0]["inputs"]["image"] == "abc.png"
    # model injected onto the UNET loader
    unet = next(n for n in out.values() if n.get("class_type") == "UNETLoader")
    assert unet["inputs"]["unet_name"] == "moodyProMix_zitV13.safetensors"
    # prompts injected
    pos = next(n for n in out.values() if n.get("class_type") == "easy positive")
    assert pos["inputs"]["positive"] == "a girl"


# --- substitution: no image -> VLM pruned ----------------------------------

def test_build_without_image_prunes_vlm():
    out = zimage.build_zimage_workflow("moodyRealMix_zitV7.safetensors", "a girl", "bad")
    assert all(n.get("class_type") != "Florence2Run" for n in out.values())
    assert all(n.get("class_type") != "LoadImage" for n in out.values())
    assert all(n.get("class_type") != "DownloadAndLoadFlorence2Model" for n in out.values())
    # styles selector positive now feeds straight from easy positive
    pos_id = next(i for i, n in out.items() if n.get("class_type") == "easy positive")
    styles = next(n for n in out.values() if n.get("class_type") == "easy stylesSelector")
    assert styles["inputs"]["positive"] == [pos_id, 0]
    # graph still valid (no dangling links after the prune+GC)
    ids = set(out)
    for n in out.values():
        for v in (n.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                assert str(v[0]) in ids


def test_saver_survives_both_paths():
    for img in ("x.png", None):
        out = zimage.build_zimage_workflow("moodyProMix_zitV13.safetensors", "p", "n", input_image_name=img)
        savers = [n for n in out.values() if n.get("class_type") == "SaveImageWithMetaData"]
        assert savers, "saver must survive"
        img_link = savers[0]["inputs"].get("images")
        assert isinstance(img_link, list) and str(img_link[0]) in out


# --- end-to-end rewrite ----------------------------------------------------

def test_maybe_rewrite_carries_prompt_and_model():
    out = zimage.maybe_rewrite_to_zimage(_catpony_like("moodyProMix_zitV13.safetensors"))
    assert out is not None
    pos = next(n for n in out.values() if n.get("class_type") == "easy positive")
    assert "tall woman on a beach" in pos["inputs"]["positive"]


def test_maybe_rewrite_none_for_non_zimage():
    assert zimage.maybe_rewrite_to_zimage(_catpony_like("cyberrealisticPony_v180Coreshift.safetensors")) is None


# --- LoRA stack injection ---------------------------------------------------

def _seed_normalized(wf: dict) -> dict:
    """Deep copy of ``wf`` with every literal ``seed``/``noise_seed`` int
    zeroed, so two builds that differ only by the per-call random seed compare
    equal."""
    out = copy.deepcopy(wf)
    for node in out.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), int):
                inputs[key] = 0
    return out


def test_loras_absent_is_noop():
    wf = zimage.build_zimage_workflow("moodyProMix_zitV13.safetensors", "p", "n", options={})
    assert not any(n.get("class_type") == "LoraLoader" for n in wf.values())
    # Byte-identical (seed aside) to the options=None baseline — the docstring's
    # promise that an empty/missing lora list leaves today's output unchanged.
    baseline = zimage.build_zimage_workflow("moodyProMix_zitV13.safetensors", "p", "n", options=None)
    assert _seed_normalized(wf) == _seed_normalized(baseline)


def test_loras_chain_wiring():
    opts = {"loras": [{"name": "a.safetensors", "strength": 0.8},
                      {"name": "b", "strength": 1.2}]}
    wf = zimage.build_zimage_workflow("moodyProMix_zitV13.safetensors", "p", "n", options=opts)
    l1, l2 = wf["9001"], wf["9002"]
    assert l1["class_type"] == "LoraLoader" and l1["inputs"]["lora_name"] == "a.safetensors"
    assert l1["inputs"]["model"] == ["474", 0] and l1["inputs"]["clip"] == ["471", 0]
    assert l1["inputs"]["strength_model"] == 0.8
    # bare name gets the extension appended
    assert l2["inputs"]["lora_name"] == "b.safetensors"
    assert l2["inputs"]["model"] == ["9001", 0] and l2["inputs"]["clip"] == ["9001", 1]
    # the two direct consumers of the base loaders (rgthree passthrough's model
    # input + CLIPSetLastLayer's clip input) now feed from the chain end
    assert wf["21"]["inputs"]["model"] == ["9002", 0]
    assert wf["473"]["inputs"]["clip"] == ["9002", 1]
    # base loaders themselves are untouched
    assert wf["474"]["inputs"]["unet_name"] == "moodyProMix_zitV13.safetensors"
    assert "clip_name" in wf["471"]["inputs"]


def test_loras_malformed_entries_dropped():
    opts = {"loras": ["junk", {"strength": 1}, {"name": ""},
                      {"name": "ok.safetensors", "strength": "high"},
                      {"name": "clamp.safetensors", "strength": 99}]}
    wf = zimage.build_zimage_workflow("moodyProMix_zitV13.safetensors", "p", "n", options=opts)
    chain = [n for n in wf.values() if n.get("class_type") == "LoraLoader"]
    assert len(chain) == 2
    assert chain[0]["inputs"]["strength_model"] == 1.0      # bad strength -> default
    assert chain[1]["inputs"]["strength_model"] == 4.0      # clamped


def test_loras_capped_at_max():
    opts = {"loras": [{"name": f"l{i}.safetensors"} for i in range(20)]}
    wf = zimage.build_zimage_workflow("moodyProMix_zitV13.safetensors", "p", "n", options=opts)
    assert sum(1 for n in wf.values() if n.get("class_type") == "LoraLoader") == 8


def test_loras_survive_txt2img_and_vlm_paths():
    # The chain feeds the KSampler via node 21 (rgthree passthrough), which
    # survives both the VLM-prune and the heavy-postprocessing-bypass GC passes
    # on either code path — so the chain must stay intact and dangling-free.
    opts = {"loras": [{"name": "x.safetensors", "strength": 1.0}]}
    for img in ("input.png", None):
        wf = zimage.build_zimage_workflow(
            "moodyProMix_zitV13.safetensors", "p", "n", input_image_name=img, options=opts
        )
        assert any(n.get("class_type") == "LoraLoader" for n in wf.values())
        ids = set(wf)
        for n in wf.values():
            for v in (n.get("inputs") or {}).values():
                if isinstance(v, list) and len(v) == 2:
                    assert str(v[0]) in ids


def test_loras_via_maybe_rewrite():
    out = zimage.maybe_rewrite_to_zimage(
        _catpony_like("moodyProMix_zitV13.safetensors"),
        options={"loras": [{"name": "a.safetensors", "strength": 1.0}]},
    )
    assert out is not None
    assert any(n.get("class_type") == "LoraLoader" for n in out.values())
