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
