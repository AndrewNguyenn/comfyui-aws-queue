"""Tests for Krea 2 detection / substitution / LoRA injection."""
import copy
import json
import os

import krea

HERE = os.path.dirname(__file__)
TEMPLATE = os.path.join(HERE, "krea_templates", "Krea2Simple.api.json")
OBJECT_INFO_CLASSES = json.load(
    open(os.path.join(HERE, "testdata", "krea2_object_info_classes.json"))
)


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

def test_detect_via_unet_loader():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_raw_fp8_scaled.safetensors"}}}
    assert krea.detect_krea_model(wf) == "krea2_raw_fp8_scaled.safetensors"


def test_detect_case_insensitive_and_uncataloged_variant():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Krea2_Raw_BF16.safetensors"}}}
    assert krea.detect_krea_model(wf) == "Krea2_Raw_BF16.safetensors"


def test_detect_via_ckpt_name_path_prefix():
    wf = {"1": {"class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "diffusion_models/krea2_raw_fp8_scaled.safetensors"}}}
    assert krea.detect_krea_model(wf) == "diffusion_models/krea2_raw_fp8_scaled.safetensors"


def test_detect_without_extension():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_raw_fp8_scaled"}}}
    assert krea.detect_krea_model(wf) == "krea2_raw_fp8_scaled"


def test_non_krea_is_passthrough():
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "cyberrealisticPony_v180Coreshift.safetensors"}}}
    assert krea.detect_krea_model(wf) is None


def test_anima_is_not_krea():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima_basev10.safetensors"}}}
    assert krea.detect_krea_model(wf) is None


def test_zimage_is_not_krea():
    wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "moodyProMix_zitV13.safetensors"}}}
    assert krea.detect_krea_model(wf) is None


# --- template integrity ----------------------------------------------------

def test_template_loads_and_has_no_dangling_links():
    wf = json.load(open(TEMPLATE))
    ids = set(wf)
    for nid, node in wf.items():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                assert str(v[0]) in ids, f"node {nid} links to missing {v[0]}"


def test_template_reachable_from_saver():
    wf = json.load(open(TEMPLATE))
    saver = next(n for n in wf.values() if n.get("class_type") == "SaveImage")
    seen: set = set()
    stack = [str(saver["inputs"]["images"][0])]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = wf.get(nid)
        assert node is not None, f"reachable node {nid} missing from graph"
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                stack.append(str(v[0]))
    # every node in the template must be reachable — no orphaned scaffolding
    assert seen == set(wf) - {next(i for i, n in wf.items() if n.get("class_type") == "SaveImage")}


def test_template_class_types_exist_in_fleet_object_info():
    # CLIPLoader's "krea2" type enum value is validated separately (the fleet
    # object_info dump predates the ComfyUI upgrade that adds it) — this test
    # only checks the NODE CLASSES themselves are real fleet nodes.
    wf = json.load(open(TEMPLATE))
    class_types = {n.get("class_type") for n in wf.values()}
    for c in class_types:
        assert c in OBJECT_INFO_CLASSES, f"{c} not found in fleet object_info dump"


def test_template_bongmath_and_cfg_pinned():
    wf = json.load(open(TEMPLATE))
    samplers = [n for n in wf.values() if n.get("class_type") == "ClownsharKSampler_Beta"]
    assert len(samplers) == 2
    for s in samplers:
        assert s["inputs"]["cfg"] == 1.0
        assert s["inputs"]["bongmath"] is True
        assert s["inputs"]["sampler_mode"] == "standard"


def test_template_two_pass_settings():
    wf = json.load(open(TEMPLATE))
    base = wf["8"]["inputs"]
    refiner = wf["9"]["inputs"]
    assert base["sampler_name"] == "exponential/res_2s"
    assert base["scheduler"] == "beta"
    assert base["steps"] == 6
    assert base["denoise"] == 1.0
    assert refiner["sampler_name"] == "multistep/deis_3m"
    assert refiner["scheduler"] == "bong_tangent"
    assert refiner["steps"] == 2
    assert refiner["denoise"] == 0.2
    # refiner chains off the base pass's latent output
    assert refiner["latent_image"] == ["8", 0]


# --- baked loras -------------------------------------------------------

def test_baked_loras_present_with_exact_strengths():
    out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n")
    baked = [n for n in out.values() if n.get("class_type") == "LoraLoaderModelOnly"]
    assert len(baked) == 2
    turbo = next(n for n in baked if "turbo" in n["inputs"]["lora_name"].lower())
    bypass = next(n for n in baked if "filterbypass" in n["inputs"]["lora_name"].lower())
    assert turbo["inputs"]["strength_model"] == 0.6
    assert bypass["inputs"]["strength_model"] == 1.0
    # chain order: UNETLoader -> turbo -> filterbypass (baked chain tail)
    unet_id = next(i for i, n in out.items() if n.get("class_type") == "UNETLoader")
    assert turbo["inputs"]["model"] == [unet_id, 0]
    turbo_id = next(i for i, n in out.items() if n is turbo)
    assert bypass["inputs"]["model"] == [turbo_id, 0]


# --- substitution: model + prompts -----------------------------------------

def test_build_injects_model_onto_unet_loader():
    out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "a girl", "bad")
    unet = next(n for n in out.values() if n.get("class_type") == "UNETLoader")
    assert unet["inputs"]["unet_name"] == "krea2_raw_fp8_scaled.safetensors"


def test_build_injects_prompts():
    out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "a girl on a beach", "worst quality")
    texts = {n["inputs"]["text"] for n in out.values() if n.get("class_type") == "CLIPTextEncode"}
    assert "a girl on a beach" in texts
    assert "worst quality" in texts


def test_prompts_always_overwritten_even_empty():
    out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "", "")
    texts = [n["inputs"]["text"] for n in out.values() if n.get("class_type") == "CLIPTextEncode"]
    assert texts == ["", ""]


def test_saver_survives():
    out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n")
    savers = [n for n in out.values() if n.get("class_type") == "SaveImage"]
    assert savers, "saver must survive"
    img_link = savers[0]["inputs"].get("images")
    assert isinstance(img_link, list) and str(img_link[0]) in out


# --- seed injection ----------------------------------------------------

def test_seed_injected_on_both_passes_and_in_range():
    out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n")
    samplers = [n for n in out.values() if n.get("class_type") == "ClownsharKSampler_Beta"]
    seeds = {s["inputs"]["seed"] for s in samplers}
    assert len(seeds) == 1, "both passes must share the same injected seed"
    (seed,) = seeds
    assert 0 <= seed < krea._SEED_MAX


def test_seed_varies_across_builds():
    seeds = set()
    for _ in range(5):
        out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n")
        s = next(n for n in out.values() if n.get("class_type") == "ClownsharKSampler_Beta")
        seeds.add(s["inputs"]["seed"])
    assert len(seeds) > 1, "seeds should be randomized per build"


# --- end-to-end rewrite ----------------------------------------------------

def test_maybe_rewrite_carries_prompt_and_model():
    out = krea.maybe_rewrite_to_krea(_catpony_like("krea2_raw_fp8_scaled.safetensors"))
    assert out is not None
    unet = next(n for n in out.values() if n.get("class_type") == "UNETLoader")
    assert unet["inputs"]["unet_name"] == "krea2_raw_fp8_scaled.safetensors"
    texts = {n["inputs"]["text"] for n in out.values() if n.get("class_type") == "CLIPTextEncode"}
    assert "a tall woman on a beach" in next(t for t in texts if "tall woman" in t)


def test_maybe_rewrite_none_for_non_krea():
    assert krea.maybe_rewrite_to_krea(_catpony_like("cyberrealisticPony_v180Coreshift.safetensors")) is None


def test_maybe_rewrite_ignores_input_image_name():
    # txt2img-only template — passing an input_image_name must not raise or
    # change the outcome; the template has no LoadImage node at all.
    out = krea.maybe_rewrite_to_krea(
        _catpony_like("krea2_raw_fp8_scaled.safetensors"), input_image_name="reference.png"
    )
    assert out is not None
    assert all(n.get("class_type") != "LoadImage" for n in out.values())


def test_maybe_rewrite_never_raises_on_malformed_input():
    for bad in (None, {}, {"1": "not-a-dict"}, {"1": {"inputs": "nope"}}, {"1": {"class_type": 5, "inputs": {}}}):
        try:
            result = krea.maybe_rewrite_to_krea(bad)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"maybe_rewrite_to_krea raised on {bad!r}: {e!r}")
        assert result is None or isinstance(result, dict)


def test_detect_tolerates_malformed_nodes():
    # detect_krea_model's contract (matching zimage.detect_zimage_model /
    # anima.detect_anima_model) is: given a dict, malformed per-node values
    # are skipped safely. A non-dict `workflow` itself is NOT guarded here —
    # that's the job of maybe_rewrite_to_krea's outer try/except, covered by
    # test_maybe_rewrite_never_raises_on_malformed_input above.
    for bad in ({}, {"1": None}, {"1": {"inputs": None}}, {"1": "not-a-dict"}):
        assert krea.detect_krea_model(bad) is None


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
    wf = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n", options={})
    assert not any(n.get("class_type") == "LoraLoader" for n in wf.values())
    baseline = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n", options=None)
    assert _seed_normalized(wf) == _seed_normalized(baseline)


def test_loras_chain_wiring():
    opts = {"loras": [{"name": "a.safetensors", "strength": 0.8},
                      {"name": "b", "strength": 1.2}]}
    wf = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n", options=opts)
    l1, l2 = wf["9001"], wf["9002"]
    assert l1["class_type"] == "LoraLoader" and l1["inputs"]["lora_name"] == "a.safetensors"
    # chain anchors: model from the baked chain tail (node 3), clip from CLIPLoader (node 4)
    assert l1["inputs"]["model"] == ["3", 0] and l1["inputs"]["clip"] == ["4", 0]
    assert l1["inputs"]["strength_model"] == 0.8
    # bare name gets the extension appended
    assert l2["inputs"]["lora_name"] == "b.safetensors"
    assert l2["inputs"]["model"] == ["9001", 0] and l2["inputs"]["clip"] == ["9001", 1]
    # both sampler passes' model input + both CLIPTextEncodes' clip input now
    # feed from the chain end
    assert wf["8"]["inputs"]["model"] == ["9002", 0]
    assert wf["9"]["inputs"]["model"] == ["9002", 0]
    assert wf["5"]["inputs"]["clip"] == ["9002", 1]
    assert wf["6"]["inputs"]["clip"] == ["9002", 1]
    # baked loras themselves are untouched (still chained UNET -> turbo -> filterbypass)
    assert wf["1"]["inputs"]["unet_name"] == "krea2_raw_fp8_scaled.safetensors"
    assert wf["3"]["inputs"]["lora_name"] == "krea2filterbypass.safetensors"


def test_loras_malformed_entries_dropped():
    opts = {"loras": ["junk", {"strength": 1}, {"name": ""},
                      {"name": "ok.safetensors", "strength": "high"},
                      {"name": "clamp.safetensors", "strength": 99}]}
    wf = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n", options=opts)
    chain = [n for n in wf.values() if n.get("class_type") == "LoraLoader"]
    assert len(chain) == 2
    assert chain[0]["inputs"]["strength_model"] == 1.0      # bad strength -> default
    assert chain[1]["inputs"]["strength_model"] == 4.0      # clamped


def test_loras_capped_at_max():
    opts = {"loras": [{"name": f"l{i}.safetensors"} for i in range(20)]}
    wf = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n", options=opts)
    assert sum(1 for n in wf.values() if n.get("class_type") == "LoraLoader") == 8


def test_loras_no_dangling_links():
    opts = {"loras": [{"name": "x.safetensors", "strength": 1.0}]}
    wf = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n", options=opts)
    ids = set(wf)
    for n in wf.values():
        for v in (n.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                assert str(v[0]) in ids


def test_loras_via_maybe_rewrite():
    out = krea.maybe_rewrite_to_krea(
        _catpony_like("krea2_raw_fp8_scaled.safetensors"),
        options={"loras": [{"name": "a.safetensors", "strength": 1.0}]},
    )
    assert out is not None
    assert any(n.get("class_type") == "LoraLoader" for n in out.values())
