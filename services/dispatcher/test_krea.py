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
        assert s["inputs"]["cfg"] == 1.4
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
    assert len(baked) == 3
    turbo = next(n for n in baked if "turbo" in n["inputs"]["lora_name"].lower())
    bypass = next(n for n in baked if "filterbypass" in n["inputs"]["lora_name"].lower())
    amateur = next(n for n in baked if "amateurslider" in n["inputs"]["lora_name"].lower())
    assert turbo["inputs"]["strength_model"] == 0.6
    assert bypass["inputs"]["strength_model"] == 1.0
    assert amateur["inputs"]["strength_model"] == 1.5
    # chain order: UNETLoader -> turbo -> filterbypass -> amateur (baked chain tail)
    unet_id = next(i for i, n in out.items() if n.get("class_type") == "UNETLoader")
    assert turbo["inputs"]["model"] == [unet_id, 0]
    turbo_id = next(i for i, n in out.items() if n is turbo)
    assert bypass["inputs"]["model"] == [turbo_id, 0]
    bypass_id = next(i for i, n in out.items() if n is bypass)
    assert amateur["inputs"]["model"] == [bypass_id, 0]


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
    # chain anchors: model from the baked chain tail (node 13), clip from CLIPLoader (node 4)
    assert l1["inputs"]["model"] == ["13", 0] and l1["inputs"]["clip"] == ["4", 0]
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
    # baked loras themselves are untouched (still chained UNET -> turbo -> filterbypass -> amateur)
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



# --- pre-baked-turbo checkpoints -------------------------------------------
#
# Krea2 TURBO fp8 and RedCraft RedMix 3.0 Lightning8 ship the distillation
# merged into the weights, so the template's baked turbo lora must be spliced
# out for them (KREA_PREBAKED_TURBO) — see krea.py's module docstring.

PREBAKED = "krea2TurboFP8_krea2TURBO.safetensors"
REDMIX = "redcraftMinimaxH3REDMIX_30Krea2.safetensors"
BOTH_PREBAKED = (PREBAKED, REDMIX)


def test_prebaked_turbo_models_are_all_allowlisted():
    assert krea.KREA_PREBAKED_TURBO <= krea.KREA_MODELS


def test_model_sampler_overrides_are_all_allowlisted():
    assert set(krea._KREA_MODEL_SAMPLER) <= krea.KREA_MODELS


def test_detect_new_krea_checkpoints():
    for model in BOTH_PREBAKED:
        wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": model}}}
        assert krea.detect_krea_model(wf) == model


def test_prebaked_turbo_drops_only_the_turbo_lora():
    for model in BOTH_PREBAKED:
        out = krea.build_krea_workflow(model, "p", "n")
        names = [
            n["inputs"]["lora_name"].lower()
            for n in out.values()
            if n.get("class_type") == "LoraLoaderModelOnly"
        ]
        assert not any("turbo" in n for n in names), f"{model}: turbo lora must be spliced out"
        assert any("filterbypass" in n for n in names), model
        assert any("amateurslider" in n for n in names), model


def test_prebaked_turbo_rewires_chain_to_the_unet_loader():
    for model in BOTH_PREBAKED:
        out = krea.build_krea_workflow(model, "p", "n")
        assert krea._TURBO_LORA_NODE not in out, model
        unet_id = next(i for i, n in out.items() if n.get("class_type") == "UNETLoader")
        bypass = next(
            n for n in out.values()
            if n.get("class_type") == "LoraLoaderModelOnly"
            and "filterbypass" in n["inputs"]["lora_name"].lower()
        )
        assert bypass["inputs"]["model"] == [unet_id, 0], model
        # samplers still read off the unchanged baked tail
        for s in out.values():
            if s.get("class_type") == "ClownsharKSampler_Beta":
                assert s["inputs"]["model"] == [krea._BAKED_TAIL_NODE, 0], model
        # no link anywhere still points at the deleted node
        for n in out.values():
            for val in (n.get("inputs") or {}).values():
                if isinstance(val, list) and len(val) == 2:
                    assert str(val[0]) != krea._TURBO_LORA_NODE, model


def test_prebaked_build_differs_from_raw_by_exactly_the_turbo_node():
    """The strongest guard on the splice: PREBAKED takes no sampler override,
    so its graph must equal the raw build minus node 2 and nothing else."""
    raw = _seed_normalized(krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n"))
    pre = _seed_normalized(krea.build_krea_workflow(PREBAKED, "p", "n"))
    assert set(raw) - set(pre) == {krea._TURBO_LORA_NODE}
    assert set(pre) - set(raw) == set()
    raw["1"]["inputs"]["unet_name"] = PREBAKED
    raw["3"]["inputs"]["model"] = ["1", 0]
    del raw[krea._TURBO_LORA_NODE]
    assert raw == pre


def test_raw_base_model_keeps_the_baked_turbo_lora():
    out = krea.build_krea_workflow("krea2_raw_fp8_scaled.safetensors", "p", "n")
    assert krea._TURBO_LORA_NODE in out
    names = [
        n["inputs"]["lora_name"].lower()
        for n in out.values()
        if n.get("class_type") == "LoraLoaderModelOnly"
    ]
    assert any("turbo" in n for n in names)


def test_prebaked_decision_survives_a_path_prefixed_model_name():
    """build_krea_workflow normalizes before deciding — a catalog path prefix
    (as detect_krea_model can return) must not fall through to the raw path."""
    for model in BOTH_PREBAKED:
        out = krea.build_krea_workflow("diffusion_models/" + model, "p", "n")
        assert krea._TURBO_LORA_NODE not in out, model


def test_user_loras_still_splice_onto_the_tail_when_turbo_is_bypassed():
    for model in BOTH_PREBAKED:
        out = krea.build_krea_workflow(
            model, "p", "n", options={"loras": [{"name": "x.safetensors", "strength": 0.8}]}
        )
        chain = [n for n in out.values() if n.get("class_type") == "LoraLoader"]
        assert len(chain) == 1, model
        assert chain[0]["inputs"]["model"] == [krea._BAKED_TAIL_NODE, 0], model
        chain_id = next(i for i, n in out.items() if n is chain[0])
        for s in out.values():
            if s.get("class_type") == "ClownsharKSampler_Beta":
                assert s["inputs"]["model"] == [chain_id, 0], model


def test_prebaked_substitution_end_to_end():
    for model in BOTH_PREBAKED:
        out = krea.maybe_rewrite_to_krea(_catpony_like(model))
        assert out is not None, model
        unet = next(n for n in out.values() if n.get("class_type") == "UNETLoader")
        assert unet["inputs"]["unet_name"] == model
        assert all(n.get("class_type") in OBJECT_INFO_CLASSES for n in out.values())


# --- _bypass_model_node guards ---------------------------------------------
#
# Every guard must leave the graph untouched — a hand-edited template that
# trips one keeps the turbo lora (today's behaviour) rather than losing a link.

def test_bypass_guards_leave_the_graph_untouched():
    cases = {
        "node missing": {"3": {"inputs": {"model": ["2", 0]}}},
        "node not a dict": {"2": "nope", "3": {"inputs": {"model": ["2", 0]}}},
        "no model input": {"2": {"inputs": {}}, "3": {"inputs": {"model": ["2", 0]}}},
        "model is a literal": {"2": {"inputs": {"model": "x"}}, "3": {"inputs": {"model": ["2", 0]}}},
        "upstream not in graph": {"2": {"inputs": {"model": ["99", 0]}},
                                  "3": {"inputs": {"model": ["2", 0]}}},
    }
    for label, wf in cases.items():
        before = copy.deepcopy(wf)
        krea._bypass_model_node(wf, "2")
        assert wf == before, f"{label}: graph must be unchanged"


def test_bypass_only_rewires_model_inputs():
    """A same-id link on another socket is not part of the MODEL chain and must
    survive untouched — only ``model`` inputs are repointed."""
    wf = {
        "1": {"inputs": {}},
        "2": {"inputs": {"model": ["1", 0]}},
        "3": {"inputs": {"model": ["2", 0], "latent_image": ["2", 0]}},
    }
    krea._bypass_model_node(wf, "2")
    assert wf["3"]["inputs"]["model"] == ["1", 0]
    assert wf["3"]["inputs"]["latent_image"] == ["2", 0]


# --- per-model sampler overrides -------------------------------------------

def test_redmix_gets_its_published_step_floor_on_the_base_pass():
    out = krea.build_krea_workflow(REDMIX, "p", "n")
    base = out[krea._BASE_SAMPLER_NODE]["inputs"]
    assert base["steps"] == 8
    # tuned template values are NOT overridden by the model card's defaults
    assert base["cfg"] == 1.4
    assert base["sampler_name"] == "exponential/res_2s"
    assert base["denoise"] == 1.0
    # refiner pass is never per-model
    refiner = next(
        n for i, n in out.items()
        if n.get("class_type") == "ClownsharKSampler_Beta" and i != krea._BASE_SAMPLER_NODE
    )["inputs"]
    assert refiner["steps"] == 2 and refiner["denoise"] == 0.2


def test_unlisted_models_keep_the_template_sampler_settings():
    for model in ("krea2_raw_fp8_scaled.safetensors", PREBAKED):
        base = krea.build_krea_workflow(model, "p", "n")[krea._BASE_SAMPLER_NODE]["inputs"]
        assert base["steps"] == 6, model
        assert base["cfg"] == 1.4, model
