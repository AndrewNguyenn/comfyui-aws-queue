"""Unit tests for the status Lambda's pure derivation logic — specifically the
character extraction that drives the viewer's pending-strip grouping.

Run: python3 services/status/test_handler.py
(or `python3 -m pytest services/status/test_handler.py`). boto3 / botocore
clients are created at import time, so we stub them before importing the
handler.
"""
import importlib.util
import os
import sys
import types

sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *a, **k: None))
_botocore = types.ModuleType("botocore")
_config = types.ModuleType("botocore.config")
_config.Config = lambda *a, **k: None
_exc = types.ModuleType("botocore.exceptions")
_exc.ClientError = type("ClientError", (Exception,), {})
_botocore.config = _config
_botocore.exceptions = _exc
sys.modules.setdefault("botocore", _botocore)
sys.modules.setdefault("botocore.config", _config)
sys.modules.setdefault("botocore.exceptions", _exc)

os.environ.setdefault("JOBS_TABLE", "x")
os.environ.setdefault("OUTPUTS_BUCKET", "x")
os.environ.setdefault("UPLOADS_BUCKET", "x")

_spec = importlib.util.spec_from_file_location(
    "status_handler", os.path.join(os.path.dirname(__file__), "handler.py")
)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def test_norm_tag_unwraps_emphasis_but_keeps_internal_parens():
    # emphasis syntax wraps the WHOLE tag → unwrap
    assert h._norm_tag("(close-up:1.2)") == "close-up"
    assert h._norm_tag("((narberal_gamma))") == "narberal_gamma"
    assert h._norm_tag("[saber]") == "saber"
    assert h._norm_tag("Artoria_Pendragon") == "artoria_pendragon"
    # a character tag with INTERNAL parens must survive intact (the closing
    # paren is part of the tag, not emphasis)
    assert h._norm_tag("ais_wallenstein_(danmachi)") == "ais_wallenstein_(danmachi)"
    assert h._norm_tag("overlord_(maruyama)") == "overlord_(maruyama)"
    # a colon that isn't a numeric weight is left alone
    assert h._norm_tag(":d") == ":d"
    # the `tag::1.3` double-colon weight form must not leave a dangling colon
    assert h._norm_tag("narberal_gamma::1.2") == "narberal_gamma"
    # leading underscore from a ",_tag" join is stripped (internal _ kept)
    assert h._norm_tag("_best_quality") == "best_quality"
    assert h._norm_tag("_very_aesthetic") == "very_aesthetic"
    # a tag carrying the leaked group-close bracket on its weight: "_c:1.2)"
    assert h._norm_tag("_absurdres:1.2)") == "absurdres"


def test_character_is_first_non_framing_tag():
    cases = {
        "close-up, face_focus BREAK ais_wallenstein_(danmachi), dungeon_ni_deai, long_hair":
            "ais_wallenstein_(danmachi)",
        "from_front, low_angle, artoria_pendragon, fate/stay_night, saber":
            "artoria_pendragon",
        "narberal_gamma, overlord_(maruyama), long_hair":
            "narberal_gamma",
        "(close-up:1.2), breast_focus, nobara_kugisaki, jujutsu_kaisen":
            "nobara_kugisaki",
        "upper_body BREAK ((narberal_gamma)), overlord":
            "narberal_gamma",
        # Pony/Illustrious quality + score leads must be skipped (catponyDark is
        # that lineage) — else every batch collapses under score_9/masterpiece.
        "score_9, score_8_up, score_7_up, narberal_gamma, overlord_(maruyama)":
            "narberal_gamma",
        "masterpiece, best_quality, absurdres, artoria_pendragon, fate/stay_night":
            "artoria_pendragon",
        "score_9_up, very_aesthetic, from_front, nobara_kugisaki, jujutsu_kaisen":
            "nobara_kugisaki",
        # inline LoRA/embedding tokens aren't a character
        "<lora:detail:0.8>, masterpiece, narberal_gamma, overlord_(maruyama)":
            "narberal_gamma",
        # persona/age descriptors lead before the character in this template
        "breast_focus, from_front, upper_body, mature_female, bronya_rand, honkai:_star_rail":
            "bronya_rand",
        "face_focus, pov, milf, artoria_pendragon, fate/stay_night":
            "artoria_pendragon",
        # series-anchor: ANY descriptors (here CLOTHING) may precede the
        # character; it's pinned as the (non-lead) tag right before the series.
        "breast_focus, from_front, upper_body BREAK crop_top, "
        "jingliu_(honkai:_star_rail), honkai:_star_rail, very_long_hair, white_hair":
            "jingliu_(honkai:_star_rail)",
        # a name_(franchise) parenthetical is a character even when no series tag
        # follows it (series omitted from the prompt)
        "from_front BREAK crop_top, tracer_(overwatch), blue_eyes":
            "tracer_(overwatch)",
        # plain character pinned by a PUNCTUATED series (no franchise-dict entry
        # needed — ":" is detected structurally)
        "close-up BREAK skirt, bronya_rand, honkai:_star_rail":
            "bronya_rand",
        # the real catpony preamble: a weighted quality GROUP joined with ",_"
        # plus a NoobAI very_awa tag, then a focus tag — the character follows.
        "from_front, close-up, extreme_close-up, bust_shot, face_focus, breast_focus "
        "BREAK (masterpiece,_best_quality,_very_aesthetic,_absurdres:1.2), very_awa "
        "BREAK vanessa_enoteca, black_clover, long_hair, pink_hair":
            "vanessa_enoteca",
        # back_focus (and any *_focus) is framing, not a character
        "back_focus, face_focus, ass_focus BREAK (masterpiece,_best_quality:1.2) "
        "BREAK ningguang_(genshin_impact), genshin_impact":
            "ningguang_(genshin_impact)",
        # fallback: a LEADING character whose franchise isn't recognized is still
        # found via the first-content-tag fallback (not lost to no-character).
        # Also exercises the added framing tags (three-quarter_angle, …).
        "three-quarter_angle, establishing_shot BREAK protagonist_x, "
        "totally_made_up_series_zzz, blue_eyes, long_hair":
            "protagonist_x",
        # composition/scale leads (macro, bokeh) are skipped; goblin_slayer! is a
        # recognized franchise (the trailing ! defeats structural detection)
        "extreme_close-up, macro, from_front, face_focus BREAK sword_maiden, "
        "goblin_slayer!, blonde_hair, blindfold, huge_breasts":
            "sword_maiden",
    }
    for prompt, expected in cases.items():
        assert h._character_from_prompt(prompt) == expected, (prompt, h._character_from_prompt(prompt))


def test_is_series_excludes_emoji_mouth_tags():
    # internal-colon series detected; leading/trailing colon emoji faces are not
    assert h._is_series("honkai:_star_rail")
    assert h._is_series("fate/stay_night")
    assert not h._is_series("d:")   # trailing-colon mouth (reviewer's case)
    assert not h._is_series(":d")   # leading-colon face
    assert not h._is_series(":3")
    assert not h._is_series("blonde_hair")
    # a name_(series) character is never a series, even with ":" in its parens
    assert not h._is_series("jingliu_(honkai:_star_rail)")


def test_character_empty_when_no_prompt():
    assert h._character_from_prompt("") == ""
    assert h._character_from_prompt("   ") == ""
    # a prompt that is ALL framing tags has no character to surface
    assert h._character_from_prompt("close-up, from_front, solo") == ""


def test_appearance_led_prompt_has_no_character():
    # real catpony prompt: an UNNAMED figure described purely by appearance —
    # the first content tag is a hair colour, so there is no character to label.
    assert h._character_from_prompt(
        "from_front, bust_shot BREAK black_hair, very_long_hair, pale_skin, 1boy "
        "BREAK tears BREAK predicament_bondage, kneeling, penis") == ""
    # other appearance leads (eyes, skin, build) → also no character
    assert h._character_from_prompt("close-up BREAK blue_eyes, large_breasts") == ""
    assert h._character_from_prompt("pov BREAK dark-skinned_female, curvy") == ""
    # but a character that merely HAS appearance tags after it is still found
    assert h._character_from_prompt(
        "from_front BREAK luna_snow, marvel_rivals, blue_eyes, blonde_hair") == "luna_snow"


def test_character_and_subject_hint():
    # unnamed figure → ("", <lead appearance tag>) so the viewer can label it
    # "<tag> figure" instead of a bare model row
    assert h._character_and_subject(
        "from_front, bust_shot BREAK black_hair, very_long_hair, pale_skin, 1boy"
    ) == ("", "black_hair")
    assert h._character_and_subject("close-up BREAK blue_eyes, large_breasts") == ("", "blue_eyes")
    # named character → (character, "")
    assert h._character_and_subject(
        "from_front BREAK luna_snow, marvel_rivals") == ("luna_snow", "")
    # scenery / all-lead → ("", "")
    assert h._character_and_subject("close-up, from_front, solo") == ("", "")


def test_extract_character_from_string_literal_graph():
    # mirrors the real catpony workflow: positive prompt wired from a
    # "String Literal (Image Saver)" node into a CLIPTextEncode → KSampler.
    wf = {
        "10": {"class_type": "String Literal (Image Saver)",
               "_meta": {"title": "Positive prompt"},
               "inputs": {"string": "from_front, low_angle, artoria_pendragon, fate/stay_night, saber"}},
        "11": {"class_type": "String Literal (Image Saver)",
               "_meta": {"title": "Negative prompt"},
               "inputs": {"string": "worst_quality, bad_hands"}},
        "20": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive prompt"},
               "inputs": {"text": ["10", 0], "clip": ["99", 1]}},
        "21": {"class_type": "CLIPTextEncode", "_meta": {"title": "Negative prompt"},
               "inputs": {"text": ["11", 0], "clip": ["99", 1]}},
        "30": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
               "inputs": {"positive": ["20", 0], "negative": ["21", 0],
                          "model": ["99", 0], "latent_image": ["98", 0]}},
    }
    assert h._extract_character(wf) == "artoria_pendragon"


def test_extract_character_from_inline_cliptextencode():
    wf = {
        "20": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive"},
               "inputs": {"text": "close-up, narberal_gamma, overlord_(maruyama), long_hair"}},
        "21": {"class_type": "CLIPTextEncode", "_meta": {"title": "Negative"},
               "inputs": {"text": "lowres, bad_anatomy"}},
        "30": {"class_type": "KSampler",
               "inputs": {"positive": ["20", 0], "negative": ["21", 0]}},
    }
    assert h._extract_character(wf) == "narberal_gamma"


def test_serialize_job_only_adds_character_for_pending_statuses():
    item = {
        "job_id": {"S": "j1"},
        "type": {"S": "image"},
        "status": {"S": "queued"},
        "workflow_json": {"S": '{"20": {"class_type": "CLIPTextEncode", '
                               '"_meta": {"title": "Positive"}, '
                               '"inputs": {"text": "narberal_gamma, overlord, long_hair"}}, '
                               '"30": {"class_type": "KSampler", '
                               '"inputs": {"positive": ["20", 0]}}}'},
    }
    queued = h._serialize_job(item, lite=True)
    assert queued["character"] == "narberal_gamma"

    item["status"] = {"S": "complete"}
    done = h._serialize_job(item, lite=True)
    # gallery rows (complete/failed/cancelled) never group by character, so the
    # field is omitted to keep the big history fetch cheap
    assert "character" not in done


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
