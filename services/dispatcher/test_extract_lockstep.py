"""Lockstep test — services/dispatcher/extract.py MUST agree, function for
function, with services/status/handler.py on model/character/subject derivation.

The dispatcher denormalizes model/character/subject onto each job row at write
time; the status Lambda derives the same fields at read time for old rows (and
cancel-group matches a viewer group on them). If the two copies of the
extraction logic drift, cancel-group's _group_key stops matching and old/new
rows get inconsistent labels. This test runs BOTH copies over the real submitted
workflow fixture and the booru-prompt corpus and asserts identical output.

Run: python3 services/dispatcher/test_extract_lockstep.py
(or `python3 -m pytest services/dispatcher/test_extract_lockstep.py`).
boto3/botocore are stubbed before importing the status handler (it builds AWS
clients at import time); extract.py is pure and needs no stubs.
"""
import importlib.util
import inspect
import json
import os
import sys
import types

# --- stub boto3/botocore so importing the status handler needs no AWS ---
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
for _k in ("JOBS_TABLE", "OUTPUTS_BUCKET", "UPLOADS_BUCKET"):
    os.environ.setdefault(_k, "x")

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load("status_handler", os.path.join(_ROOT, "services", "status", "handler.py"))
ex = _load("dispatcher_extract", os.path.join(_HERE, "extract.py"))

# Booru-prompt corpus — mirrors services/status/test_handler.py's character
# cases plus the empty / all-framing / appearance-led edges.
_PROMPTS = [
    "close-up, face_focus BREAK ais_wallenstein_(danmachi), dungeon_ni_deai, long_hair",
    "from_front, low_angle, artoria_pendragon, fate/stay_night, saber",
    "narberal_gamma, overlord_(maruyama), long_hair",
    "(close-up:1.2), breast_focus, nobara_kugisaki, jujutsu_kaisen",
    "upper_body BREAK ((narberal_gamma)), overlord",
    "score_9, score_8_up, score_7_up, narberal_gamma, overlord_(maruyama)",
    "masterpiece, best_quality, absurdres, artoria_pendragon, fate/stay_night",
    "<lora:detail:0.8>, masterpiece, narberal_gamma, overlord_(maruyama)",
    "breast_focus, from_front, upper_body, mature_female, bronya_rand, honkai:_star_rail",
    "breast_focus, from_front, upper_body BREAK crop_top, "
    "jingliu_(honkai:_star_rail), honkai:_star_rail, very_long_hair, white_hair",
    "from_front BREAK crop_top, tracer_(overwatch), blue_eyes",
    "close-up BREAK skirt, bronya_rand, honkai:_star_rail",
    "from_front, close-up, extreme_close-up, bust_shot, face_focus, breast_focus "
    "BREAK (masterpiece,_best_quality,_very_aesthetic,_absurdres:1.2), very_awa "
    "BREAK vanessa_enoteca, black_clover, long_hair, pink_hair",
    "three-quarter_angle, establishing_shot BREAK protagonist_x, "
    "totally_made_up_series_zzz, blue_eyes, long_hair",
    "from_front BREAK painting_(object), tifa_lockhart, final_fantasy_vii, brown_hair",
    "close-up BREAK fubuki_(one_punch_man), green_dress, black_hair",
    "",
    "   ",
    "close-up, from_front, solo",
    "bust_shot BREAK black_hair, very_long_hair, pale_skin, 1boy",
]

_WF = json.load(open(os.path.join(_HERE, "testdata", "anima_submitted_real.json")))


def test_character_and_subject_agree_on_corpus():
    for p in _PROMPTS:
        assert h._character_and_subject(p) == ex._character_and_subject(p), p
        assert h._character_from_prompt(p) == ex._character_from_prompt(p), p


def test_model_prompts_subject_agree_on_real_workflow():
    assert ex._extract_model(_WF) == h._extract_model(_WF)
    assert ex._extract_subject(_WF) == h._extract_subject(_WF)
    assert ex._extract_prompts(_WF) == h._extract_prompts(_WF)


def test_real_workflow_yields_nonempty_model():
    # Guards the both-copies-broken case: the real fixture must produce a model,
    # so an agreement that's "both return ''" can't pass silently.
    assert ex._extract_model(_WF), "expected a non-empty model from the real workflow"


# Structural backstop (vs the behavioral corpus above): assert the shared
# functions and tag sets are BYTE-IDENTICAL between the two modules, so a
# divergence the corpus doesn't happen to exercise (e.g. editing one tag in a
# 142-entry frozenset in only one file) is still caught.
_SHARED_FNS = (
    "_parse_workflow", "_extract_model", "_generators", "_extract_prompts",
    "_paren_content", "_is_lead_tag", "_is_appearance", "_series_base",
    "_is_series", "_is_paren_character", "_norm_tag", "_character_and_subject",
    "_character_from_prompt", "_extract_subject", "_extract_character",
)
_SHARED_CONSTS = (
    "_PROMPT_MAX", "_TEXT_KEYS", "_PROVIDER_KEYS", "_NON_CHARACTER_TAGS",
    "_APPEARANCE_TAGS", "_PAREN_QUALIFIERS", "_FRANCHISES",
)


def test_shared_source_is_byte_identical():
    for fn in _SHARED_FNS:
        assert inspect.getsource(getattr(h, fn)) == inspect.getsource(getattr(ex, fn)), \
            f"{fn} source diverged between status/handler.py and dispatcher/extract.py"
    for c in _SHARED_CONSTS:
        assert getattr(h, c) == getattr(ex, c), f"{c} value diverged between the two copies"
    assert h._SCORE_RE.pattern == ex._SCORE_RE.pattern


if __name__ == "__main__":
    import traceback
    fails = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("ok  ", _name)
            except Exception:  # noqa: BLE001
                fails += 1
                print("FAIL", _name)
                traceback.print_exc()
    print(f"\n{'PASS' if not fails else 'FAIL'} ({fails} failed)")
    sys.exit(1 if fails else 0)
