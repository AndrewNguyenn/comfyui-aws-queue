"""Unit tests for the status Lambda's pure derivation logic — specifically the
character extraction that drives the viewer's pending-strip grouping.

Run: python3 services/status/test_handler.py
(or `python3 -m pytest services/status/test_handler.py`). boto3 / botocore
clients are created at import time, so we stub them before importing the
handler.
"""
import importlib.util
import json
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
        # name_(category) qualifier tags are skipped — painting_(object) is a
        # medium tag, not a character; the real character follows
        "from_front BREAK painting_(object), tifa_lockhart, final_fantasy_vii, "
        "brown_hair":
            "tifa_lockhart",
        # a real name_(franchise) parenthetical is still a character (the paren
        # content is a franchise, not a category qualifier)
        "close-up BREAK fubuki_(one_punch_man), green_dress, black_hair":
            "fubuki_(one_punch_man)",
        # "masterwork" is a quality synonym (like masterpiece), skipped
        "masterwork, (masterpiece,_best_quality:1.2), very_awa BREAK rossweisse, "
        "high_school_dxd, grey_hair":
            "rossweisse",
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


def test_extract_prompts_through_concat_chain():
    # Mirrors the real Anima workflow: the positive runs POSITIVE wildcard ->
    # StringConcatenate(trigger_words, POSITIVE) -> RegexReplace -> CLIPTextEncode
    # -> KSampler. The walk must follow the concat's string_a/string_b parts; the
    # negative is wired directly (the case that always worked).
    wf = {
        "3": {"class_type": "ImpactWildcardProcessor", "_meta": {"title": "POSITIVE"},
              "inputs": {"populated_text": "masterwork, tsukino_usagi, blonde_hair, twintails", "mode": "fixed"}},
        "4": {"class_type": "ImpactWildcardProcessor", "_meta": {"title": "NEGATIVE"},
              "inputs": {"populated_text": "worst_quality, bad_hands", "mode": "fixed"}},
        "37": {"class_type": "TriggerWord Toggle (LoraManager)",
               "inputs": {"orinalMessage": "", "trigger_words": ["5", 2]}},
        "46": {"class_type": "StringConcatenate",
               "inputs": {"string_a": ["37", 0], "string_b": ["3", 0], "delimiter": ","}},
        "48": {"class_type": "RegexReplace",
               "inputs": {"string": ["46", 0], "regex_pattern": "^,|,$", "replace": ""}},
        "54": {"class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
               "inputs": {"text": ["48", 0], "clip": ["18", 0]}},
        "55": {"class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
               "inputs": {"text": ["4", 0], "clip": ["18", 0]}},
        "6": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
              "inputs": {"positive": ["54", 0], "negative": ["55", 0]}},
    }
    secs = {s["label"]: s["text"] for s in h._extract_prompts(wf)}
    assert secs.get("Positive", "").startswith("masterwork, tsukino_usagi"), secs
    assert secs.get("Negative", "").startswith("worst_quality"), secs


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


def test_serialize_job_set_scene_on_detail_path_only():
    item = {
        "job_id": {"S": "j2"},
        "type": {"S": "image"},
        "status": {"S": "complete"},
        "workflow_json": {"S": "{}"},
        "set_name": {"S": "Single - Bikini Car Wash"},
        "scene_name": {"S": "Customer's Trophy Shot"},
    }
    detail = h._serialize_job(item, lite=False)
    assert detail["set_name"] == "Single - Bikini Car Wash"
    assert detail["scene_name"] == "Customer's Trophy Shot"
    # lite list rows come off the jobs-by-status GSI, which doesn't project
    # these attrs — the serializer must not emit them there
    lite = h._serialize_job(item, lite=True)
    assert "set_name" not in lite and "scene_name" not in lite
    # editor submissions / old rows carry no source attrs → fields omitted
    bare = dict(item)
    del bare["set_name"], bare["scene_name"]
    bare_detail = h._serialize_job(bare, lite=False)
    assert "set_name" not in bare_detail and "scene_name" not in bare_detail


def test_serialize_job_prefers_denormalized_attrs_over_derivation():
    # New rows carry model/character/subject as stored attrs (denormalized at
    # dispatch). _serialize_job must use them verbatim and NOT re-derive from
    # workflow_json — that is what lets the GSI drop the graph. The stored values
    # here deliberately disagree with what the graph would yield, to prove it.
    item = {
        "job_id": {"S": "j2"}, "type": {"S": "image"}, "status": {"S": "queued"},
        "model": {"S": "stored_model"},
        "character": {"S": "stored_char"},
        "subject": {"S": "stored_subject"},
        "workflow_json": {"S": '{"20": {"class_type": "CLIPTextEncode", '
                               '"_meta": {"title": "Positive"}, '
                               '"inputs": {"text": "narberal_gamma, overlord"}}, '
                               '"30": {"class_type": "KSampler", '
                               '"inputs": {"positive": ["20", 0]}}}'},
    }
    out = h._serialize_job(item, lite=True)
    assert out["model"] == "stored_model"
    assert out["character"] == "stored_char"
    assert out["subject"] == "stored_subject"


def test_serialize_job_falls_back_for_old_rows():
    # Old row, lite list, no stored attrs AND no workflow_json on the reprojected
    # GSI → blank model/character, never an error (cosmetic; self-heals on backfill).
    old = {"job_id": {"S": "j3"}, "type": {"S": "image"}, "status": {"S": "queued"}}
    out_old = h._serialize_job(old, lite=True)
    assert out_old["model"] == ""
    assert out_old["character"] == ""

    # Old row on the DETAIL path (base-table GetItem) still has workflow_json, so
    # the fallback derives the model for every row regardless of backfill state.
    old_detail = {
        "job_id": {"S": "j4"}, "type": {"S": "image"}, "status": {"S": "complete"},
        "workflow_json": {"S": '{"1": {"class_type": "CheckpointLoaderSimple", '
                               '"inputs": {"ckpt_name": "catpony.safetensors"}}}'},
    }
    assert h._serialize_job(old_detail, lite=False)["model"] == "catpony"


def test_group_key_mirrors_frontend_grouping():
    # (type, model||type||'job', character||subject) — must match groupQueue() in app.js.
    assert h._group_key({"type": "image", "model": "catpony", "character": "aqua", "subject": ""}) \
        == ("image", "catpony", "aqua")
    # no character → the subject hint is the key part (unnamed-figure grouping)
    assert h._group_key({"type": "image", "model": "m", "character": "", "subject": "blue_hair figure"}) \
        == ("image", "m", "blue_hair figure")
    # no model → falls back to type, then 'job'
    assert h._group_key({"type": "video", "model": "", "character": "rem", "subject": ""}) \
        == ("video", "video", "rem")
    assert h._group_key({"type": "", "model": "", "character": "", "subject": ""}) == ("", "job", "")


def test_cancel_group_cancels_the_whole_stack_not_a_sample():
    # The bug this fixes: cancelling a group must hit EVERY matching queued job,
    # not just the ids the viewer loaded. Patch the query + serializer + ddb.
    saved = (h.ddb, h._query_newest, h._serialize_job)
    cancelled_ids = []
    fake = types.SimpleNamespace(
        update_item=lambda **kw: cancelled_ids.append(kw["Key"]["job_id"]["S"]))
    # 6 queued jobs: 4 in the target stack (aqua), 2 in another (rem).
    chars = {"j0": "aqua", "j1": "rem", "j2": "aqua", "j3": "aqua", "j4": "rem", "j5": "aqua"}
    items = [{"job_id": {"S": jid}} for jid in chars]
    try:
        h.ddb = fake
        h._query_newest = lambda status, n: items if status == "queued" else []
        h._serialize_job = lambda it, lite=False: {
            "job_id": it["job_id"]["S"], "type": "image", "model": "catpony",
            "character": chars[it["job_id"]["S"]], "subject": "",
        }
        resp = h._cancel_group({"body": json.dumps(
            {"type": "image", "model": "catpony", "character": "aqua", "subject": ""})})
    finally:
        h.ddb, h._query_newest, h._serialize_job = saved
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200, resp
    assert body["cancelled"] == 4, body                      # all 4 aqua — not a sample
    assert sorted(cancelled_ids) == ["j0", "j2", "j3", "j5"]  # exactly the aqua ids


def test_cancel_group_rejects_empty_identity():
    resp = h._cancel_group({"body": json.dumps({"type": "", "character": "", "subject": ""})})
    assert resp["statusCode"] == 400, resp


# ---------------------------------------------------------------------------
# /jobs keyset (cursor) pagination
# ---------------------------------------------------------------------------
class _FakeGSI:
    """Simulates a Query on the jobs-by-status GSI over an in-memory dataset:
    a status partition sorted by created_at (desc via ScanIndexForward=False,
    job_id as the tiebreak for a total order), Limit, and ExclusiveStartKey
    resume. `page_cap` models DynamoDB's own ~1 MB short-page truncation so
    _query_page's inner re-query loop is exercised. Tracks rows_read (to prove
    O(N), no re-scan) and call count."""

    def __init__(self, rows, page_cap=10_000):
        self.rows = rows
        self.page_cap = page_cap
        self.rows_read = 0
        self.calls = 0

    def query(self, **kw):
        self.calls += 1
        assert kw["IndexName"] == "jobs-by-status"
        assert kw["ScanIndexForward"] is False
        status = kw["ExpressionAttributeValues"][":s"]["S"]
        pool = [r for r in self.rows if r["status"]["S"] == status]
        pool.sort(key=lambda r: (r["created_at"]["S"], r["job_id"]["S"]), reverse=True)
        start = kw.get("ExclusiveStartKey")
        if start:
            sk = (start["created_at"]["S"], start["job_id"]["S"])
            idx = next((i for i, r in enumerate(pool)
                        if (r["created_at"]["S"], r["job_id"]["S"]) == sk), None)
            pool = pool[idx + 1:] if idx is not None else []
        n = min(kw["Limit"], self.page_cap)
        page = pool[:n]
        self.rows_read += len(page)
        out = {"Items": page}
        if len(page) < len(pool):  # more rows remain → hand back a resume key
            last = page[-1]
            out["LastEvaluatedKey"] = {"status": last["status"],
                                       "created_at": last["created_at"],
                                       "job_id": last["job_id"]}
        return out


def _row(job_id, status, created_at):
    return {"job_id": {"S": job_id}, "type": {"S": "image"},
            "status": {"S": status}, "created_at": {"S": created_at},
            "output_keys": {"S": json.dumps([f"out/{job_id}.png"])}}


def _rows(n, status="complete"):
    # created_at sorts lexicographically with i, so job-000NN is the newest.
    return [_row(f"job-{i:05d}", status, f"2026-06-11T00:00:{i:05d}")
            for i in range(n)]


def _sweep(fake, qs_base):
    """Page the whole history via the cursor, returning the job_ids in order."""
    saved = h.ddb
    seen, cursor, pages = [], None, 0
    try:
        h.ddb = fake
        while True:
            qs = dict(qs_base)
            if cursor:
                qs["cursor"] = cursor
            body = json.loads(h._list_jobs({"queryStringParameters": qs})["body"])
            seen.extend(j["job_id"] for j in body["jobs"])
            cursor = body["next_cursor"]
            pages += 1
            assert pages < 1000, "cursor sweep did not terminate"
            if not cursor:
                break
    finally:
        h.ddb = saved
    return seen


def test_keyset_pagination_covers_every_row_exactly_once():
    fake = _FakeGSI(_rows(1000))
    seen = _sweep(fake, {"status": "complete", "limit": "300"})
    assert len(seen) == 1000
    assert len(set(seen)) == 1000             # no page overlap / duplicates
    assert seen[0] == "job-00999"             # newest first
    assert seen[-1] == "job-00000"            # oldest last, contiguous
    # The whole point: each row is read exactly once across the sweep — the old
    # offset/limit path re-read everything above each page (O(N^2)).
    assert fake.rows_read == 1000, fake.rows_read


def test_keyset_inner_loop_fills_page_across_short_dynamo_pages():
    # DynamoDB hands back at most 120 rows/query; _query_page must re-query to
    # fill a 300-row API page, threading the LastEvaluatedKey correctly.
    fake = _FakeGSI(_rows(500), page_cap=120)
    seen = _sweep(fake, {"status": "complete", "limit": "300"})
    assert len(seen) == 500 and len(set(seen)) == 500
    assert seen[0] == "job-00499"
    assert fake.rows_read == 500              # still O(N) despite short pages
    assert fake.calls > 500 / 300             # >1 ddb query per API page


def test_bad_cursor_returns_400_without_querying():
    fake = _FakeGSI(_rows(10))
    saved = h.ddb
    try:
        h.ddb = fake
        resp = h._list_jobs({"queryStringParameters": {
            "status": "complete", "limit": "150", "cursor": "%%%not-base64%%%"}})
    finally:
        h.ddb = saved
    assert resp["statusCode"] == 400, resp
    assert fake.calls == 0                    # rejected before any DynamoDB read


def test_cursor_status_mismatch_returns_400():
    # a token minted for `complete` must not resume a `running` query.
    cur = h._encode_cursor({"status": {"S": "complete"},
                            "created_at": {"S": "2026-06-11T00:00:00500"},
                            "job_id": {"S": "job-00500"}})
    fake = _FakeGSI(_rows(10, status="running"))
    saved = h.ddb
    try:
        h.ddb = fake
        resp = h._list_jobs({"queryStringParameters": {
            "status": "running", "limit": "150", "cursor": cur}})
    finally:
        h.ddb = saved
    assert resp["statusCode"] == 400, resp


def test_malformed_cursor_shape_returns_400_not_500():
    # A token with the right status but a wrong key shape must be rejected by
    # _decode_cursor (a clean 400) rather than passed to DynamoDB as an
    # ExclusiveStartKey, where it would raise ValidationException → a 500.
    bad = h._encode_cursor({"status": {"S": "complete"}, "junk": {"S": "x"}})
    fake = _FakeGSI(_rows(10))
    saved = h.ddb
    try:
        h.ddb = fake
        resp = h._list_jobs({"queryStringParameters": {
            "status": "complete", "limit": "150", "cursor": bad}})
    finally:
        h.ddb = saved
    assert resp["statusCode"] == 400, resp
    assert fake.calls == 0


def test_single_status_offset_without_cursor_keeps_legacy_slice():
    # A stale client still paging by offset (no cursor) must keep working.
    fake = _FakeGSI(_rows(10))
    saved = h.ddb
    try:
        h.ddb = fake
        body = json.loads(h._list_jobs({"queryStringParameters": {
            "status": "complete", "limit": "3", "offset": "3"}})["body"])
    finally:
        h.ddb = saved
    ids = [j["job_id"] for j in body["jobs"]]
    assert ids == ["job-00006", "job-00005", "job-00004"], ids  # newest, skip 3
    assert body["next_cursor"] is None        # legacy path exposes no cursor
    assert "offset" in body and "total" in body


def test_multi_status_uses_legacy_offset_merge():
    rows = ([_row(f"c{i}", "complete", f"2026-06-11T00:00:0{i}") for i in range(3)] +
            [_row(f"f{i}", "failed", f"2026-06-11T00:00:1{i}") for i in range(3)])
    fake = _FakeGSI(rows)
    saved = h.ddb
    try:
        h.ddb = fake
        body = json.loads(h._list_jobs({"queryStringParameters": {
            "status": "complete,failed", "limit": "10"}})["body"])
    finally:
        h.ddb = saved
    assert body["next_cursor"] is None
    assert "offset" in body and "total" in body
    ids = [j["job_id"] for j in body["jobs"]]
    assert len(ids) == 6
    assert ids[0] == "f2"                      # newest across both statuses
    assert ids[-1] == "c0"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
