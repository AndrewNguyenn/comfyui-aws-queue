"""Tests for the HuggingFace download source (URL validation + HEAD resolve)."""
import os
import sys
import types

os.environ.setdefault("MODELS_BUCKET", "b")
os.environ.setdefault("MODELS_TABLE", "m")
os.environ.setdefault("DOWNLOADS_TABLE", "d")
os.environ.setdefault("DOWNLOAD_WORKER_FN", "f")

import worker


# --- URL validation (kickoff) ---------------------------------------------

def _kickoff():
    sys.modules.setdefault(
        "boto3_stub_marker", types.SimpleNamespace()
    )
    import kickoff
    return kickoff


def test_resolve_and_blob_urls_are_accepted():
    k = _kickoff()
    for u in (
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/blob/main/SCAIL/model.safetensors",
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/x.onnx",
    ):
        assert k._looks_like_hf_url(u), u


def test_repo_root_and_foreign_hosts_are_rejected():
    k = _kickoff()
    for u in (
        "https://huggingface.co/Kijai/WanVideo_comfy",          # whole repo, not a file
        "https://huggingface.co/Kijai/WanVideo_comfy/tree/main",
        "https://evil.com/Kijai/WanVideo_comfy/resolve/main/x.safetensors",
        "https://civitai.com/models/123",
        "",
    ):
        assert not k._looks_like_hf_url(u), u


# --- HEAD resolution (worker) ---------------------------------------------

class _Resp:
    def __init__(self, headers, status=200):
        self.headers = headers
        self.status = status


def _patch_head(monkeypatch, headers, status=200):
    seen = {}

    def fake_request(method, url, **kw):
        seen["method"], seen["url"] = method, url
        seen["redirect"] = kw.get("redirect")
        return _Resp(headers, status)

    monkeypatch.setattr(worker.http, "request", fake_request)
    return seen


_HEADERS = {
    "x-linked-size": "16400000000",
    "x-linked-etag": '"ABC123DEF"',
    "content-length": "1028",
}


def test_uses_linked_size_not_pointer_content_length(monkeypatch):
    # content-length on an LFS file is the ~1 KB pointer. Trusting it would
    # report a 15 GiB model as 1 KB and break every progress readout.
    _patch_head(monkeypatch, _HEADERS)
    m = worker._hf_file_meta(
        "https://huggingface.co/o/r/resolve/main/big.safetensors", ""
    )
    assert int(m["sizeKB"] * 1024) == 16400000000


def test_extracts_sha256_from_linked_etag(monkeypatch):
    _patch_head(monkeypatch, _HEADERS)
    m = worker._hf_file_meta("https://huggingface.co/o/r/resolve/main/x.safetensors", "")
    # quotes stripped, lowercased — the shape lora-manager's seed expects
    assert m["hashes"]["SHA256"] == "abc123def"


def test_blob_urls_are_normalised_to_resolve(monkeypatch):
    seen = _patch_head(monkeypatch, _HEADERS)
    worker._hf_file_meta(
        "https://huggingface.co/Kijai/WanVideo_comfy/blob/main/SCAIL/m.safetensors", ""
    )
    assert "/resolve/main/SCAIL/m.safetensors" in seen["url"]
    assert "/blob/" not in seen["url"]


def test_filename_is_the_basename_of_a_nested_path(monkeypatch):
    _patch_head(monkeypatch, _HEADERS)
    m = worker._hf_file_meta(
        "https://huggingface.co/o/r/resolve/main/onnx/wholebody/vitpose-l.onnx", ""
    )
    assert m["name"] == "vitpose-l.onnx"


def test_missing_size_raises_rather_than_uploading_garbage(monkeypatch):
    _patch_head(monkeypatch, {"x-linked-size": "0"})
    try:
        worker._hf_file_meta("https://huggingface.co/o/r/resolve/main/x.safetensors", "")
    except RuntimeError as e:
        assert "size" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_http_error_mentions_the_token_secret(monkeypatch):
    _patch_head(monkeypatch, {}, status=401)
    try:
        worker._hf_file_meta("https://huggingface.co/o/r/resolve/main/x.safetensors", "")
    except RuntimeError as e:
        assert "401" in str(e) and "huggingface/api-token" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_missing_sha_yields_no_hashes_entry(monkeypatch):
    _patch_head(monkeypatch, {"x-linked-size": "100"})
    m = worker._hf_file_meta("https://huggingface.co/o/r/resolve/main/x.safetensors", "")
    assert m["hashes"] == {}


def test_head_does_not_follow_redirects(monkeypatch):
    # huggingface.co puts x-linked-etag (the LFS sha256) on the 302 hop; its CDN's
    # final 200 drops it and keeps only content-length. Following the redirect
    # loses the hash silently — size still looks right — and the fleet then
    # re-hashes multi-GB models on every boot.
    seen = _patch_head(monkeypatch, _HEADERS, status=302)
    m = worker._hf_file_meta("https://huggingface.co/o/r/resolve/main/x.safetensors", "")
    assert seen["redirect"] is False
    assert m["hashes"]["SHA256"] == "abc123def"


def test_a_302_is_not_treated_as_an_error(monkeypatch):
    _patch_head(monkeypatch, _HEADERS, status=302)
    m = worker._hf_file_meta("https://huggingface.co/o/r/resolve/main/x.safetensors", "")
    assert int(m["sizeKB"] * 1024) == 16400000000


# --- parallel ranged transfer ---------------------------------------------

class _S3Rec:
    def __init__(self):
        self.parts = {}
        self.completed = None
        self.aborted = False

    def create_multipart_upload(self, **kw):
        return {"UploadId": "u1"}

    def upload_part(self, **kw):
        self.parts[kw["PartNumber"]] = kw["Body"]
        return {"ETag": f'"etag{kw["PartNumber"]}"'}

    def complete_multipart_upload(self, **kw):
        self.completed = kw["MultipartUpload"]["Parts"]

    def abort_multipart_upload(self, **kw):
        self.aborted = True


def _setup_ranged(monkeypatch, total, status=206, short=False):
    rec = _S3Rec()
    monkeypatch.setattr(worker, "s3", rec)
    monkeypatch.setattr(worker, "_set_bytes_done", lambda *a, **k: None)

    def fake_request(method, url, headers=None, **kw):
        rng = headers["Range"].split("=")[1]
        first, last = (int(x) for x in rng.split("-"))
        n = last - first + 1
        if short:
            n -= 1
        return _Resp({}, status)._with(b"\0" * n)

    def _with(self, data):
        self.data = data
        return self

    _Resp._with = _with
    monkeypatch.setattr(worker.http, "request", fake_request)
    return rec


def test_ranged_transfer_covers_every_byte_exactly_once(monkeypatch):
    total = worker._RANGE_PART_SIZE * 3 + 12345
    rec = _setup_ranged(monkeypatch, total)
    got = worker._stream_to_s3_ranged("https://hf/x", "", "k", total, "d")
    assert got == total
    assert sum(len(b) for b in rec.parts.values()) == total
    # part numbers are 1-based and contiguous
    assert sorted(rec.parts) == list(range(1, len(rec.parts) + 1))
    assert [p["PartNumber"] for p in rec.completed] == sorted(rec.parts)


def test_a_200_response_aborts_instead_of_corrupting(monkeypatch):
    # A 200 means the origin ignored Range and is sending the WHOLE file for
    # every part; assembling those would produce a corrupt object.
    total = worker._RANGE_PART_SIZE * 2
    rec = _setup_ranged(monkeypatch, total, status=200)
    monkeypatch.setattr(worker.time, "sleep", lambda *_: None)
    try:
        worker._stream_to_s3_ranged("https://hf/x", "", "k", total, "d")
    except Exception:
        pass
    else:
        raise AssertionError("expected failure on a non-206 response")
    assert rec.completed is None
    assert rec.aborted


def test_a_short_part_aborts_instead_of_completing(monkeypatch):
    total = worker._RANGE_PART_SIZE * 2
    rec = _setup_ranged(monkeypatch, total, short=True)
    monkeypatch.setattr(worker.time, "sleep", lambda *_: None)
    try:
        worker._stream_to_s3_ranged("https://hf/x", "", "k", total, "d")
    except Exception:
        pass
    else:
        raise AssertionError("expected failure on a short part")
    assert rec.completed is None and rec.aborted


def test_unknown_length_is_refused(monkeypatch):
    _setup_ranged(monkeypatch, 0)
    try:
        worker._stream_to_s3_ranged("https://hf/x", "", "k", 0, "d")
    except RuntimeError as e:
        assert "content length" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
