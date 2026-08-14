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
