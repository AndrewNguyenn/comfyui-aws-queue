"""Tests for the file-selection logic (the file_name selector)."""
import os
os.environ.setdefault("MODELS_BUCKET", "b")
os.environ.setdefault("MODELS_TABLE", "m")
os.environ.setdefault("DOWNLOADS_TABLE", "d")
import worker

_META = {"files": [
    {"name": "nyaIrisAnima_baseV10.safetensors", "primary": True, "sizeKB": 4000000},
    {"name": "nyaIrisAnima_baseV10_txt.safetensors", "sizeKB": 1100000},
    {"name": "qwen_image_vae.safetensors", "sizeKB": 240000},
]}


def test_pick_file_by_name_selects_nonprimary():
    f = worker._pick_file(_META, "nyaIrisAnima_baseV10_txt.safetensors")
    assert f["name"] == "nyaIrisAnima_baseV10_txt.safetensors"


def test_pick_file_by_name_case_insensitive():
    f = worker._pick_file(_META, "NYAIRISANIMA_BASEV10_TXT.SAFETENSORS")
    assert f["name"] == "nyaIrisAnima_baseV10_txt.safetensors"


def test_pick_file_no_name_falls_back_to_primary():
    f = worker._pick_file(_META, "")
    assert f["name"] == "nyaIrisAnima_baseV10.safetensors"


def test_pick_file_unknown_name_raises():
    try:
        worker._pick_file(_META, "does_not_exist.safetensors")
    except RuntimeError as e:
        assert "not in version files" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


# ---- _write_loramanager_metadata (lora-manager OOM pre-seed) ----
import json as _json


class _FakeS3:
    def __init__(self):
        self.calls = []

    def put_object(self, **kw):
        self.calls.append(kw)


def _seed(file_meta, model_type, s3_key, size, meta):
    fake = _FakeS3()
    orig = worker.s3
    worker.s3 = fake
    try:
        worker._write_loramanager_metadata(file_meta, model_type, s3_key, size, meta)
    finally:
        worker.s3 = orig
    return fake.calls


_LORA_FILE = {"name": "My.Lora_v1.safetensors", "sizeKB": 150000,
              "hashes": {"SHA256": "ABC123" + "0" * 58}}  # 64-char, uppercase
_VMETA = {"baseModel": "Illustrious", "model": {"name": "My Cool Lora"},
          "images": [{"url": "https://img/preview.png"}]}


def test_seed_writes_valid_lorametadata_schema():
    calls = _seed(_LORA_FILE, "lora", "lora/My.Lora_v1.safetensors", 153600000, _VMETA)
    assert len(calls) == 1
    # metadata key strips only the final extension (matches lora-manager splitext)
    assert calls[0]["Key"] == "lora/My.Lora_v1.metadata.json"
    m = _json.loads(calls[0]["Body"])
    for k in ("file_name", "model_name", "file_path", "size", "modified",
              "sha256", "base_model", "preview_url"):
        assert k in m, f"missing required LoraMetadata field: {k}"
    assert m["sha256"] == ("abc123" + "0" * 58)          # lowercased
    assert m["sha256"]                                    # non-empty -> no re-hash
    assert m["file_name"] == "My.Lora_v1"
    assert m["size"] == 153600000
    assert m["base_model"] == "Illustrious"
    assert m["hash_status"] == "completed"


def test_seed_skips_unscanned_types():
    # lora-manager only scans loras/checkpoints/diffusion_models — others are skipped
    assert _seed(_LORA_FILE, "vae", "vae/X.safetensors", 1, _VMETA) == []
    assert _seed(_LORA_FILE, "controlnet", "controlnet/X.safetensors", 1, _VMETA) == []
    assert _seed(_LORA_FILE, "embedding", "embedding/X.safetensors", 1, _VMETA) == []


def test_seed_checkpoint_uses_checkpoints_dir_and_subtype():
    f = {"name": "myCkpt_v1.safetensors", "hashes": {"SHA256": "DEAD" + "0" * 60}}
    calls = _seed(f, "checkpoint", "checkpoint/myCkpt_v1.safetensors", 7_000_000_000, _VMETA)
    assert len(calls) == 1
    assert calls[0]["Key"] == "checkpoint/myCkpt_v1.metadata.json"
    m = _json.loads(calls[0]["Body"])
    assert m["file_path"] == "/opt/comfy/models/checkpoints/myCkpt_v1.safetensors"
    assert m["sub_type"] == "checkpoint"
    assert m["sha256"] == ("dead" + "0" * 60)


def test_seed_diffusion_models_subtype():
    f = {"name": "anima_base.safetensors", "hashes": {"SHA256": "BEEF" + "0" * 60}}
    calls = _seed(f, "diffusion_models", "diffusion_models/anima_base.safetensors", 5_000_000_000, _VMETA)
    assert len(calls) == 1
    assert calls[0]["Key"] == "diffusion_models/anima_base.metadata.json"
    m = _json.loads(calls[0]["Body"])
    assert m["file_path"] == "/opt/comfy/models/diffusion_models/anima_base.safetensors"
    assert m["sub_type"] == "diffusion_model"


def test_seed_skips_when_no_civitai_sha256():
    f = {"name": "X.safetensors", "hashes": {}}
    assert _seed(f, "lora", "lora/X.safetensors", 1, _VMETA) == []
    f2 = {"name": "X.safetensors"}  # no hashes key at all
    assert _seed(f2, "lora", "lora/X.safetensors", 1, _VMETA) == []


if __name__ == "__main__":
    import sys
    _fns = [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]
    _fail = 0
    for _fn in _fns:
        try:
            _fn()
            print(f"PASS {_fn.__name__}")
        except Exception as _e:  # noqa: BLE001
            _fail += 1
            print(f"FAIL {_fn.__name__}: {_e!r}")
    print(f"--- {len(_fns) - _fail}/{len(_fns)} passed ---")
    sys.exit(1 if _fail else 0)
