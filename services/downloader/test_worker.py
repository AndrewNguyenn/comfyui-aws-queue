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
