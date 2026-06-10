"""Unit tests for set/scene provenance on POST /prompt (body.source).

The tag generator sends `source: {set, scene}` with each queued job; the
dispatcher persists them as `set_name`/`scene_name` on the job record (the
status Lambda returns them on the /jobs/{id} detail path for the viewer's
modal). These tests stub boto3 and capture the DDB put_item to assert the
mapping.

Run: python3 services/dispatcher/test_source.py
(or `python3 -m pytest services/dispatcher/test_source.py`).
"""
import importlib.util
import json
import os
import sys
import types


class _Recorder:
    """Stands in for every boto3 client — records writes, answers reads inertly."""

    def __init__(self):
        self.put_items = []
        self.sent = []

    def put_item(self, **kw):
        self.put_items.append(kw)

    def send_message(self, **kw):
        self.sent.append(kw)

    def update_item(self, **kw):
        pass


_recorder = _Recorder()

sys.modules.setdefault(
    "boto3", types.SimpleNamespace(client=lambda *a, **k: _recorder)
)
_botocore = types.ModuleType("botocore")
_exc = types.ModuleType("botocore.exceptions")
_exc.ClientError = type("ClientError", (Exception,), {})
_botocore.exceptions = _exc
sys.modules.setdefault("botocore", _botocore)
sys.modules.setdefault("botocore.exceptions", _exc)

for _k in ("JOBS_TABLE", "MODELS_TABLE", "OBJECT_INFO_TABLE",
           "IMAGE_QUEUE_URL", "VIDEO_QUEUE_URL"):
    os.environ.setdefault(_k, "x")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # handler's sibling imports (anima, extract, …)
_spec = importlib.util.spec_from_file_location(
    "dispatcher_handler", os.path.join(_HERE, "handler.py")
)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

# Minimal non-Anima workflow: enough graph for the denormalizers to walk.
_WF = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "catponyDark_aniIlV40.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl"}},
}


def _post(body: dict) -> dict:
    _recorder.put_items.clear()
    _recorder.sent.clear()
    r = h._post_prompt({"body": json.dumps(body)})
    assert r["statusCode"] == 200, r
    assert len(_recorder.put_items) == 1
    return _recorder.put_items[0]["Item"]


def test_source_set_and_scene_persisted():
    item = _post({"prompt": _WF, "type": "image",
                  "source": {"set": "Single - Bikini Car Wash",
                             "scene": "Customer's Trophy Shot"}})
    assert item["set_name"] == {"S": "Single - Bikini Car Wash"}
    assert item["scene_name"] == {"S": "Customer's Trophy Shot"}


def test_source_absent_leaves_no_attrs():
    item = _post({"prompt": _WF, "type": "image"})
    assert "set_name" not in item
    assert "scene_name" not in item


def test_source_partial_and_blank_fields():
    # blank set (unsaved set) → only the scene lands; whitespace is stripped
    item = _post({"prompt": _WF, "type": "image",
                  "source": {"set": "  ", "scene": " Opening Tease "}})
    assert "set_name" not in item
    assert item["scene_name"] == {"S": "Opening Tease"}


def test_source_non_dict_ignored():
    for bad in ("a-string", 42, ["set", "scene"], None):
        item = _post({"prompt": _WF, "type": "image", "source": bad})
        assert "set_name" not in item
        assert "scene_name" not in item


def test_source_values_capped_and_stringified():
    item = _post({"prompt": _WF, "type": "image",
                  "source": {"set": "x" * 500, "scene": 7}})
    assert item["set_name"] == {"S": "x" * 200}
    assert item["scene_name"] == {"S": "7"}


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
