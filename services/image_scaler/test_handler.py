"""Unit tests for the image autoscaler's pure decision logic.

Run: MAX_WORKERS=4 python3 -m pytest services/image_scaler/test_handler.py
(or just `python3 services/image_scaler/test_handler.py`). boto3 clients are
created at import time, so we stub the module before importing the handler.
"""
import importlib.util
import os
import sys
import types

sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *a, **k: None))
os.environ.setdefault("IMAGE_QUEUE_URL", "x")
os.environ.setdefault("CLUSTER", "c")
os.environ.setdefault("SERVICE", "s")
os.environ["MAX_WORKERS"] = "4"

_spec = importlib.util.spec_from_file_location(
    "scaler_handler", os.path.join(os.path.dirname(__file__), "handler.py")
)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def test_step_target_bands():
    cases = {0: 0, 1: 1, 49: 1, 50: 2, 149: 2, 150: 3, 299: 3, 300: 4, 500: 4, 9999: 4}
    for visible, expected in cases.items():
        assert h.step_target(visible) == expected, (visible, h.step_target(visible))


def test_lazy_ramp_up():
    # fresh batches spin up only to the band target, never the whole fleet
    assert h.decide(visible=100, inflight=0, current=0) == 2
    assert h.decide(visible=250, inflight=0, current=2) == 3
    assert h.decide(visible=450, inflight=0, current=3) == 4
    assert h.decide(visible=10, inflight=0, current=0) == 1


def test_sticky_down_holds_until_clear():
    # a 500-batch ramped to 4 holds 4 all the way down while work remains
    assert h.decide(visible=200, inflight=4, current=4) == 4
    assert h.decide(visible=40, inflight=4, current=4) == 4
    assert h.decide(visible=0, inflight=4, current=4) == 4  # drained but busy


def test_release_steps_down_one_per_tick():
    # fully cleared releases gradually (false-empty caps damage to one worker)
    assert h.decide(visible=0, inflight=0, current=4) == 3
    assert h.decide(visible=0, inflight=0, current=3) == 2
    assert h.decide(visible=0, inflight=0, current=1) == 0
    assert h.decide(visible=0, inflight=0, current=0) == 0


def test_new_work_during_release_re_holds():
    # stepped down to 3 after a clear, then a small batch arrives -> hold 3
    assert h.decide(visible=30, inflight=0, current=3) == 3


def test_lowered_cap_shrinks_live_fleet():
    # MAX_WORKERS lowered below current pulls a live fleet down to the new cap
    saved = h.MAX_WORKERS
    h.MAX_WORKERS = 2
    try:
        assert h.decide(visible=400, inflight=0, current=4) == 2
    finally:
        h.MAX_WORKERS = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
