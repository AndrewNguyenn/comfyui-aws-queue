"""Unit tests for the image autoscaler's pure decision logic.

Run: python3 -m pytest services/image_scaler/test_handler.py
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
os.environ["MAX_WORKERS"] = "3"  # matches config.ts scaling.imageMax

_spec = importlib.util.spec_from_file_location(
    "scaler_handler", os.path.join(os.path.dirname(__file__), "handler.py")
)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def test_step_target_bands():
    # 1 queued job -> 1 worker (not 2): <12→1, 12-39→2, ≥40→MAX_WORKERS(3)
    cases = {0: 0, 1: 1, 11: 1, 12: 2, 39: 2, 40: 3, 100: 3, 300: 3, 9999: 3}
    for visible, expected in cases.items():
        assert h.step_target(visible) == expected, (visible, h.step_target(visible))


def test_lazy_ramp_up():
    # fresh batches ramp to the band target
    assert h.decide(visible=5, inflight=0, current=0) == 1    # trickle
    assert h.decide(visible=20, inflight=0, current=0) == 2   # moderate
    assert h.decide(visible=50, inflight=0, current=0) == 3   # backlog -> full fleet
    assert h.decide(visible=15, inflight=0, current=1) == 2   # ramps 1 -> 2


def test_sticky_down_holds_until_clear():
    # a 40+ batch ramped to 3 (MAX) holds 3 all the way down while work remains
    assert h.decide(visible=100, inflight=3, current=3) == 3
    assert h.decide(visible=20, inflight=3, current=3) == 3
    assert h.decide(visible=0, inflight=3, current=3) == 3  # drained but busy


def test_release_steps_down_one_per_tick():
    # fully cleared releases gradually (false-empty caps damage to one worker)
    assert h.decide(visible=0, inflight=0, current=3) == 2
    assert h.decide(visible=0, inflight=0, current=2) == 1
    assert h.decide(visible=0, inflight=0, current=1) == 0
    assert h.decide(visible=0, inflight=0, current=0) == 0


def test_new_work_during_release_re_holds():
    # stepped down to 2 after a clear, then a small batch arrives -> hold 2
    # (current 2 > step_target(5)=1, so stickiness keeps the live fleet)
    assert h.decide(visible=5, inflight=0, current=2) == 2


def test_lowered_cap_shrinks_live_fleet():
    # MAX_WORKERS lowered below current pulls a live fleet down to the new cap
    saved = h.MAX_WORKERS
    h.MAX_WORKERS = 1
    try:
        assert h.decide(visible=400, inflight=0, current=3) == 1
    finally:
        h.MAX_WORKERS = saved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
