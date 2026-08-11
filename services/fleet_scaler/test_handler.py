"""Unit tests for the fleet autoscaler's pure decision logic (image + video).

Run: python3 -m pytest services/fleet_scaler/test_handler.py
(or just `python3 services/fleet_scaler/test_handler.py`). boto3 clients are
created at import time, so we stub the module before importing the handler.
"""
import importlib.util
import os
import sys
import types

sys.modules.setdefault("boto3", types.SimpleNamespace(client=lambda *a, **k: None))
os.environ.setdefault("CLUSTER", "c")
os.environ.setdefault("IMAGE_QUEUE_URL", "x-image")
os.environ.setdefault("IMAGE_SERVICE", "s-image")
os.environ["IMAGE_MAX_WORKERS"] = "3"  # matches config.ts scaling.imageMax
os.environ.setdefault("VIDEO_QUEUE_URL", "x-video")
os.environ.setdefault("VIDEO_SERVICE", "s-video")
os.environ["VIDEO_MAX_WORKERS"] = "3"  # matches config.ts scaling.videoMax

_spec = importlib.util.spec_from_file_location(
    "scaler_handler", os.path.join(os.path.dirname(__file__), "handler.py")
)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


# ---- image bands: 1 queued job -> 1 worker (not 2) ----

def test_image_step_target_bands():
    # <12→1, 12-39→2, ≥40→MAX_WORKERS(3)
    cases = {0: 0, 1: 1, 11: 1, 12: 2, 39: 2, 40: 3, 100: 3, 300: 3, 9999: 3}
    for visible, expected in cases.items():
        got = h.step_target(visible, h.IMAGE_BANDS, 3)
        assert got == expected, (visible, got)


def test_image_lazy_ramp_up():
    # fresh batches ramp to the band target
    assert h.decide(5, 0, 0, h.IMAGE_BANDS, 3) == 1    # trickle
    assert h.decide(20, 0, 0, h.IMAGE_BANDS, 3) == 2   # moderate
    assert h.decide(50, 0, 0, h.IMAGE_BANDS, 3) == 3   # backlog -> full fleet
    assert h.decide(15, 0, 1, h.IMAGE_BANDS, 3) == 2   # ramps 1 -> 2


# ---- video bands: single queued job also -> 1 worker immediately (video
# jobs are long-running, not worth batching) ----

def test_video_step_target_bands():
    # 0→0, 1→1, 2-5→2, ≥6→MAX_WORKERS(3)
    cases = {0: 0, 1: 1, 2: 2, 5: 2, 6: 3, 50: 3}
    for visible, expected in cases.items():
        got = h.step_target(visible, h.VIDEO_BANDS, 3)
        assert got == expected, (visible, got)


def test_video_lazy_ramp_up():
    assert h.decide(1, 0, 0, h.VIDEO_BANDS, 3) == 1
    assert h.decide(4, 0, 0, h.VIDEO_BANDS, 3) == 2
    assert h.decide(10, 0, 0, h.VIDEO_BANDS, 3) == 3


# ---- shared decide() behavior, band-agnostic (exercised via image bands) ----

def test_sticky_down_holds_until_clear():
    # a 40+ batch ramped to 3 (MAX) holds 3 all the way down while work remains
    assert h.decide(100, 3, 3, h.IMAGE_BANDS, 3) == 3
    assert h.decide(20, 3, 3, h.IMAGE_BANDS, 3) == 3
    assert h.decide(0, 3, 3, h.IMAGE_BANDS, 3) == 3  # drained but busy


def test_release_steps_down_one_per_tick():
    # fully cleared releases gradually (false-empty caps damage to one worker)
    assert h.decide(0, 0, 3, h.IMAGE_BANDS, 3) == 2
    assert h.decide(0, 0, 2, h.IMAGE_BANDS, 3) == 1
    assert h.decide(0, 0, 1, h.IMAGE_BANDS, 3) == 0
    assert h.decide(0, 0, 0, h.IMAGE_BANDS, 3) == 0


def test_new_work_during_release_re_holds():
    # stepped down to 2 after a clear, then a small batch arrives -> hold 2
    # (current 2 > step_target(5)=1, so stickiness keeps the live fleet)
    assert h.decide(5, 0, 2, h.IMAGE_BANDS, 3) == 2


def test_lowered_cap_shrinks_live_fleet():
    # a lowered max_workers pulls a live fleet down to the new cap
    assert h.decide(400, 0, 3, h.IMAGE_BANDS, 1) == 1


# ---- FLEETS wiring + per-fleet isolation ----

def test_fleets_env_wiring():
    names = {f["name"] for f in h.FLEETS}
    assert names == {"image", "video"}
    by_name = {f["name"]: f for f in h.FLEETS}
    assert by_name["image"]["queue_url"] == "x-image"
    assert by_name["image"]["service"] == "s-image"
    assert by_name["image"]["bands"] is h.IMAGE_BANDS
    assert by_name["video"]["queue_url"] == "x-video"
    assert by_name["video"]["service"] == "s-video"
    assert by_name["video"]["bands"] is h.VIDEO_BANDS


def test_one_fleet_error_does_not_block_the_other():
    # stub sqs/ecs: image fleet's GetQueueAttributes blows up, video's succeeds
    class FakeSqs:
        def get_queue_attributes(self, QueueUrl, AttributeNames):
            if QueueUrl == "x-image":
                raise RuntimeError("sqs boom")
            return {"Attributes": {"ApproximateNumberOfMessages": "0",
                                    "ApproximateNumberOfMessagesNotVisible": "0"}}

    class FakeEcs:
        def describe_services(self, cluster, services):
            return {"services": [{"desiredCount": 0}]}

        def update_service(self, **kwargs):
            raise AssertionError("should not scale an already-0, already-drained service")

    saved_sqs, saved_ecs = h.sqs, h.ecs
    h.sqs, h.ecs = FakeSqs(), FakeEcs()
    try:
        results = h.lambda_handler({}, None)
    finally:
        h.sqs, h.ecs = saved_sqs, saved_ecs

    assert results["image"]["action"] == "error"
    assert "sqs boom" in results["image"]["error"]
    assert results["video"]["action"] == "hold"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
