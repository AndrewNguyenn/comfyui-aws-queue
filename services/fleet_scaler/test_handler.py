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
os.environ.setdefault("IMAGE_QUEUE_URL", "x-image")
os.environ.setdefault("CLUSTER", "c")
os.environ.setdefault("IMAGE_SERVICE", "s-image")
os.environ["IMAGE_MAX_WORKERS"] = "3"  # matches config.ts scaling.imageMax
os.environ.setdefault("VIDEO_QUEUE_URL", "x-video")
os.environ.setdefault("VIDEO_ASG", "asg-video")
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
    assert by_name["image"]["service"] == "s-image"   # SERVICE mode
    assert by_name["image"]["asg"] is None
    assert by_name["image"]["bands"] is h.IMAGE_BANDS
    assert by_name["video"]["queue_url"] == "x-video"
    assert by_name["video"]["asg"] == "asg-video"     # ASG mode
    assert by_name["video"]["service"] is None
    assert by_name["video"]["bands"] is h.VIDEO_BANDS


def test_one_fleet_error_does_not_block_the_other():
    # stub sqs/asg: image fleet's GetQueueAttributes blows up, video's succeeds
    class FakeSqs:
        def get_queue_attributes(self, QueueUrl, AttributeNames):
            if QueueUrl == "x-image":
                raise RuntimeError("sqs boom")
            return {"Attributes": {"ApproximateNumberOfMessages": "0",
                                    "ApproximateNumberOfMessagesNotVisible": "0"}}

    class FakeAsg:
        def describe_auto_scaling_groups(self, AutoScalingGroupNames):
            return {"AutoScalingGroups": [{"DesiredCapacity": 0}]}

        def set_desired_capacity(self, **kwargs):
            raise AssertionError("should not scale an already-0, already-drained fleet")

    class FakeEcs:
        def describe_services(self, cluster, services):
            return {"services": [{"desiredCount": 0}]}

        def update_service(self, **kwargs):
            raise AssertionError("should not scale an already-0, already-drained service")

    saved = h.sqs, h.asg, h.ecs
    h.sqs, h.asg, h.ecs = FakeSqs(), FakeAsg(), FakeEcs()
    try:
        results = h.lambda_handler({}, None)
    finally:
        h.sqs, h.asg, h.ecs = saved

    assert results["image"]["action"] == "error"
    assert "sqs boom" in results["image"]["error"]
    assert results["video"]["action"] == "hold"



def test_minimax_does_not_wake_the_whole_fleet_for_one_clip():
    # "Above the last bound, max_workers applies" — so a single-entry band table
    # means ANY queued job targets max. That was fine at minimaxMax=1 and became
    # a trap at 3: one clip would pay three ~6-7 min cold starts (~50 GiB of
    # MiniMax weights plus ~13 GiB of RedMix each) to render one clip.
    assert h.step_target(0, h.MINIMAX_BANDS, 3) == 0
    assert h.step_target(1, h.MINIMAX_BANDS, 3) == 1
    assert h.step_target(5, h.MINIMAX_BANDS, 3) == 1
    assert h.step_target(6, h.MINIMAX_BANDS, 3) == 2
    assert h.step_target(15, h.MINIMAX_BANDS, 3) == 3


def test_no_fleet_ramps_straight_to_max_on_one_message():
    # The property, not just the one fleet that tripped it: a table that reaches
    # max_workers at depth 1 has no ramp at all.
    for name, bands in h._BANDS.items():
        assert h.step_target(1, bands, 3) < 3, f"{name} wakes the whole fleet for one job"


# --- warm capacity ---------------------------------------------------------
# The bands count only queue depth, which let the capacity provider hold an
# idle instance while a job waited: billed for two, using one. Observed live —
# 2 g6e running, 1 task, 1 job queued, CapacityProviderReservation at 50.

def test_a_waiting_job_uses_a_warm_instance_the_bands_would_ignore():
    # 1 queued, 1 running, 2 warm boxes. Bands say 1 (depth < 6); the warm one
    # should be put to work rather than sit idle at full price.
    assert h.decide(1, 1, 1, h.MINIMAX_BANDS, 3, warm=2) == 2


def test_warm_capacity_never_exceeds_the_work_that_is_waiting():
    # 3 warm boxes, 1 job waiting -> 2 workers (the running one plus that job),
    # not 3. Idle tasks for work that does not exist help nobody.
    assert h.decide(1, 1, 1, h.MINIMAX_BANDS, 3, warm=3) == 2


def test_warm_capacity_is_still_bounded_by_max_workers():
    assert h.decide(5, 0, 1, h.MINIMAX_BANDS, 3, warm=9) == 3


def test_warm_capacity_never_launches_beyond_the_band_when_nothing_is_warm():
    # The lazy ramp is the whole point of the bands; warm=0 must not change it.
    assert h.decide(1, 0, 0, h.MINIMAX_BANDS, 3, warm=0) == 1
    assert h.decide(5, 0, 0, h.MINIMAX_BANDS, 3, warm=0) == 1


def test_warm_capacity_does_not_block_release_to_zero():
    # The dangerous case: a fleet with registered instances must still drain to
    # 0 when its queue is clear, or it can never scale in.
    assert h.decide(0, 0, 2, h.MINIMAX_BANDS, 3, warm=2) == 1
    assert h.decide(0, 0, 1, h.MINIMAX_BANDS, 3, warm=2) == 0


def test_inflight_only_does_not_summon_warm_workers():
    # Nothing is WAITING — the job in flight already has a worker. Adding one
    # would be an idle task.
    assert h.decide(0, 1, 1, h.MINIMAX_BANDS, 3, warm=3) == 1


def test_warm_capacity_defaults_off_for_existing_callers():
    assert h.decide(1, 1, 1, h.MINIMAX_BANDS, 3) == 1


def test_scale_up_actuates_on_the_asg_not_the_service():
    """The actuator moved from ECS desiredCount to ASG desired capacity.

    The sibling error test only asserts we DON'T scale a drained fleet, so
    without this nothing checks that a real scale-up reaches AWS at all, or
    that it names the right ASG.
    """
    calls = []

    class FakeSqs:
        def get_queue_attributes(self, QueueUrl, AttributeNames):
            # 3 waiting on video -> bands (2,1),(6,2) put the target at 2
            n = "3" if QueueUrl == "x-video" else "0"
            return {"Attributes": {"ApproximateNumberOfMessages": n,
                                   "ApproximateNumberOfMessagesNotVisible": "0"}}

    class FakeAsg:
        def describe_auto_scaling_groups(self, AutoScalingGroupNames):
            return {"AutoScalingGroups": [{"DesiredCapacity": 0}]}

        def set_desired_capacity(self, **kw):
            calls.append(kw)

    class FakeEcs:
        def describe_services(self, cluster, services):
            return {"services": [{"desiredCount": 0}]}

        def update_service(self, **kwargs):
            raise AssertionError("SERVICE-mode fleet must not be touched here")

    saved = h.sqs, h.asg, h.ecs
    h.sqs, h.asg, h.ecs = FakeSqs(), FakeAsg(), FakeEcs()
    try:
        results = h.lambda_handler({}, None)
    finally:
        h.sqs, h.asg, h.ecs = saved

    assert results["video"]["action"] == "scale-up"
    assert results["video"]["desired"] == 2
    # image is SERVICE mode with an empty queue -> never reaches the ASG API
    assert results["image"]["action"] == "hold"
    assert calls == [{"AutoScalingGroupName": "asg-video", "DesiredCapacity": 2}]


def test_asg_not_found_is_a_noop_not_a_crash():
    """A missing ASG must not take the whole tick down with it."""
    class FakeSqs:
        def get_queue_attributes(self, QueueUrl, AttributeNames):
            return {"Attributes": {"ApproximateNumberOfMessages": "5",
                                   "ApproximateNumberOfMessagesNotVisible": "0"}}

    class FakeAsg:
        def describe_auto_scaling_groups(self, AutoScalingGroupNames):
            return {"AutoScalingGroups": []}

        def set_desired_capacity(self, **kw):
            raise AssertionError("must not actuate on an ASG we could not read")

    class FakeEcs:
        def describe_services(self, cluster, services):
            return {"services": [{"desiredCount": 0}]}

        def update_service(self, **kwargs):
            raise AssertionError("SERVICE-mode fleet must not be touched here")

    saved = h.sqs, h.asg, h.ecs
    h.sqs, h.asg, h.ecs = FakeSqs(), FakeAsg(), FakeEcs()
    try:
        results = h.lambda_handler({}, None)
    finally:
        h.sqs, h.asg, h.ecs = saved

    assert results["video"]["action"] == "noop"
    assert results["video"]["reason"] == "asg-not-found"


def test_scale_down_actuates_on_the_asg():
    """A drained ASG-mode fleet steps its DESIRED CAPACITY down by one — on the
    ASG, not on a service. The scale-up sibling proved the up path; nothing
    proved the release ever reaches the ASG API."""
    calls = []

    class FakeSqs:
        def get_queue_attributes(self, QueueUrl, AttributeNames):
            return {"Attributes": {"ApproximateNumberOfMessages": "0",
                                   "ApproximateNumberOfMessagesNotVisible": "0"}}

    class FakeAsg:
        def describe_auto_scaling_groups(self, AutoScalingGroupNames):
            return {"AutoScalingGroups": [{"DesiredCapacity": 2}]}

        def set_desired_capacity(self, **kw):
            calls.append(kw)

    class FakeEcs:
        def describe_services(self, cluster, services):
            return {"services": [{"desiredCount": 0}]}

        def update_service(self, **kwargs):
            raise AssertionError("release must not touch a service in ASG mode")

    saved = h.sqs, h.asg, h.ecs
    h.sqs, h.asg, h.ecs = FakeSqs(), FakeAsg(), FakeEcs()
    try:
        results = h.lambda_handler({}, None)
    finally:
        h.sqs, h.asg, h.ecs = saved

    assert results["video"]["action"] == "scale-down"
    assert calls == [{"AutoScalingGroupName": "asg-video", "DesiredCapacity": 1}]


def test_asg_wins_when_both_actuators_are_set():
    """Half-finished migration: both VIDEO_ASG and VIDEO_SERVICE present. The
    scaler must drive the ASG and forget the service, or it would be setting
    desiredCount on a DAEMON service that has none."""
    os.environ["VIDEO_SERVICE"] = "s-video-stale"
    try:
        fleets = {f["name"]: f for f in h._discover_fleets()}
    finally:
        del os.environ["VIDEO_SERVICE"]
    assert fleets["video"]["asg"] == "asg-video"
    assert fleets["video"]["service"] is None
    assert fleets["image"]["service"] == "s-image" and fleets["image"]["asg"] is None


def test_asg_mode_never_consults_warm_capacity():
    """warm_capacity() counts ECS container instances to raise a SERVICE-mode
    target toward boxes that are already up. Under DAEMON scheduling task
    count == instance count by construction, so the scaler skips it — and
    must not even call ECS for it (the Lambda's IAM has no ecs:* for ASG
    fleets)."""
    class FakeSqs:
        def get_queue_attributes(self, QueueUrl, AttributeNames):
            n = "1" if QueueUrl == "x-video" else "0"
            return {"Attributes": {"ApproximateNumberOfMessages": n,
                                   "ApproximateNumberOfMessagesNotVisible": "0"}}

    class FakeAsg:
        def describe_auto_scaling_groups(self, AutoScalingGroupNames):
            return {"AutoScalingGroups": [{"DesiredCapacity": 1}]}

        def set_desired_capacity(self, **kw):
            raise AssertionError("1 waiting on 1 worker is a hold, not a scale")

    class FakeEcs:
        def describe_services(self, cluster, services):
            return {"services": [{"desiredCount": 0}]}

        def list_container_instances(self, **kw):
            raise AssertionError("ASG-mode fleet consulted warm_capacity()")

        def describe_container_instances(self, **kw):
            raise AssertionError("ASG-mode fleet consulted warm_capacity()")

        def update_service(self, **kwargs):
            raise AssertionError("SERVICE call in ASG mode")

    saved = h.sqs, h.asg, h.ecs
    h.sqs, h.asg, h.ecs = FakeSqs(), FakeAsg(), FakeEcs()
    try:
        results = h.lambda_handler({}, None)
    finally:
        h.sqs, h.asg, h.ecs = saved

    assert results["video"]["action"] == "hold", results["video"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")

