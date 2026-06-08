"""Graduated, sticky autoscaler for the image worker fleet.

Runs every minute (EventBridge). Sets the comfy-image ECS service's desired
count from the live image-queue depth, replacing the old App Auto Scaling
target-tracking policy (which jumped straight to max on any message).

Scale-UP is graduated and lazy — workers track the visible backlog:

    visible queue depth  ->  target workers
    0                         0
    1 - 49                    2
    50 - 149                  3
    150 - 299                 4
    >= 300                    MAX_WORKERS (fleet cap)

Scale-DOWN is sticky — we never shed workers while work remains. Once N workers
are up they stay up (`max(current, target)`) for as long as anything is visible
or in flight. So a 500-job batch holds the full fleet until the last job
finishes, while a fresh 100-job batch only spins up 3 workers, never the whole
fleet pre-emptively.

When the queue is *fully cleared* (nothing visible AND nothing in flight) we
release — but one worker per tick, not all at once. ApproximateNumberOfMessages
and ...NotVisible are both eventually-consistent and can momentarily read 0
mid-batch; a one-shot release-to-0 would drain the whole fleet on such a false
empty. Stepping down caps that risk to a single worker (the next tick re-reads
the truth and the ratchet restores it), and a genuine clear still releases fully
within `current` minutes.
"""
import os

import boto3

QUEUE_URL = os.environ["IMAGE_QUEUE_URL"]
CLUSTER = os.environ["CLUSTER"]
SERVICE = os.environ["SERVICE"]
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

sqs = boto3.client("sqs")
ecs = boto3.client("ecs")


def step_target(visible: int) -> int:
    """Graduated scale-up target by visible (waiting) queue depth."""
    if visible <= 0:
        return 0
    if visible < 50:
        return 2
    if visible < 150:
        return 3
    if visible < 300:
        return 4
    return MAX_WORKERS  # >= 300, capped at the fleet max


def decide(visible: int, inflight: int, current: int) -> int:
    """Desired worker count for the current queue + fleet state."""
    if visible == 0 and inflight == 0:
        # Fully cleared — release gradually (see module docstring).
        return max(0, current - 1)
    # Ratchet up to the step target; never shed while work remains. The
    # min(..., MAX_WORKERS) also lets a lowered imageMax shrink a live fleet.
    return min(max(current, step_target(visible)), MAX_WORKERS)


def lambda_handler(_event, _context):
    try:
        return _scale()
    except Exception as e:  # noqa: BLE001 — a failed tick self-heals next minute
        print(f"scaler: ERROR {e!r}")
        return {"action": "error", "error": repr(e)}


def _scale():
    attrs = sqs.get_queue_attributes(
        QueueUrl=QUEUE_URL,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    ).get("Attributes", {})
    visible = int(attrs.get("ApproximateNumberOfMessages", 0))
    inflight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))

    resp = ecs.describe_services(cluster=CLUSTER, services=[SERVICE])
    services = resp.get("services") or []
    if not services:
        print(f"scaler: service {SERVICE} not found; failures={resp.get('failures')}")
        return {"action": "noop", "reason": "service-not-found"}
    current = int(services[0]["desiredCount"])

    new_desired = decide(visible, inflight, current)

    action = "hold"
    if new_desired != current:
        ecs.update_service(cluster=CLUSTER, service=SERVICE, desiredCount=new_desired)
        action = "scale-up" if new_desired > current else "scale-down"

    print(
        f"scaler: visible={visible} inflight={inflight} current={current} "
        f"step_target={step_target(visible)} -> {action} desired={new_desired}"
    )
    return {
        "visible": visible,
        "inflight": inflight,
        "current": current,
        "desired": new_desired,
        "action": action,
    }
