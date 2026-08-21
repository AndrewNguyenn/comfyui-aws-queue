"""Graduated, sticky autoscaler for ALL GPU worker fleets (image, video, minimax).

Runs every minute (EventBridge). For each fleet, sets its ECS service's
desired count from that fleet's own SQS queue depth.

Originally image-only (replacing image's old App Auto Scaling
target-tracking policy, which jumped straight to max on any message).
Extended 2026-08-10 to also drive the video fleet, replacing ITS target-
tracking policy: target-tracking's scale-in alarm needs ~15 consecutive
minutes below target to fire (see infra/lib/stacks/compute.ts history), so
every video burst paid ~15-30 min of idle GPU at burst end. This scaler
releases one worker per tick (~1-2 min) once a fleet's queue drains, same as
image already did. One Lambda, one 1-min tick, drives both — each fleet's
queue/service/bands/max are independent (see FLEETS below).

Scale-UP is graduated and lazy — workers track the visible backlog. Bands
(visible queue depth -> target workers), per fleet:

    IMAGE                          VIDEO
    0          -> 0                0        -> 0
    1  - 11    -> 1                1        -> 1
    12 - 39    -> 2                2 - 5    -> 2
    >= 40      -> MAX (3)          >= 6     -> MAX (3)

Image bands (re-tuned 2026-08-10): the previous ladder (1-24->2, 25-74->4)
booted a SECOND worker for a single queued job. At ~60 s/image on A10G, a
dozen queued images finishes in ~12 min on ONE worker — acceptable latency
for a low-volume deployment. A second worker now only kicks in once a real
backlog (12+) has formed, and the third (== MAX_WORKERS) only past 40 queued.

Video bands: video jobs are long-running (minutes, not seconds) and aren't
worth batching, so a single queued job gets a worker immediately. A couple
more queued jobs justifies a second worker; >=6 goes to MAX.

Scale-DOWN is sticky, per fleet — a fleet never sheds workers while its own
queue has work. Once N workers are up they stay up (`max(current, target)`)
for as long as anything is visible or in flight on THAT fleet's queue.

When a fleet's queue is *fully cleared* (nothing visible AND nothing in
flight) we release — but one worker per tick, not all at once.
ApproximateNumberOfMessages and ...NotVisible are both eventually-consistent
and can momentarily read 0 mid-batch; a one-shot release-to-0 would drain the
whole fleet on such a false empty. Stepping down caps that risk to a single
worker (the next tick re-reads the truth and the ratchet restores it), and a
genuine clear still releases fully within `current` minutes.

A failure scaling one fleet does not block the other — each is scaled
independently and errors are caught per-fleet (a failed tick self-heals next
minute regardless).
"""
import os

import boto3

CLUSTER = os.environ["CLUSTER"]  # both fleets share one ECS cluster

# (exclusive_upper_bound, target_workers) pairs, ascending. Above the last
# bound, the fleet's max_workers applies. See module docstring for the table.
IMAGE_BANDS = [(1, 0), (12, 1), (40, 2)]
VIDEO_BANDS = [(1, 0), (2, 1), (6, 2)]
# MiniMax H3 used to be a single-worker fleet, so its only decision was on/off
# and one band said so: [(1, 0)]. That is a trap now that minimaxMax is 3 —
# "above the last bound, max_workers applies" means a SINGLE queued clip would
# wake all three g6e workers, each paying a ~6-7 min cold start to stage ~50 GiB
# of MiniMax weights plus ~13 GiB of RedMix for frame 0. Three cold starts to
# render one clip is worse than the serial case it replaced.
#
# So spread it. At ~16 min a clip, a second worker only earns its cold start
# once the queue is deep enough that it saves more than it costs; below that,
# waiting is cheaper than warming.
MINIMAX_BANDS = [(1, 0), (6, 1), (15, 2)]

_BANDS = {"image": IMAGE_BANDS, "video": VIDEO_BANDS, "minimax": MINIMAX_BANDS}

# Fleets are discovered from the environment (<NAME>_QUEUE_URL / _SERVICE /
# _MAX_WORKERS), not hardcoded — compute.ts emits one triple per fleet, so
# adding a fleet there needs no change here. A fleet whose vars are absent is
# skipped rather than raising, so a partially-deployed stack still scales the
# fleets it does have instead of the Lambda failing outright on import.
def _discover_fleets() -> list:
    found = []
    for name, bands in _BANDS.items():
        key = name.upper()
        queue_url = os.environ.get(f"{key}_QUEUE_URL")
        service = os.environ.get(f"{key}_SERVICE")
        if not queue_url or not service:
            continue
        found.append({
            "name": name,
            "queue_url": queue_url,
            "service": service,
            "max_workers": int(os.environ.get(f"{key}_MAX_WORKERS", "3")),
            "bands": bands,
        })
    return found


FLEETS = _discover_fleets()

sqs = boto3.client("sqs")
ecs = boto3.client("ecs")


def step_target(visible: int, bands: list, max_workers: int) -> int:
    """Graduated scale-up target by visible (waiting) queue depth.

    `bands` is a list of (exclusive_upper_bound, target) pairs, ascending
    (see IMAGE_BANDS / VIDEO_BANDS). Above the last bound, returns
    max_workers.
    """
    for upper, target in bands:
        if visible < upper:
            return target
    return max_workers


def decide(visible: int, inflight: int, current: int, bands: list, max_workers: int) -> int:
    """Desired worker count for one fleet's queue + current state."""
    if visible == 0 and inflight == 0:
        # Fully cleared — release gradually (see module docstring).
        return max(0, current - 1)
    # Ratchet up to the step target; never shed while work remains. The
    # min(..., max_workers) also lets a lowered *Max shrink a live fleet.
    return min(max(current, step_target(visible, bands, max_workers)), max_workers)


def lambda_handler(_event, _context):
    results = {}
    for fleet in FLEETS:
        try:
            results[fleet["name"]] = _scale_fleet(fleet)
        except Exception as e:  # noqa: BLE001 — one fleet's failure shouldn't block the other; both self-heal next tick
            print(f"scaler: ERROR fleet={fleet['name']} {e!r}")
            results[fleet["name"]] = {"action": "error", "error": repr(e)}
    return results


def _scale_fleet(fleet: dict) -> dict:
    attrs = sqs.get_queue_attributes(
        QueueUrl=fleet["queue_url"],
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    ).get("Attributes", {})
    visible = int(attrs.get("ApproximateNumberOfMessages", 0))
    inflight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))

    resp = ecs.describe_services(cluster=CLUSTER, services=[fleet["service"]])
    services = resp.get("services") or []
    if not services:
        print(f"scaler: service {fleet['service']} not found; failures={resp.get('failures')}")
        return {"action": "noop", "reason": "service-not-found"}
    current = int(services[0]["desiredCount"])

    bands = fleet["bands"]
    max_workers = fleet["max_workers"]
    new_desired = decide(visible, inflight, current, bands, max_workers)

    action = "hold"
    if new_desired != current:
        ecs.update_service(cluster=CLUSTER, service=fleet["service"], desiredCount=new_desired)
        action = "scale-up" if new_desired > current else "scale-down"

    print(
        f"scaler: fleet={fleet['name']} visible={visible} inflight={inflight} current={current} "
        f"step_target={step_target(visible, bands, max_workers)} -> {action} desired={new_desired}"
    )
    return {
        "visible": visible,
        "inflight": inflight,
        "current": current,
        "desired": new_desired,
        "action": action,
    }
