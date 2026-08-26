"""Graduated, sticky autoscaler for ALL GPU worker fleets (image, video, minimax).

Runs every minute (EventBridge). For each fleet, sets its Auto Scaling
group's desired capacity from that fleet's own SQS queue depth.

2026-08-25: fleets may now actuate in EITHER of two modes, chosen per fleet
by which env vars are set. The decision logic is identical in both.

  SERVICE mode ({FLEET}_SERVICE)  — the original. Sets the ECS service's
      desiredCount; the capacity provider's managed scaling turns that into
      instances. Used by image and video.

  ASG mode ({FLEET}_ASG)  — sets the Auto Scaling group's desired capacity
      directly. Used where the service is DAEMON-scheduled (one task per
      matching instance, no desiredCount to set), which is how one ASG can
      launch a mix of GPU architectures and have each instance pick up the
      task definition — and therefore the container image — built for its own
      GPU. DAEMON tasks never sit pending, so managed scaling has nothing to
      react to and is turned off for those fleets; this Lambda owns capacity.

UNITS, NOT WORKERS, whenever an ASG carries instance weights (config.ts
fleets.*.instanceWeights — unset as of 2026-08-26, when minimax went
g7e-only; it was g7e.4xlarge=2, g6e.8xlarge=1 while both were pooled). With
weights set, every number this module produces or reads — band targets, max_workers, DesiredCapacity — is in capacity UNITS
(g6e-worker-equivalents), not instances. A desired of 1 or 2 is ONE g7e; the
group sits at or above desired (AWS never terminates an instance that would
drop it below), so the release ladder 4->3->2->1->0 can take a tick that
moves nothing. The bands below still read naturally: one g7e does two g6e
worth of clips in the same time, so "2 units at 6 queued" is the same
throughput promise either way.

Both modes coexist deliberately: converting a fleet is a service REPLACEMENT
(schedulingStrategy is immutable), so fleets migrate one at a time rather than
in one high-blast-radius deploy.

Originally image-only (replacing image's old App Auto Scaling
target-tracking policy, which jumped straight to max on any message).
Extended 2026-08-10 to also drive the video fleet, replacing ITS target-
tracking policy: target-tracking's scale-in alarm needs ~15 consecutive
minutes below target to fire (see infra/lib/stacks/compute.ts history), so
every video burst paid ~15-30 min of idle GPU at burst end. This scaler
releases one worker per tick (~1-2 min) once a fleet's queue drains, same as
image already did. One Lambda, one 1-min tick, drives both — each fleet's
queue/asg/bands/max are independent (see FLEETS below).

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

Scale-UP has one override on the bands, in SERVICE mode only: if work is
WAITING and the fleet already has warm, agent-connected container instances,
the target rises to use them (capped by how many jobs are actually waiting).
The bands exist to avoid paying a cold start for a shallow queue; that says
nothing about a box already running. Without it the capacity provider could
hold an idle instance while a job sat in the queue — paying for two, using
one. It never LAUNCHES an instance, so the lazy ramp is unchanged.

In ASG mode the override is meaningless and is skipped: task count and
instance count are the same number by construction, so `warm` would always
equal `current` and `max(target, min(warm, current + visible))` could only
return `current` — which the ratchet already guarantees.

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

# Only SERVICE-mode fleets use this; ASG-mode fleets never touch ECS.
CLUSTER = os.environ.get("CLUSTER", "")

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
MINIMAX_BANDS = [(1, 0), (6, 1), (15, 2)]  # instances today; UNITS if weights return — see docstring

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
        # Exactly one actuator per fleet. ASG wins if both are somehow set, so
        # a half-finished migration fails toward the newer path rather than
        # driving a DAEMON service's nonexistent desiredCount.
        asg_name = os.environ.get(f"{key}_ASG")
        service = os.environ.get(f"{key}_SERVICE")
        if not queue_url or not (asg_name or service):
            continue
        found.append({
            "name": name,
            "queue_url": queue_url,
            "asg": asg_name,
            "service": None if asg_name else service,
            "max_workers": int(os.environ.get(f"{key}_MAX_WORKERS", "3")),
            "bands": bands,
        })
    return found


FLEETS = _discover_fleets()

sqs = boto3.client("sqs")
ecs = boto3.client("ecs")
asg = boto3.client("autoscaling")


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


def decide(visible: int, inflight: int, current: int, bands: list,
           max_workers: int, warm: int = 0) -> int:
    """Desired worker count for one fleet's queue + current state.

    `warm` is how many container instances are registered and agent-connected
    for this fleet — capacity that is already running and already being paid
    for. The bands alone ignore it, and that produced the worst possible
    outcome: a job waiting in the queue while a warm, idle GPU sat next to it,
    because the band table said a shallow queue does not justify a worker.

    That reasoning is about COLD STARTS — roughly 7 minutes to stage ~50 GiB of
    weights — and it does not apply to a box that is already up. So when work is
    actually waiting, never target fewer workers than the warm capacity can
    absorb. This only ever RAISES the target toward instances that already
    exist; it never launches one, so the lazy ramp is untouched.
    """
    if visible == 0 and inflight == 0:
        # Fully cleared — release gradually (see module docstring). No warm
        # boost here, or a fleet with registered instances could never reach 0.
        return max(0, current - 1)
    target = step_target(visible, bands, max_workers)
    if visible > 0:
        # Cap at current + visible: put a warm box to work on a job that is
        # actually waiting, never spin idle tasks for work that does not exist.
        target = max(target, min(warm, current + visible))
    # Ratchet up to the target; never shed while work remains. The
    # min(..., max_workers) also lets a lowered *Max shrink a live fleet.
    return min(max(current, target), max_workers)


def warm_capacity(fleet_name: str) -> int:
    """Container instances for this fleet that could take a task right now.

    SERVICE mode only — see the module docstring. Best-effort: any failure
    returns 0, which degrades to band-only behaviour rather than breaking the
    tick. An instance whose agent is disconnected is not counted; it cannot be
    placed on.
    """
    try:
        arns = ecs.list_container_instances(
            cluster=CLUSTER,
            filter=f"attribute:fleet == {fleet_name}",
            status="ACTIVE",
        ).get("containerInstanceArns") or []
        if not arns:
            return 0
        instances = ecs.describe_container_instances(
            cluster=CLUSTER, containerInstances=arns,
        ).get("containerInstances") or []
        return sum(1 for i in instances if i.get("agentConnected"))
    except Exception as e:  # noqa: BLE001
        print(f"scaler: warm_capacity({fleet_name}) failed, ignoring: {e!r}")
        return 0


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

    on_asg = bool(fleet["asg"])

    if on_asg:
        resp = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[fleet["asg"]])
        groups = resp.get("AutoScalingGroups") or []
        if not groups:
            print(f"scaler: asg {fleet['asg']} not found")
            return {"action": "noop", "reason": "asg-not-found"}
        current = int(groups[0]["DesiredCapacity"])
    else:
        resp = ecs.describe_services(cluster=CLUSTER, services=[fleet["service"]])
        services = resp.get("services") or []
        if not services:
            print(f"scaler: service {fleet['service']} not found; failures={resp.get('failures')}")
            return {"action": "noop", "reason": "service-not-found"}
        current = int(services[0]["desiredCount"])

    bands = fleet["bands"]
    max_workers = fleet["max_workers"]
    # The warm override is a SERVICE-mode concept only (module docstring).
    warm = 0 if on_asg else (warm_capacity(fleet["name"]) if visible > 0 else 0)
    new_desired = decide(visible, inflight, current, bands, max_workers, warm)

    action = "hold"
    if new_desired != current:
        if on_asg:
            # HonorCooldown defaults to False, which is what we want: the tick
            # is already the rate limit, and a cooldown would swallow the
            # one-per-tick release just as the queue clears.
            asg.set_desired_capacity(
                AutoScalingGroupName=fleet["asg"], DesiredCapacity=new_desired,
            )
        else:
            ecs.update_service(
                cluster=CLUSTER, service=fleet["service"], desiredCount=new_desired,
            )
        action = "scale-up" if new_desired > current else "scale-down"

    print(
        f"scaler: fleet={fleet['name']}({'asg' if on_asg else 'svc'}) visible={visible} "
        f"inflight={inflight} current={current} "
        f"step_target={step_target(visible, bands, max_workers)} -> {action} desired={new_desired}"
    )
    return {
        "visible": visible,
        "inflight": inflight,
        "current": current,
        "desired": new_desired,
        "action": action,
    }
