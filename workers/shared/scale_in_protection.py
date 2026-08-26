"""Per-instance ASG scale-in protection, held for the duration of a job.

WHY THIS EXISTS
---------------
DAEMON-scheduled fleets cannot use ECS managed termination protection. AWS is
explicit: "Tasks that are run by a service that uses the DAEMON scheduling
strategy are ignored and an instance can be terminated by cluster auto scaling
even when the instance is running these tasks." So the capacity provider that
used to keep voluntary scale-in off a busy worker no longer does.

That matters more than it sounds. The ASG carries an ECS managed-draining
lifecycle hook on EC2_INSTANCE_TERMINATING with a 1-hour heartbeat, and
terminating an instance mid-render is exactly the shape that once wedged one in
Terminating:Wait for ~17h, jamming the ASG's scale-in and holding GPUs up
across idle windows (2026-07-12). A MiniMax H3 clip takes ~9-23 minutes, so the
window is wide.

This module puts the flag back under the worker's own control: protect on job
pickup, release when the job ends. The fleet-scaler only scales down once a
queue is fully drained, so in the normal case this never even fires — it is
the guard for the case where those two facts disagree.

NOT A SPOT GUARD. Scale-in protection does not block spot reclamation; AWS
reclaims spot regardless. spot_handler.py covers that path separately.

FAIL-OPEN, ALWAYS. Every call is best-effort. A worker that cannot set the flag
still runs its job — losing a render to an unlucky scale-in is bad, refusing to
render at all is worse.
"""

from __future__ import annotations

import logging
import os

import boto3

import imds

log = logging.getLogger(__name__)

# Set by the task definition. Empty on fleets that still use ECS managed
# termination protection (image, video), which disables this module entirely.
ASG_NAME = os.environ.get("ASG_NAME", "")
REGION = os.environ.get("AWS_REGION", "")

_asg = boto3.client("autoscaling", region_name=REGION) if ASG_NAME else None
_instance_id: str | None = None


def _fetch_instance_id() -> str:
    """This instance's id via IMDSv2, or '' if unavailable. Cached: it cannot
    change under a running worker, and we ask on every job."""
    global _instance_id
    if _instance_id is None:
        _instance_id = imds.text("meta-data/instance-id")
    return _instance_id


def _set(protected: bool) -> bool:
    """Set the flag. Returns whether it took. Never raises."""
    if _asg is None:
        return False
    iid = _fetch_instance_id()
    if not iid:
        return False
    try:
        _asg.set_instance_protection(
            InstanceIds=[iid],
            AutoScalingGroupName=ASG_NAME,
            ProtectedFromScaleIn=protected,
        )
        log.info("scale-in protection %s for %s", "ON" if protected else "OFF", iid)
        return True
    except Exception as e:  # noqa: BLE001
        # Most likely benign: the instance is already mid-termination, or was
        # detached. Never fatal — see module docstring.
        log.warning("could not set scale-in protection=%s: %r", protected, e)
        return False


def protect() -> bool:
    """Hold this instance against voluntary ASG scale-in. Call on job pickup."""
    return _set(True)


def release() -> bool:
    """Release the hold. Call when the job ends, in a finally."""
    return _set(False)


def release_stale() -> None:
    """Clear a hold left behind by a previous worker on this instance.

    A worker killed between protect() and release() — OOM, crash, container
    restart — leaves the flag set with nothing to clear it, and the ASG can
    then never scale that instance in. The task restarts on the same instance,
    so clearing once at startup closes that hole. No-op when the flag is
    already off.
    """
    if _asg is None:
        return
    if _set(False):
        log.info("cleared any stale scale-in protection at startup")
