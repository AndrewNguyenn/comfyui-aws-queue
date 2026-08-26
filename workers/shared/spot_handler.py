"""
Spot interruption handler.

Watches the EC2 IMDSv2 spot/instruction endpoint. When AWS announces a 2-min
termination notice, we:
  1. Set a flag the main loop polls (so it exits cleanly after current job)
  2. Reset the in-flight SQS message visibility to 0 so it redelivers immediately
  3. Update the DDB job record back to status=queued

The 2 min between notice and termination is enough to flush state, but NOT
enough to finish a typical generation, so abandoning the in-flight job is
correct. Workflows are idempotent (workflow JSON in DDB, all inputs in S3).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

import boto3

import imds

log = logging.getLogger(__name__)
SPOT_INSTRUCTION_PATH = "meta-data/spot/instruction"
# Target lifecycle state of THIS instance in its ASG: "InService" while it
# should keep working; "Terminated"/"Detached"/... once the ASG has decided
# otherwise (scale-in, health replacement, max-size cut). 404 outside an ASG.
# Catching this here means an ASG-initiated termination releases the job
# before ECS even starts draining, not after. Review M1.
ASG_LIFECYCLE_PATH = "meta-data/autoscaling/target-lifecycle-state"


class SpotHandler:
    """Polls spot interruption endpoint. Calls on_terminate(in_flight_id) on notice."""

    def __init__(
        self,
        on_terminate: Callable[[Optional[str]], None],
        poll_interval_seconds: int = 5,
    ):
        self.terminating = False
        self._in_flight: Optional[tuple[str, str]] = None  # (job_id, sqs_receipt_handle)
        self._on_terminate = on_terminate
        self._lock = threading.Lock()
        self._poll = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True, name="spot-watch")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def set_in_flight(self, job_id: str, receipt_handle: str) -> None:
        with self._lock:
            self._in_flight = (job_id, receipt_handle)

    def clear_in_flight(self) -> None:
        with self._lock:
            self._in_flight = None

    def take_in_flight(self) -> Optional[tuple[str, str]]:
        """Return and clear the in-flight (job_id, receipt) so exactly one
        path — spot notice, ASG lifecycle, or SIGTERM — releases it."""
        with self._lock:
            in_flight, self._in_flight = self._in_flight, None
            return in_flight

    def _watch(self) -> None:
        while not self._stop_event.is_set():
            try:
                reason = None
                status, body = imds.get(SPOT_INSTRUCTION_PATH)
                if status == 200 and body.lower() in ("terminate", "stop"):
                    reason = "spot termination notice"
                else:
                    status, body = imds.get(ASG_LIFECYCLE_PATH)
                    if status == 200 and body and body != "InService":
                        reason = f"ASG target lifecycle state {body}"
                if reason:
                    log.warning("%s received", reason)
                    self.terminating = True
                    try:
                        # Both job_id and receipt_handle, so the callback can
                        # reset SQS visibility as well as DDB.
                        self._on_terminate(self.take_in_flight())
                    except Exception:  # noqa: BLE001 — keep watcher alive
                        log.exception("on_terminate callback failed")
                    return
            except Exception:  # noqa: BLE001
                log.exception("spot watcher error (continuing)")

            self._stop_event.wait(self._poll)


def make_default_on_terminate(
    queue_url: str, jobs_table: str, region: str
) -> Callable[[Optional[tuple[str, str]]], None]:
    """Standard handler: reset SQS visibility on the in-flight message AND
    mark DDB status=queued. Both must succeed for the job to redeliver
    quickly (resolves code review C4)."""
    sqs = boto3.client("sqs", region_name=region)
    ddb = boto3.client("dynamodb", region_name=region)

    def handler(in_flight: Optional[tuple[str, str]]) -> None:
        if not in_flight:
            log.info("no in-flight job at termination time")
            return
        job_id, receipt_handle = in_flight
        log.warning("releasing in-flight job %s back to queue", job_id)
        # Reset SQS visibility FIRST so message redelivers immediately.
        try:
            sqs.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=0,
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to reset SQS visibility for %s", job_id)
        # Then update DDB so a worker picking it up sees status=queued — but
        # only if it is still 'running'. A job that finished inside the
        # notice-to-stop window has already been written 'complete' and its
        # message deleted; a 'queued' landing after that would orphan it.
        try:
            ddb.update_item(
                TableName=jobs_table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET #s = :s",
                ConditionExpression="#s = :running",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": {"S": "queued"}, ":running": {"S": "running"}},
            )
        except ddb.exceptions.ConditionalCheckFailedException:
            log.info("job %s already left 'running'; not re-queued", job_id)
        except Exception:  # noqa: BLE001
            log.exception("failed to revert job %s to queued", job_id)

    return handler
