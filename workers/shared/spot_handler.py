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
import urllib3

log = logging.getLogger(__name__)
http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=2.0, read=2.0))

IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
SPOT_INSTRUCTION_URL = "http://169.254.169.254/latest/meta-data/spot/instruction"


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

    def _watch(self) -> None:
        # Get IMDSv2 token once; refresh hourly via TTL.
        token = self._refresh_imds_token()
        last_token_refresh = time.monotonic()

        while not self._stop_event.is_set():
            try:
                if time.monotonic() - last_token_refresh > 3600:
                    token = self._refresh_imds_token()
                    last_token_refresh = time.monotonic()

                r = http.request(
                    "GET",
                    SPOT_INSTRUCTION_URL,
                    headers={"X-aws-ec2-metadata-token": token},
                )
                if r.status == 200 and r.data.strip().lower() in (b"terminate", b"stop"):
                    log.warning("spot termination notice received")
                    self.terminating = True
                    with self._lock:
                        in_flight = self._in_flight
                    job_id = in_flight[0] if in_flight else None
                    try:
                        self._on_terminate(job_id)
                    except Exception:  # noqa: BLE001 — keep watcher alive
                        log.exception("on_terminate callback failed")
                    return
            except urllib3.exceptions.HTTPError:
                # No interruption notice (404 or no instruction file). Normal.
                pass
            except Exception:  # noqa: BLE001
                log.exception("spot watcher error (continuing)")

            self._stop_event.wait(self._poll)

    def _refresh_imds_token(self) -> str:
        try:
            r = http.request(
                "PUT",
                IMDS_TOKEN_URL,
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            )
            if r.status == 200:
                return r.data.decode().strip()
        except Exception:  # noqa: BLE001
            log.exception("imds token refresh failed")
        return ""


def make_default_on_terminate(
    queue_url: str, jobs_table: str, region: str
) -> Callable[[Optional[str]], None]:
    """Standard handler: reset SQS visibility on the in-flight message, mark
    DDB status=queued so it appears available for the next worker."""
    sqs = boto3.client("sqs", region_name=region)
    ddb = boto3.client("dynamodb", region_name=region)

    def handler(job_id: Optional[str]) -> None:
        if not job_id:
            log.info("no in-flight job at termination time")
            return
        log.warning("releasing in-flight job %s back to queue", job_id)
        try:
            ddb.update_item(
                TableName=jobs_table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET #s = :s",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": {"S": "queued"}},
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to revert job %s to queued", job_id)

    return handler
