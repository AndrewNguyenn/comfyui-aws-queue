"""
Reaper Lambda — SQS-tick-driven zombie sweeper.

A GPU worker heartbeats every 60 s into its job's DynamoDB record while it
runs. If a worker dies without cleaning up — a spot reclaim with no notice,
an ECS task rotation, an OOM, a crash — the job is stranded in `running`
forever: a "zombie". This Lambda finds zombies and either re-queues them
(up to MAX_RETRIES) or, once they've burned every retry, marks them failed
and parks the workflow JSON in S3 for later analysis (3 deaths in a row is
not bad luck — it's the workflow).

It is NOT on a fixed cron. It runs only while there is work: the dispatcher
seeds a tick when a job is submitted, and every run re-arms a delayed tick
*only while jobs are queued or running*. When the system goes idle the tick
chain stops and the Lambda goes dormant.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

ddb = boto3.client("dynamodb")
sqs = boto3.client("sqs")
s3 = boto3.client("s3")

JOBS_TABLE = os.environ["JOBS_TABLE"]
IMAGE_QUEUE_URL = os.environ["IMAGE_QUEUE_URL"]
VIDEO_QUEUE_URL = os.environ["VIDEO_QUEUE_URL"]
REAPER_QUEUE_URL = os.environ["REAPER_QUEUE_URL"]
FAILED_WORKFLOWS_BUCKET = os.environ["FAILED_WORKFLOWS_BUCKET"]

STATUS_INDEX = "jobs-by-status"  # INCLUDE-projection GSI (carries last_heartbeat + attempt_count)
ZOMBIE_AFTER_SEC = 300   # no heartbeat for 5 min → the worker is gone
TICK_INTERVAL_SEC = 180  # re-arm cadence while work remains
MAX_RETRIES = 3          # re-queue a zombie this many times, then give up


def lambda_handler(event: dict, _context) -> dict:
    """Triggered by a batch of ticks from the reaper queue. The tick content
    is irrelevant — any tick means "sweep now" — so the whole batch collapses
    into one sweep.

    Reaps `running` zombies only — the dead-worker case. A job stuck in
    `queued` (e.g. the dispatcher died between writing the DDB record and
    sending the SQS message) is a different failure mode this does not touch.

    On error the handler raises, so the SQS batch is not deleted and the
    ticks redeliver — the sweep simply retries. Re-sweeping is idempotent
    (already-requeued jobs are no longer `running`)."""
    running = _query_status("running")
    reaped = sum(1 for it in running if _maybe_reap(it))
    queued = _count_status("queued")

    # Re-arm only while there is work. Using the pre-reap running count is
    # fine: a just-re-queued zombie is counted as work, and one harmless
    # trailing tick after everything settles is acceptable.
    work = len(running) + queued
    if work > 0:
        _arm(TICK_INTERVAL_SEC)

    print(f"reaper: running={len(running)} queued={queued} "
          f"reaped={reaped} rearmed={work > 0}")
    return {"running": len(running), "queued": queued, "reaped": reaped}


def _maybe_reap(item: dict) -> bool:
    """Reap `item` if it's a zombie. Per-job try/except so one bad record
    can't abort the whole sweep."""
    try:
        now = time.time()
        # last_heartbeat is the freshest liveness signal; started_at covers a
        # worker that died before its first (60 s) heartbeat.
        beat = (item.get("last_heartbeat", {}).get("S")
                or item.get("started_at", {}).get("S"))
        beat_epoch = _epoch(beat) if beat else None
        if beat_epoch is None:
            # No timestamp, or one we can't parse. An unreadable heartbeat is
            # not evidence the worker died — skip rather than reap a live job.
            return False
        if (now - beat_epoch) < ZOMBIE_AFTER_SEC:
            return False  # heartbeat is fresh — worker alive
        return _handle_zombie(item)
    except Exception as e:  # noqa: BLE001
        print(f"reaper: error handling {item.get('job_id', {}).get('S', '?')}: {e!r}")
        return False


def _handle_zombie(item: dict) -> bool:
    job_id = item["job_id"]["S"]
    attempts = int(item.get("attempt_count", {}).get("N", "0"))
    if attempts < MAX_RETRIES:
        return _requeue(item, attempts)
    return _give_up(item, attempts)


def _requeue(item: dict, attempts: int) -> bool:
    """Zombie with retries left → running → queued (attempt_count++), then send
    a fresh job message so a live worker picks it up."""
    job_id = item["job_id"]["S"]
    job_type = item.get("type", {}).get("S", "image")
    prev_hb = item.get("last_heartbeat", {}).get("S")

    names, values, cond = _zombie_condition(prev_hb)
    values[":queued"] = {"S": "queued"}
    values[":ac"] = {"N": str(attempts + 1)}
    try:
        ddb.update_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
            # Clear the stale heartbeat/progress so the re-queued job presents
            # cleanly until the new run overwrites them.
            UpdateExpression="SET #s = :queued, attempt_count = :ac "
                             "REMOVE last_heartbeat, progress",
            ConditionExpression=cond,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
    except ddb.exceptions.ConditionalCheckFailedException:
        print(f"reaper: {job_id} no longer a zombie (worker resumed?); skipped")
        return False

    # Send a fresh job message. The original message (held by the dead worker)
    # may still redeliver when its visibility window lapses; that stale
    # duplicate is harmless because the worker's _claim_job is conditional on
    # status == "queued" — a redelivered message for a job that has since gone
    # running/complete/failed fails the condition and is skipped. (The plain
    # status == "running" dequeue check only covers the running case; the
    # conditional claim is the load-bearing guard.) The reaper owns retry
    # counting via attempt_count rather than leaning on SQS redelivery.
    sqs.send_message(
        QueueUrl=IMAGE_QUEUE_URL if job_type == "image" else VIDEO_QUEUE_URL,
        MessageBody=json.dumps({"job_id": job_id, "type": job_type}),
        MessageAttributes={
            "job_id": {"DataType": "String", "StringValue": job_id},
            "type": {"DataType": "String", "StringValue": job_type},
        },
    )
    print(f"reaper: re-queued zombie {job_id} (attempt {attempts + 1}/{MAX_RETRIES})")
    return True


def _give_up(item: dict, attempts: int) -> bool:
    """Zombie that's burned every retry → mark failed and archive the workflow
    JSON for analysis (it keeps dying — the workflow itself is suspect)."""
    job_id = item["job_id"]["S"]
    prev_hb = item.get("last_heartbeat", {}).get("S")

    # Archive BEFORE the status flip. The S3 key is deterministic so a re-sweep
    # just overwrites; losing the analysis copy to an S3 hiccup is worse than a
    # stray archive for a job that turns out (rarely) to have resumed.
    _archive(item)

    names, values, cond = _zombie_condition(prev_hb)
    names["#e"] = "error"
    values[":failed"] = {"S": "failed"}
    values[":err"] = {"S": f"abandoned after {MAX_RETRIES} retries — "
                           f"likely a workflow problem (archived for analysis)"}
    values[":t"] = {"S": _now_iso()}
    try:
        ddb.update_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
            UpdateExpression="SET #s = :failed, #e = :err, completed_at = :t",
            ConditionExpression=cond,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
    except ddb.exceptions.ConditionalCheckFailedException:
        print(f"reaper: {job_id} no longer a zombie (worker resumed?); skipped")
        return False

    print(f"reaper: gave up on {job_id} after {MAX_RETRIES} retries; archived to S3")
    return True


def _zombie_condition(prev_hb: str | None) -> tuple[dict, dict, str]:
    """Condition that the record is still the zombie we read — status is
    `running` AND the heartbeat is unchanged — so we never fight a worker
    that resumed between the sweep's read and this write."""
    names = {"#s": "status"}
    values: dict = {":running": {"S": "running"}}
    cond = "#s = :running"
    if prev_hb:
        values[":hb"] = {"S": prev_hb}
        cond += " AND last_heartbeat = :hb"
    else:
        cond += " AND attribute_not_exists(last_heartbeat)"
    return names, values, cond


def _archive(item: dict) -> None:
    """Write the failed job's workflow + diagnostics to the analysis bucket."""
    job_id = item["job_id"]["S"]
    record: dict = {
        "job_id": job_id,
        "type": item.get("type", {}).get("S", ""),
        "created_at": item.get("created_at", {}).get("S", ""),
        "started_at": item.get("started_at", {}).get("S", ""),
        "attempts": int(item.get("attempt_count", {}).get("N", "0")),
        "archived_at": _now_iso(),
    }
    # workflow_json is NOT on the jobs-by-status GSI (the read-cost reproject dropped
    # it), and `item` here came from a GSI query — so fetch the graph with a
    # targeted base-table GetItem rather than reading it off `item` (which would
    # archive an empty workflow). One extra read, only on jobs that exhausted all
    # retries (rare).
    try:
        got = ddb.get_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
            ProjectionExpression="workflow_json",
        )
        raw = got.get("Item", {}).get("workflow_json", {}).get("S", "")
    except ClientError as e:
        print(f"reaper: could not fetch workflow_json for {job_id} to archive: {e!r}")
        raw = ""
    try:
        record["workflow"] = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        record["workflow_json"] = raw  # keep it raw if it won't parse
    key = f"{datetime.now(timezone.utc):%Y/%m/%d}/{job_id}.json"
    s3.put_object(
        Bucket=FAILED_WORKFLOWS_BUCKET,
        Key=key,
        Body=json.dumps(record, indent=2).encode(),
        ContentType="application/json",
    )


def _query_status(status: str) -> list:
    """All job records in a given status, via the jobs-by-status GSI."""
    items: list = []
    kwargs: dict = {}
    while True:
        r = ddb.query(
            TableName=JOBS_TABLE,
            IndexName=STATUS_INDEX,
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": {"S": status}},
            **kwargs,
        )
        items.extend(r.get("Items", []))
        lek = r.get("LastEvaluatedKey")
        if not lek:
            return items
        kwargs["ExclusiveStartKey"] = lek


def _count_status(status: str) -> int:
    """How many job records are in a given status — Select=COUNT, so the
    queued backlog is sized without materialising every item."""
    total = 0
    kwargs: dict = {}
    while True:
        r = ddb.query(
            TableName=JOBS_TABLE,
            IndexName=STATUS_INDEX,
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": {"S": status}},
            Select="COUNT",
            **kwargs,
        )
        total += r.get("Count", 0)
        lek = r.get("LastEvaluatedKey")
        if not lek:
            return total
        kwargs["ExclusiveStartKey"] = lek


def _arm(delay_sec: int) -> None:
    """Send the next delayed tick so the reaper runs again while work remains."""
    sqs.send_message(
        QueueUrl=REAPER_QUEUE_URL,
        MessageBody=json.dumps({"tick": _now_iso()}),
        DelaySeconds=delay_sec,
    )


def _epoch(iso: str) -> float | None:
    """ISO-8601 timestamp → epoch seconds, or None if it can't be parsed."""
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
