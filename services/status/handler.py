"""
Status Lambda — read-only job lookup + presigned S3 URL generation.

Routes:
  GET  /jobs             → list jobs filtered by status (editor queue/history)
  GET  /jobs/{id}        → job record from DDB
  GET  /view?key=<s3>    → 302 redirect to presigned GET URL on outputs bucket
  POST /upload/image     → returns presigned PUT URL for direct browser upload
                            to uploads bucket (sidesteps Lambda 6 MB body limit)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ddb = boto3.client("dynamodb")
s3 = boto3.client("s3", config=Config(signature_version="s3v4"))

JOBS_TABLE = os.environ["JOBS_TABLE"]
OUTPUTS_BUCKET = os.environ["OUTPUTS_BUCKET"]
UPLOADS_BUCKET = os.environ["UPLOADS_BUCKET"]

PRESIGNED_GET_TTL = 3600  # 1 h — lets the viewer reuse a URL (browser image
# cache stays warm) across reloads instead of re-presigning every render.
PRESIGNED_PUT_TTL = 600  # 10 min for upload


def lambda_handler(event: dict, _context: Any) -> dict:
    method = event["httpMethod"]
    path = event["resource"]

    try:
        if method == "GET" and path == "/jobs":
            return _list_jobs(event)
        if method == "GET" and path == "/jobs/{id}":
            return _get_job(event)
        if method == "DELETE" and path == "/jobs/{id}":
            return _delete_job(event)
        if method == "GET" and path == "/view":
            return _view(event)
        if method == "POST" and path == "/upload/image":
            return _upload(event)
        return _resp(404, {"error": "not found"})
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {method} {path}: {e!r}")
        return _resp(500, {"error": "internal error"})


def _extract_model(workflow_json: str) -> str:
    """Best-effort: the primary model a workflow used — the *biggest* one.

    A workflow loads many models (LoRA, VAE, CLIP, ControlNet…). The main
    model is either a checkpoint (CheckpointLoaderSimple → ckpt_name) or, for
    Flux/Wan-style graphs, a standalone diffusion model (UNETLoader /
    UnetLoaderGGUF / a 'Load Diffusion Model' node → unet_name). We scan for
    either, skipping the auxiliary loaders, and prefer a checkpoint."""
    if not workflow_json:
        return ""
    try:
        wf = json.loads(workflow_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(wf, dict):
        return ""
    ckpt = diffusion = None
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").lower()
        inp = node.get("inputs", {}) or {}
        # Skip auxiliary loaders — none of these is "the model".
        if any(x in ct for x in ("lora", "vae", "clip", "controlnet",
                                 "upscale", "ipadapter", "style", "embedding")):
            continue
        if "checkpoint" in ct and ckpt is None:
            ckpt = inp.get("ckpt_name") or inp.get("model_name")
        elif ("unet" in ct or "diffusion" in ct) and diffusion is None:
            diffusion = inp.get("unet_name") or inp.get("model_name") or inp.get("model")
    name = ckpt or diffusion
    if not name or not isinstance(name, str):
        return ""
    # strip directory + extension for a clean display name
    return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _list_jobs(event: dict) -> dict:
    """GET /jobs?status=completed,failed,cancelled&limit=64&offset=0

    Editor's new queue/history menu paginates by status. We Query the
    status-index GSI once per requested status, merge results, sort by
    created_at desc, then apply offset/limit in Python. Volume is small
    (1 user, ~120 jobs/day) so the per-status Query is fine.
    """
    qs = event.get("queryStringParameters") or {}
    statuses = [s.strip() for s in (qs.get("status") or "").split(",") if s.strip()]
    if not statuses:
        statuses = ["completed", "failed", "cancelled", "in_progress", "pending"]

    try:
        limit = max(1, min(int(qs.get("limit") or 64), 500))
    except ValueError:
        limit = 64
    try:
        offset = max(0, int(qs.get("offset") or 0))
    except ValueError:
        offset = 0

    items: list[dict] = []
    for status in statuses:
        r = ddb.query(
            TableName=JOBS_TABLE,
            IndexName="status-index",
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": {"S": status}},
            ScanIndexForward=False,  # newest first
            Limit=offset + limit,
        )
        items.extend(r.get("Items", []))

    items.sort(key=lambda it: it.get("created_at", {}).get("S", ""), reverse=True)
    page = items[offset : offset + limit]

    jobs = [
        {
            "job_id": it["job_id"]["S"],
            "type": it.get("type", {"S": ""})["S"],
            "status": it.get("status", {"S": ""})["S"],
            "created_at": it.get("created_at", {"S": ""})["S"],
            "started_at": it.get("started_at", {"S": ""})["S"],
            "completed_at": it.get("completed_at", {"S": ""})["S"],
            "output_keys": json.loads(it.get("output_keys", {"S": "[]"})["S"]),
            "error": it.get("error", {"S": ""})["S"],
            "model": _extract_model(it.get("workflow_json", {}).get("S", "")),
            "progress": it.get("progress", {}).get("S", ""),
        }
        for it in page
    ]
    return _resp(200, {"jobs": jobs, "limit": limit, "offset": offset, "total": len(items)})


def _get_job(event: dict) -> dict:
    job_id = event["pathParameters"]["id"]
    r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    if "Item" not in r:
        return _resp(404, {"error": "job not found"})

    item = r["Item"]
    output = {
        "job_id": item["job_id"]["S"],
        "type": item.get("type", {"S": ""})["S"],
        "status": item.get("status", {"S": ""})["S"],
        "created_at": item.get("created_at", {"S": ""})["S"],
        "started_at": item.get("started_at", {"S": ""})["S"],
        "completed_at": item.get("completed_at", {"S": ""})["S"],
        "attempt_count": int(item.get("attempt_count", {"N": "0"})["N"]),
        "output_keys": json.loads(item.get("output_keys", {"S": "[]"})["S"]),
        "error": item.get("error", {"S": ""})["S"],
        "model": _extract_model(item.get("workflow_json", {}).get("S", "")),
        "progress": item.get("progress", {}).get("S", ""),
    }
    return _resp(200, output)


def _delete_job(event: dict) -> dict:
    """DELETE /jobs/{id} — delete a generation: its output objects from the
    outputs bucket, then the job record from DynamoDB so it drops out of the
    viewer. Used by the viewer's per-tile delete button."""
    job_id = event["pathParameters"]["id"]
    r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    if "Item" not in r:
        return _resp(404, {"error": "job not found"})
    keys = json.loads(r["Item"].get("output_keys", {"S": "[]"})["S"])
    deleted = 0
    for key in keys:
        try:
            s3.delete_object(Bucket=OUTPUTS_BUCKET, Key=key)
            deleted += 1
        except ClientError as e:  # noqa: PERF203
            print(f"delete object {key} failed: {e!r}")
    ddb.delete_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    return _resp(200, {"deleted": job_id, "objects": deleted})


def _view(event: dict) -> dict:
    """Return a presigned URL for an output S3 key.

    Default: 302 redirect — an <img>/<video> whose src points here follows it
    straight to S3.
    With ?json=1: 200 + {"url": ...}. A cross-origin fetch() cannot read a
    302's Location (opaque-redirect responses hide headers, and the 302
    carries no CORS header), so JS callers that fetch this — e.g. the
    standalone viewer page — must use the JSON form and set the <img> src
    to the returned URL themselves.
    """
    qs = event.get("queryStringParameters") or {}
    key = qs.get("key")
    if not key:
        return _resp(400, {"error": "missing key parameter"})

    # SECURITY: only allow keys under outputs bucket. The key is opaque to us
    # but we don't expose models/uploads buckets through this path.
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": OUTPUTS_BUCKET, "Key": key},
            ExpiresIn=PRESIGNED_GET_TTL,
        )
    except ClientError as e:
        print(f"presign error: {e}")
        return _resp(500, {})
    if qs.get("json") in ("1", "true"):
        return _resp(200, {"url": url})
    return {
        "statusCode": 302,
        "headers": {"Location": url, "Cache-Control": "no-store"},
        "body": "",
    }


def _upload(event: dict) -> dict:
    """Return a presigned PUT URL for the browser to upload directly to S3.
    Avoids passing image bytes through Lambda (6 MB limit)."""
    body = json.loads(event.get("body") or "{}")
    content_type = body.get("content_type", "image/png")
    upload_id = str(uuid.uuid4())
    key = f"uploads/{datetime.now(timezone.utc):%Y/%m/%d}/{upload_id}"

    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": UPLOADS_BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=PRESIGNED_PUT_TTL,
    )
    return _resp(200, {"upload_url": url, "key": key, "bucket": UPLOADS_BUCKET})


def _resp(status: int, body: Any) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body),
    }
