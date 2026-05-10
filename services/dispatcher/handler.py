"""
Dispatcher Lambda — entry point for ComfyUI workflow submission.

Routes:
  POST /prompt              → enqueue to image-jobs or video-jobs SQS queue
  GET  /history/{prompt_id} → return job state from DDB (ComfyUI-compatible shape)
  GET  /object_info         → merged ComfyUI /object_info from cached fleet data,
                               with model dropdowns sourced from DDB catalog
  POST /internal/object_info→ worker pushes its /object_info on boot (cached in DDB)

Authentication:
  All routes are guarded by API Gateway's Cognito authorizer. Verified claims
  arrive in event["requestContext"]["authorizer"]["claims"]. We don't need to
  verify the JWT ourselves.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from workflow_router import classify_workflow

# AWS clients are created at module load (re-used across warm invocations).
sqs = boto3.client("sqs")
ddb = boto3.client("dynamodb")

JOBS_TABLE = os.environ["JOBS_TABLE"]
MODELS_TABLE = os.environ["MODELS_TABLE"]
OBJECT_INFO_TABLE = os.environ["OBJECT_INFO_TABLE"]
IMAGE_QUEUE_URL = os.environ["IMAGE_QUEUE_URL"]
VIDEO_QUEUE_URL = os.environ["VIDEO_QUEUE_URL"]

# Job records expire from DDB after 30 days (TTL attribute).
JOB_TTL = timedelta(days=30)

# In-memory cache for /object_info responses to avoid hammering DDB on every
# poll. TTL is 60s — when a model is added via the catalog Lambda, that Lambda
# also bumps a version key in DDB so we can detect changes proactively.
_object_info_cache: dict[str, Any] = {"data": None, "fetched_at": 0.0}
OBJECT_INFO_CACHE_TTL_SECONDS = 60


def lambda_handler(event: dict, context: Any) -> dict:
    method = event["httpMethod"]
    path = event["resource"]

    try:
        if method == "POST" and path == "/prompt":
            return _post_prompt(event)
        if method == "GET" and path == "/history/{id}":
            return _get_history(event)
        if method == "GET" and path == "/object_info":
            return _get_object_info(event)
        if method == "POST" and path == "/internal/object_info":
            return _post_internal_object_info(event)
        # Stub endpoints that ComfyUI's frontend polls but we don't need to
        # implement fully (resolves code review N16).
        if method == "GET" and path == "/queue":
            return _resp(200, {"queue_running": [], "queue_pending": []})
        if method == "GET" and path == "/system_stats":
            return _resp(200, {
                "system": {"comfyui_version": "remote", "ram_total": 0, "ram_free": 0},
                "devices": [],
            })
        if method == "GET" and path == "/embeddings":
            return _resp(200, [])
        if method == "GET" and path == "/extensions":
            return _resp(200, [])
        return _resp(404, {"error": "not found"})
    except Exception as e:  # noqa: BLE001 — return JSON to client, log to CW
        print(f"ERROR {method} {path}: {e!r}")
        return _resp(500, {"error": "internal error"})


# ----- POST /prompt -----
def _post_prompt(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")

    workflow = body.get("prompt") or body.get("workflow")
    if not workflow or not isinstance(workflow, dict):
        return _resp(400, {"error": "missing or invalid 'prompt' (workflow JSON)"})

    explicit_type = body.get("type")
    if explicit_type in ("image", "video"):
        job_type = explicit_type
    else:
        # v3 C3 fallback: if frontend didn't specify, sniff workflow class_types.
        job_type = classify_workflow(workflow)

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expire_at = int((now + JOB_TTL).timestamp())

    queue_url = IMAGE_QUEUE_URL if job_type == "image" else VIDEO_QUEUE_URL

    # Write job record FIRST so even if SQS send fails the job is recoverable.
    ddb.put_item(
        TableName=JOBS_TABLE,
        Item={
            "job_id": {"S": job_id},
            "type": {"S": job_type},
            "status": {"S": "queued"},
            "workflow_json": {"S": json.dumps(workflow)},
            "client_id": {"S": body.get("client_id", "")},
            "created_at": {"S": now.isoformat()},
            "expire_at": {"N": str(expire_at)},
            "attempt_count": {"N": "0"},
        },
    )

    # SQS message: small (just job_id + type). Worker reads full workflow from DDB.
    # Pure SQS body avoids hitting the 256 KB SQS message limit on big workflows.
    # If SQS send fails, mark the job failed so user gets a clear signal in /jobs/{id}
    # (resolves code review N13).
    try:
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps({"job_id": job_id, "type": job_type}),
            MessageAttributes={
                "job_id": {"DataType": "String", "StringValue": job_id},
                "type": {"DataType": "String", "StringValue": job_type},
            },
        )
    except Exception as e:  # noqa: BLE001
        ddb.update_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
            UpdateExpression="SET #s = :s, #e = :e",
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
            ExpressionAttributeValues={
                ":s": {"S": "failed"},
                ":e": {"S": f"sqs send failed: {e}"[:500]},
            },
        )
        return _resp(503, {"error": "queue temporarily unavailable; job marked failed"})

    # Match ComfyUI's native /prompt response shape so existing client libraries work.
    return _resp(
        200,
        {
            "prompt_id": job_id,
            "number": 1,
            "node_errors": {},
        },
    )


# ----- GET /history/{id} -----
def _get_history(event: dict) -> dict:
    job_id = event["pathParameters"]["id"]
    try:
        result = ddb.get_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
        )
    except ClientError as e:
        print(f"ddb error: {e}")
        return _resp(500, {})

    item = result.get("Item")
    if not item:
        # ComfyUI returns empty {} for unknown prompt_ids
        return _resp(200, {})

    status = item["status"]["S"]
    output_keys = json.loads(item.get("output_keys", {"S": "[]"})["S"])

    # ComfyUI history shape: { "<prompt_id>": { "outputs": {...}, "status": {...} } }
    history_entry = {
        "outputs": _format_outputs(output_keys),
        "status": {
            "status_str": _comfyui_status_str(status),
            "completed": status in ("complete", "failed"),
            "messages": [],
        },
    }
    if status == "failed":
        history_entry["status"]["messages"].append(
            ["execution_error", {"message": item.get("error", {"S": ""})["S"]}]
        )

    return _resp(200, {job_id: history_entry})


def _format_outputs(s3_keys: list[str]) -> dict:
    """Convert worker's S3 output keys into ComfyUI's expected output shape.
    Each entry is grouped under a node_id; ComfyUI's frontend uses the
    'images'/'videos' arrays to render results. The frontend will then call
    /view?filename=...&type=output for the actual bytes.
    """
    images: list[dict] = []
    videos: list[dict] = []
    for key in s3_keys:
        filename = key.rsplit("/", 1)[-1]
        ref = {"filename": filename, "subfolder": "", "type": "output", "s3_key": key}
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("mp4", "webm", "gif", "webp", "mkv", "mov"):
            videos.append(ref)
        else:
            images.append(ref)

    # ComfyUI expects outputs keyed by node_id. We use a synthetic "0" since the
    # worker doesn't preserve per-node mappings; this is a known limitation.
    out: dict[str, Any] = {"0": {}}
    if images:
        out["0"]["images"] = images
    if videos:
        out["0"]["gifs"] = videos
    return out


def _comfyui_status_str(internal_status: str) -> str:
    return {
        "queued": "executing",
        "running": "executing",
        "complete": "success",
        "failed": "error",
    }.get(internal_status, "executing")


# ----- GET /object_info -----
def _get_object_info(_event: dict) -> dict:
    """Return a merged /object_info from cached fleet data, with model
    dropdowns swapped for the live DDB catalog.
    """
    now = time.time()
    if (
        _object_info_cache["data"] is None
        or now - _object_info_cache["fetched_at"] > OBJECT_INFO_CACHE_TTL_SECONDS
    ):
        _object_info_cache["data"] = _build_object_info()
        _object_info_cache["fetched_at"] = now

    return _resp(200, _object_info_cache["data"])


def _build_object_info() -> dict:
    """Read worker-pushed /object_info from DDB, union image + video, swap
    model name lists for live DDB catalog values.
    """
    # Fetch both fleets' object_info
    merged: dict[str, Any] = {}
    for fleet in ("image", "video"):
        try:
            r = ddb.get_item(TableName=OBJECT_INFO_TABLE, Key={"fleet": {"S": fleet}})
            if "Item" in r:
                fleet_oi = json.loads(r["Item"]["object_info_json"]["S"])
                merged.update(fleet_oi)
        except ClientError:
            pass

    # Replace model dropdowns with current catalog
    catalog_by_type = _scan_catalog_by_type()
    for node_class, schema in merged.items():
        inputs = schema.get("input", {}).get("required", {})
        for input_name, spec in inputs.items():
            # ComfyUI input spec format: [<type_or_choices>, <metadata>]
            if not isinstance(spec, list) or not spec:
                continue
            choices_or_type = spec[0]
            # If it's a list of strings, it's a dropdown — possibly a model list.
            if isinstance(choices_or_type, list) and choices_or_type and all(
                isinstance(x, str) for x in choices_or_type
            ):
                model_type = _guess_model_type_from_input(node_class, input_name)
                if model_type and model_type in catalog_by_type:
                    spec[0] = sorted(catalog_by_type[model_type])

    return merged


_INPUT_TO_MODEL_TYPE = {
    "ckpt_name": "checkpoint",
    "lora_name": "lora",
    "vae_name": "vae",
    "control_net_name": "controlnet",
    "clip_name": "clip",
    "unet_name": "checkpoint",
    "model_name": "checkpoint",
}


def _guess_model_type_from_input(node_class: str, input_name: str) -> str | None:
    """Map ComfyUI input names (and node-class hints) to our DDB catalog 'type'."""
    direct = _INPUT_TO_MODEL_TYPE.get(input_name)
    if direct:
        return direct
    if "lora" in input_name.lower():
        return "lora"
    if "vae" in input_name.lower():
        return "vae"
    if "wan" in node_class.lower() and "model" in input_name.lower():
        return "checkpoint"
    return None


def _scan_catalog_by_type() -> dict[str, list[str]]:
    """Pull all models from DDB catalog grouped by type."""
    grouped: dict[str, list[str]] = {}
    paginator = ddb.get_paginator("scan")
    for page in paginator.paginate(TableName=MODELS_TABLE):
        for item in page.get("Items", []):
            t = item.get("type", {}).get("S", "")
            n = item.get("name", {}).get("S", "")
            if t and n:
                grouped.setdefault(t, []).append(n)
    return grouped


# ----- POST /internal/object_info (worker → dispatcher) -----
def _post_internal_object_info(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")
    fleet = body.get("fleet")
    object_info = body.get("object_info")
    if fleet not in ("image", "video") or not isinstance(object_info, dict):
        return _resp(400, {"error": "fleet must be image|video, object_info must be dict"})

    ddb.put_item(
        TableName=OBJECT_INFO_TABLE,
        Item={
            "fleet": {"S": fleet},
            "object_info_json": {"S": json.dumps(object_info)},
            "updated_at": {"S": datetime.now(timezone.utc).isoformat()},
        },
    )
    # Invalidate this Lambda's in-memory cache so next /object_info re-fetches.
    _object_info_cache["data"] = None
    return _resp(200, {"ok": True})


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
