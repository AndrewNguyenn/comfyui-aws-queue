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


_PROMPT_MAX = 2000  # cap each prompt — keeps the /jobs list response bounded


def _extract_prompts(workflow_json: str) -> tuple[str, str]:
    """Best-effort: the positive + negative text prompts a workflow used.

    ComfyUI's API-format graph wires two CONDITIONING chains into a sampler's
    `positive` / `negative` inputs (or a guider's `conditioning`). The text
    itself lives in a CLIPTextEncode node's `inputs.text`. We find the sampler,
    then walk each conditioning input upstream — following only conditioning-
    *named* inputs so a ControlNet/Concat node can't leak the other polarity's
    text — until we hit a node carrying a `text` string. Returns ("", "") when
    nothing is recoverable (no workflow, an unusual graph, etc.)."""
    if not workflow_json:
        return "", ""
    try:
        wf = json.loads(workflow_json)
    except (json.JSONDecodeError, TypeError):
        return "", ""
    if not isinstance(wf, dict):
        return "", ""

    def _node(ref: Any) -> dict | None:
        # A link is [node_id, output_index]; node_id is a str in API format.
        if not (isinstance(ref, list) and ref):
            return None
        n = wf.get(str(ref[0]))
        return n if isinstance(n, dict) else None

    def _resolve(ref: Any, polarity: str, seen: set[str], depth: int = 0) -> str:
        node = _node(ref)
        if node is None or depth > 8:
            return ""
        nid = str(ref[0])
        if nid in seen:
            return ""
        seen.add(nid)
        inp = node.get("inputs", {}) or {}
        # Direct text — the CLIPTextEncode case (also SDXL's text_g / text_l).
        for key in ("text", "text_g", "text_l"):
            val = inp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Otherwise walk upstream, but only through conditioning-named inputs so
        # a ControlNetApplyAdvanced (has both `positive` and `negative`) can't
        # cross polarities. `seen` is copied per branch so it guards against
        # path cycles without letting one branch starve a sibling that shares
        # an upstream node (diamond-shaped conditioning graphs).
        for key, val in inp.items():
            if "cond" not in key.lower() and key != polarity:
                continue
            got = _resolve(val, polarity, set(seen), depth + 1)
            if got:
                return got
        return ""

    samplers = [
        n for n in wf.values()
        if isinstance(n, dict)
        and any(t in (n.get("class_type") or "").lower() for t in ("sampler", "guider"))
    ]
    pos = neg = ""
    for node in samplers:
        inp = node.get("inputs", {}) or {}
        p_ref = inp.get("positive") or inp.get("conditioning")
        p = _resolve(p_ref, "positive", set()) if p_ref else ""
        if p:
            pos = p
            n_ref = inp.get("negative")
            neg = _resolve(n_ref, "negative", set()) if n_ref else ""
            break

    # Fallback: no sampler resolved — read CLIPTextEncode nodes directly, in
    # graph order (first = positive, second = negative).
    if not pos:
        texts = []
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            if "cliptextencode" in (node.get("class_type") or "").lower():
                t = (node.get("inputs", {}) or {}).get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        if texts:
            pos = texts[0]
            if len(texts) > 1 and not neg:
                neg = texts[1]

    return pos[:_PROMPT_MAX], neg[:_PROMPT_MAX]


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

    jobs = [_serialize_job(it) for it in page]
    return _resp(200, {"jobs": jobs, "limit": limit, "offset": offset, "total": len(items)})


def _serialize_job(it: dict) -> dict:
    """Shape a raw DDB job item into the API job object. `model` and the
    positive/negative prompts are derived on the fly from the stored
    `workflow_json` (the job record itself never carries them)."""
    workflow_json = it.get("workflow_json", {}).get("S", "")
    positive, negative = _extract_prompts(workflow_json)
    return {
        "job_id": it["job_id"]["S"],
        "type": it.get("type", {"S": ""})["S"],
        "status": it.get("status", {"S": ""})["S"],
        "created_at": it.get("created_at", {"S": ""})["S"],
        "started_at": it.get("started_at", {"S": ""})["S"],
        "completed_at": it.get("completed_at", {"S": ""})["S"],
        "output_keys": json.loads(it.get("output_keys", {"S": "[]"})["S"]),
        "error": it.get("error", {"S": ""})["S"],
        "model": _extract_model(workflow_json),
        "positive_prompt": positive,
        "negative_prompt": negative,
        "progress": it.get("progress", {}).get("S", ""),
    }


def _get_job(event: dict) -> dict:
    job_id = event["pathParameters"]["id"]
    r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    if "Item" not in r:
        return _resp(404, {"error": "job not found"})

    item = r["Item"]
    output = _serialize_job(item)
    output["attempt_count"] = int(item.get("attempt_count", {"N": "0"})["N"])
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
