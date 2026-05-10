"""
Status Lambda — read-only job lookup + presigned S3 URL generation.

Routes:
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

PRESIGNED_GET_TTL = 300  # 5 min — long enough to download, short enough to limit replay
PRESIGNED_PUT_TTL = 600  # 10 min for upload


def lambda_handler(event: dict, _context: Any) -> dict:
    method = event["httpMethod"]
    path = event["resource"]

    try:
        if method == "GET" and path == "/jobs/{id}":
            return _get_job(event)
        if method == "GET" and path == "/view":
            return _view(event)
        if method == "POST" and path == "/upload/image":
            return _upload(event)
        return _resp(404, {"error": "not found"})
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {method} {path}: {e!r}")
        return _resp(500, {"error": "internal error"})


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
    }
    return _resp(200, output)


def _view(event: dict) -> dict:
    """Return a presigned URL for an output S3 key. Use 302 redirect for
    direct download/display in browser."""
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
