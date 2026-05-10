"""
Catalog Lambda — model catalog CRUD against DynamoDB.

Routes:
  GET    /models?type=<t>   → list models, optionally filtered by type
  POST   /models            → manual add (name, type, s3_key, size_gb)
  DELETE /models/{name}     → remove from catalog (does NOT delete from S3)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

ddb = boto3.client("dynamodb")
MODELS_TABLE = os.environ["MODELS_TABLE"]


def lambda_handler(event: dict, _context: Any) -> dict:
    method = event["httpMethod"]
    path = event["resource"]
    try:
        if method == "GET" and path == "/models":
            return _list_models(event)
        if method == "POST" and path == "/models":
            return _add_model(event)
        if method == "DELETE" and path == "/models/{name}":
            return _delete_model(event)
        return _resp(404, {"error": "not found"})
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {method} {path}: {e!r}")
        return _resp(500, {"error": "internal error"})


def _list_models(event: dict) -> dict:
    qs = event.get("queryStringParameters") or {}
    model_type = qs.get("type")

    items: list[dict] = []
    if model_type:
        # Use the type GSI for efficient query.
        paginator = ddb.get_paginator("query")
        for page in paginator.paginate(
            TableName=MODELS_TABLE,
            IndexName="type-index",
            KeyConditionExpression="#t = :t",
            ExpressionAttributeNames={"#t": "type"},
            ExpressionAttributeValues={":t": {"S": model_type}},
        ):
            items.extend(page.get("Items", []))
    else:
        paginator = ddb.get_paginator("scan")
        for page in paginator.paginate(TableName=MODELS_TABLE):
            items.extend(page.get("Items", []))

    out = [_unmarshal(item) for item in items]
    out.sort(key=lambda m: (m["type"], m["name"]))
    return _resp(200, {"models": out, "count": len(out)})


def _add_model(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")
    required = ("name", "type", "s3_key")
    if not all(k in body and body[k] for k in required):
        return _resp(400, {"error": f"required fields: {required}"})

    item: dict[str, Any] = {
        "name": {"S": body["name"]},
        "type": {"S": body["type"]},
        "s3_key": {"S": body["s3_key"]},
        "size_gb": {"N": str(float(body.get("size_gb", 0.0)))},
        "pinned": {"BOOL": bool(body.get("pinned", False))},
        "added_at": {"S": datetime.now(timezone.utc).isoformat()},
    }
    if body.get("preview_url"):
        item["preview_url"] = {"S": body["preview_url"]}
    if body.get("civitai_version_id"):
        item["civitai_version_id"] = {"N": str(int(body["civitai_version_id"]))}

    try:
        ddb.put_item(
            TableName=MODELS_TABLE,
            Item=item,
            ConditionExpression="attribute_not_exists(#n)",
            ExpressionAttributeNames={"#n": "name"},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return _resp(409, {"error": f"model {body['name']} already exists"})
        raise

    return _resp(201, {"name": body["name"], "added": True})


def _delete_model(event: dict) -> dict:
    name = event["pathParameters"]["name"]
    ddb.delete_item(
        TableName=MODELS_TABLE,
        Key={"name": {"S": name}},
    )
    return _resp(200, {"name": name, "deleted": True})


def _unmarshal(item: dict) -> dict:
    return {
        "name": item.get("name", {"S": ""})["S"],
        "type": item.get("type", {"S": ""})["S"],
        "s3_key": item.get("s3_key", {"S": ""})["S"],
        "size_gb": float(item.get("size_gb", {"N": "0"})["N"]),
        "pinned": item.get("pinned", {"BOOL": False})["BOOL"],
        "preview_url": item.get("preview_url", {"S": ""})["S"],
        "civitai_version_id": int(item.get("civitai_version_id", {"N": "0"})["N"]),
        "added_at": item.get("added_at", {"S": ""})["S"],
    }


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
