"""
WebSocket forward Lambda — REST endpoint POST /internal/ws-event.

Workers call this to push a ComfyUI event (status, executing, progress,
executed) to the connected browser. The Lambda:
  1. Looks up the connection_id by client_id in the connections table GSI
  2. Calls API GW Management API's PostToConnection to push the event
  3. If the connection is gone (410), removes it from the table

Auth: API key (same scheme as /internal/object_info — workers can't easily
do Cognito).

Request body: {"client_id": "...", "type": "executing", "data": {...}}
Response: {"delivered": true|false, "reason": "..."}
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ddb = boto3.client("dynamodb")
CONNECTIONS_TABLE = os.environ["CONNECTIONS_TABLE"]
WS_API_ENDPOINT = os.environ["WS_API_ENDPOINT"]  # e.g., https://abc123.execute-api.us-west-2.amazonaws.com/v1

# apigatewaymanagementapi client targets a specific stage's WS API endpoint.
ws_api = boto3.client(
    "apigatewaymanagementapi",
    endpoint_url=WS_API_ENDPOINT,
    config=Config(retries={"max_attempts": 2}),
)


def lambda_handler(event: dict, _context: Any) -> dict:
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _resp(400, {"error": "invalid json"})

    client_id = body.get("client_id")
    if not client_id:
        return _resp(400, {"error": "client_id required"})

    # Find the connection_id(s) for this client. Multiple browsers might be
    # connected with the same client_id (refresh, multiple tabs); fan out.
    try:
        r = ddb.query(
            TableName=CONNECTIONS_TABLE,
            IndexName="client-id-index",
            KeyConditionExpression="client_id = :c",
            ExpressionAttributeValues={":c": {"S": client_id}},
        )
    except ClientError as e:
        print(f"ddb query failed: {e}")
        return _resp(500, {"error": "lookup failed"})

    items = r.get("Items", [])
    if not items:
        return _resp(404, {"delivered": False, "reason": "no active connection for client_id"})

    # Strip our routing wrapper; forward only the ComfyUI event payload.
    payload = {k: v for k, v in body.items() if k != "client_id"}
    payload_bytes = json.dumps(payload).encode("utf-8")

    delivered = 0
    stale: list[str] = []
    for item in items:
        cid = item["connection_id"]["S"]
        try:
            ws_api.post_to_connection(ConnectionId=cid, Data=payload_bytes)
            delivered += 1
        except ws_api.exceptions.GoneException:
            stale.append(cid)
        except ClientError as e:
            print(f"post_to_connection failed for {cid}: {e}")

    # Clean up dead connections so we don't keep trying them.
    for cid in stale:
        try:
            ddb.delete_item(
                TableName=CONNECTIONS_TABLE,
                Key={"connection_id": {"S": cid}},
            )
        except ClientError:
            pass

    return _resp(200, {"delivered": delivered, "stale_pruned": len(stale)})


def _resp(status: int, body: Any) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
