"""
WebSocket control Lambda — handles $connect and $disconnect routes.

On $connect:
  - Validates the HMAC ticket from query string ?ticket=<base64.payload.sig>
  - Extracts username + client_id from the ticket payload
  - Stores (connection_id, client_id, username) in DDB connections table
  - Returns 200 to allow the connection

On $disconnect:
  - Removes the connection_id row from the table

Tickets are issued by the dispatcher Lambda's /ws-ticket REST endpoint
(Cognito-authed). They live for 60 seconds — long enough to open a WS,
short enough to limit replay risk.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import boto3

ddb = boto3.client("dynamodb")
sm = boto3.client("secretsmanager")

CONNECTIONS_TABLE = os.environ["CONNECTIONS_TABLE"]
WS_TICKET_SECRET_ARN = os.environ["WS_TICKET_SECRET_ARN"]

# Lazy cache for the HMAC secret across warm invocations.
_secret_cache: bytes | None = None


def _get_hmac_secret() -> bytes:
    global _secret_cache
    if _secret_cache is None:
        r = sm.get_secret_value(SecretId=WS_TICKET_SECRET_ARN)
        # Secrets Manager returns the auto-generated string as-is.
        _secret_cache = r["SecretString"].encode("utf-8")
    return _secret_cache


def _b64url_decode(s: str) -> bytes:
    pad = 4 - (len(s) % 4)
    if pad < 4:
        s = s + ("=" * pad)
    return base64.urlsafe_b64decode(s)


def _verify_ticket(ticket: str) -> dict | None:
    """Returns the payload dict if valid, None if not.
    Ticket format: <base64url(payload_json)>.<base64url(hmac_sha256)>
    """
    if not ticket or "." not in ticket:
        return None
    try:
        payload_b64, sig_b64 = ticket.rsplit(".", 1)
        expected_sig = hmac.new(
            _get_hmac_secret(),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        provided_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:  # noqa: BLE001
        return None


def lambda_handler(event: dict, _context: Any) -> dict:
    route = event.get("requestContext", {}).get("routeKey", "")
    connection_id = event.get("requestContext", {}).get("connectionId", "")

    if route == "$connect":
        return _on_connect(event, connection_id)
    if route == "$disconnect":
        return _on_disconnect(connection_id)
    # We don't define other routes (server-side push only); reject any
    # unexpected client-sent message politely.
    return {"statusCode": 400, "body": json.dumps({"error": "unsupported route"})}


def _on_connect(event: dict, connection_id: str) -> dict:
    qs = event.get("queryStringParameters") or {}
    ticket = qs.get("ticket", "")
    client_id = qs.get("client_id", "")

    payload = _verify_ticket(ticket)
    if not payload:
        print(f"reject connect: invalid/expired ticket for connection {connection_id}")
        return {"statusCode": 401, "body": "invalid ticket"}

    username = payload.get("sub", "anonymous")
    # Use client_id from query if provided, else from the ticket payload.
    client_id = client_id or payload.get("client_id", connection_id)

    # 6-hour TTL; API GW disconnects idle connections after 10 min anyway,
    # this is just to clean up zombies if $disconnect doesn't fire.
    ttl = int(time.time()) + (6 * 3600)

    ddb.put_item(
        TableName=CONNECTIONS_TABLE,
        Item={
            "connection_id": {"S": connection_id},
            "client_id": {"S": client_id},
            "username": {"S": username},
            "connected_at": {"S": datetime.now(timezone.utc).isoformat()},
            "ttl": {"N": str(ttl)},
        },
    )
    print(f"connected: user={username} client={client_id} conn={connection_id}")
    return {"statusCode": 200, "body": "connected"}


def _on_disconnect(connection_id: str) -> dict:
    try:
        ddb.delete_item(
            TableName=CONNECTIONS_TABLE,
            Key={"connection_id": {"S": connection_id}},
        )
    except Exception:  # noqa: BLE001
        pass
    return {"statusCode": 200, "body": "disconnected"}
