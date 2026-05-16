"""
Push the local ComfyUI's /object_info to the dispatcher API on boot.

This is what makes the dispatcher's /object_info endpoint return real
node definitions (which the frontend needs to render workflow editors).
Without this, the dispatcher would have to cold-start a worker just to
serve /object_info to the browser.

Requires the worker API key (looked up from API GW by WORKER_API_KEY_ID env
var) since /internal/object_info uses API-key auth, not Cognito.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import urllib3

log = logging.getLogger(__name__)
_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10.0, read=30.0))

WORKER_API_KEY_ID = os.environ.get("WORKER_API_KEY_ID", "")
_api_key_cache: Optional[str] = None


def _get_api_key() -> str:
    """One-shot fetch of the API GW key value, cached across calls."""
    global _api_key_cache
    if _api_key_cache or not WORKER_API_KEY_ID:
        return _api_key_cache or ""
    try:
        import boto3
        # Explicit region — AWS_REGION env not always honored at client
        # construction for apigateway service (EndpointResolver failure).
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        client = boto3.client("apigateway", region_name=region)
        r = client.get_api_key(apiKey=WORKER_API_KEY_ID, includeValue=True)
        _api_key_cache = r.get("value", "")
    except Exception:  # noqa: BLE001
        log.exception("failed to fetch worker API key value")
    return _api_key_cache or ""


def publish_object_info(fleet: str, object_info: dict[str, Any]) -> bool:
    """POST to dispatcher's internal endpoint. Returns success bool."""
    api_url = os.environ.get("DISPATCHER_API_URL", "").rstrip("/")
    if not api_url:
        log.warning("DISPATCHER_API_URL unset; skipping object_info publish")
        return False

    body = json.dumps({"fleet": fleet, "object_info": object_info}).encode()
    api_key = _get_api_key()
    if not api_key:
        log.warning("worker API key unavailable; /internal/object_info will likely 403")
    try:
        r = _http.request(
            "POST",
            f"{api_url}/internal/object_info",
            body=body,
            headers={"Content-Type": "application/json", "x-api-key": api_key},
        )
        if r.status >= 400:
            log.error("object_info publish failed: %s %s", r.status, r.data[:200])
            return False
        log.info("published object_info for fleet=%s", fleet)
        return True
    except Exception:  # noqa: BLE001
        log.exception("object_info publish exception")
        return False
