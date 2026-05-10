"""
Push the local ComfyUI's /object_info to the dispatcher API on boot.

This is what makes the dispatcher's /object_info endpoint return real
node definitions (which the frontend needs to render workflow editors).
Without this, the dispatcher would have to cold-start a worker just to
serve /object_info to the browser.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import urllib3

log = logging.getLogger(__name__)
_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10.0, read=30.0))


def publish_object_info(fleet: str, object_info: dict[str, Any]) -> bool:
    """POST to dispatcher's internal endpoint. Returns success bool."""
    api_url = os.environ.get("DISPATCHER_API_URL", "").rstrip("/")
    if not api_url:
        log.warning("DISPATCHER_API_URL unset; skipping object_info publish")
        return False

    body = json.dumps({"fleet": fleet, "object_info": object_info}).encode()
    try:
        r = _http.request(
            "POST",
            f"{api_url}/internal/object_info",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        if r.status >= 400:
            log.error("object_info publish failed: %s %s", r.status, r.data[:200])
            return False
        log.info("published object_info for fleet=%s", fleet)
        return True
    except Exception:  # noqa: BLE001
        log.exception("object_info publish exception")
        return False
