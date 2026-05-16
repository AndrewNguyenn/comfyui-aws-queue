"""
ComfyUI custom-node extensions publisher.

ComfyUI custom nodes (e.g., ComfyUI-Manager, WanVideoWrapper) register JS
extensions that the editor loads to render their UI. Vanilla ComfyUI serves
these at GET /extensions (returns a list) and GET /extensions/<path> (the
JS files themselves).

In our split architecture, the editor's frontend lives in S3 and the worker
runs ComfyUI ephemerally. To make the editor see custom-node UIs, on worker
boot we:
  1. Fetch the list of extension paths from local ComfyUI's /extensions.
  2. Download each JS file from local ComfyUI.
  3. Upload them to the frontend S3 bucket at the same paths.
  4. POST the list to dispatcher's /internal/extensions which caches it
     in DDB so /extensions can return it without a worker present.

This is one-shot at boot. Re-uploads on every worker boot are idempotent
(overwrites with same content unless the worker image is newer). Cost is
trivial — extension JS files are tens to hundreds of KB total.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import boto3
import urllib3

log = logging.getLogger(__name__)
_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5.0, read=15.0))
_s3 = boto3.client("s3")

FRONTEND_BUCKET = os.environ.get("FRONTEND_BUCKET", "")
DISPATCHER_API_URL = os.environ.get("DISPATCHER_API_URL", "").rstrip("/")
WORKER_API_KEY_ID = os.environ.get("WORKER_API_KEY_ID", "")
COMFY_BASE = "http://127.0.0.1:8188"

_api_key_cache: Optional[str] = None


def _get_api_key() -> str:
    global _api_key_cache
    if _api_key_cache or not WORKER_API_KEY_ID:
        return _api_key_cache or ""
    try:
        # Explicit region — AWS_REGION env var isn't always picked up for
        # apigateway service client construction, breaks with EndpointResolver.
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        client = boto3.client("apigateway", region_name=region)
        r = client.get_api_key(apiKey=WORKER_API_KEY_ID, includeValue=True)
        _api_key_cache = r.get("value", "")
    except Exception:  # noqa: BLE001
        log.exception("failed to fetch worker API key value")
    return _api_key_cache or ""


def publish_extensions(fleet: str) -> bool:
    """Returns True on success, False on failure (best-effort)."""
    if not FRONTEND_BUCKET:
        log.warning("FRONTEND_BUCKET unset; skipping extensions publish")
        return False
    if not DISPATCHER_API_URL:
        log.warning("DISPATCHER_API_URL unset; skipping extensions publish")
        return False

    # Step 1: list of extension URLs from local ComfyUI
    try:
        r = _http.request("GET", f"{COMFY_BASE}/extensions")
        if r.status != 200:
            log.error("local /extensions returned %d", r.status)
            return False
        extensions = json.loads(r.data.decode())
    except Exception:  # noqa: BLE001
        log.exception("failed to fetch local /extensions")
        return False

    if not isinstance(extensions, list):
        log.error("local /extensions returned non-list: %r", type(extensions))
        return False

    log.info("found %d extension paths to publish", len(extensions))

    # Step 2/3: download each, upload to S3 frontend bucket
    uploaded: list[str] = []
    for ext_path in extensions:
        if not isinstance(ext_path, str):
            continue
        # ext_path looks like "/extensions/ComfyUI-Manager/foo.js" or sometimes
        # without a leading slash. Normalize.
        clean = ext_path.lstrip("/")
        if not clean.startswith("extensions/"):
            log.warning("skip non-extensions path: %s", ext_path)
            continue
        try:
            r = _http.request("GET", f"{COMFY_BASE}/{clean}")
            if r.status != 200:
                log.warning("fetch %s -> %d", clean, r.status)
                continue
            content_type = r.headers.get("Content-Type", "application/javascript")
            # Strip any charset param Boto3 doesn't love
            content_type = content_type.split(";")[0].strip() or "application/javascript"
            _s3.put_object(
                Bucket=FRONTEND_BUCKET,
                Key=clean,
                Body=r.data,
                ContentType=content_type,
                CacheControl="public, max-age=300",
            )
            uploaded.append(f"/{clean}")
        except Exception:  # noqa: BLE001
            log.exception("upload %s failed", clean)
            continue

    log.info("uploaded %d/%d extension files to s3://%s/extensions/",
             len(uploaded), len(extensions), FRONTEND_BUCKET)

    # Step 4: POST the list to dispatcher
    try:
        body = json.dumps({"fleet": fleet, "extensions": uploaded}).encode()
        r = _http.request(
            "POST",
            f"{DISPATCHER_API_URL}/internal/extensions",
            body=body,
            headers={"Content-Type": "application/json", "x-api-key": _get_api_key()},
        )
        if r.status >= 400:
            log.error("dispatcher /internal/extensions returned %d: %s",
                      r.status, r.data[:200])
            return False
        log.info("published %d extensions to dispatcher", len(uploaded))
        return True
    except Exception:  # noqa: BLE001
        log.exception("dispatcher publish failed")
        return False
