"""
ComfyUI → API Gateway WebSocket bridge.

For each job the worker processes, this opens a WebSocket to the local
ComfyUI server (`ws://127.0.0.1:8188/ws?clientId=<job-client-id>`),
listens for status/executing/progress/executed events scoped to that
client_id, and forwards each one as an HTTP POST to our REST endpoint
`/internal/ws-event`.

The forward Lambda then looks up the live WS connection for that
client_id and pushes the event through to the browser.

Skipped in v1: binary preview frames. ComfyUI sends those as binary
WS messages on the same socket; forwarding them would require the WS
push side to support binary frames, plus base64 encoding through HTTP.
Easy follow-up if we want it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

import urllib3
import websocket  # websocket-client package

log = logging.getLogger(__name__)
_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5.0, read=10.0))

DISPATCHER_API_URL = os.environ.get("DISPATCHER_API_URL", "").rstrip("/")
WORKER_API_KEY_ID = os.environ.get("WORKER_API_KEY_ID", "")
COMFY_WS_HOST = os.environ.get("COMFY_WS_HOST", "127.0.0.1:8188")

# Lazy cache: actual API key value (looked up from API GW once on first use).
_api_key_cache: Optional[str] = None


def _get_api_key() -> str:
    global _api_key_cache
    if _api_key_cache or not WORKER_API_KEY_ID:
        return _api_key_cache or ""
    try:
        import boto3
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        client = boto3.client("apigateway", region_name=region)
        r = client.get_api_key(apiKey=WORKER_API_KEY_ID, includeValue=True)
        _api_key_cache = r.get("value", "")
    except Exception:  # noqa: BLE001
        log.exception("failed to fetch worker API key value")
    return _api_key_cache or ""


class WsBridge:
    """Listens to one ComfyUI WS connection and forwards events to our REST.

    Lifecycle:
      bridge = WsBridge(client_id="abc123")
      bridge.start()           # spawns a background thread
      ... ComfyUI runs the prompt ...
      bridge.stop()            # called when the job finishes (or errors)

    Each instance handles ONE job/client_id. The worker should create a new
    bridge per job (the cost is just a Python thread + WS connection).
    """

    def __init__(self, client_id: str):
        self.client_id = client_id
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not DISPATCHER_API_URL:
            log.warning("DISPATCHER_API_URL unset; ws bridge inactive")
            return
        url = f"ws://{COMFY_WS_HOST}/ws?clientId={self.client_id}"
        log.info("ws bridge connecting to %s", url)
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._run_forever,
            daemon=True,
            name=f"ws-bridge-{self.client_id[:8]}",
        )
        self._thread.start()

    def _run_forever(self) -> None:
        try:
            self._ws.run_forever(
                ping_interval=30,
                ping_timeout=10,
                reconnect=2,  # auto-reconnect with 2s delay if dropped
            )
        except Exception:  # noqa: BLE001
            log.exception("ws bridge run_forever crashed for %s", self.client_id)

    def _on_message(self, _ws, raw_message) -> None:
        # ComfyUI sends JSON for status events and binary for preview frames.
        # We forward only JSON in v1 (binary previews would need base64 +
        # bigger Lambda payloads; defer).
        if isinstance(raw_message, (bytes, bytearray)):
            return
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            return
        # Forward to /internal/ws-event with our client_id wrapping
        body = json.dumps({"client_id": self.client_id, **msg}).encode("utf-8")
        try:
            r = _http.request(
                "POST",
                f"{DISPATCHER_API_URL}/internal/ws-event",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": _get_api_key(),
                },
            )
            if r.status >= 400:
                # Don't log every failed forward — the browser may have
                # disconnected. Log once per few errors at INFO.
                if msg.get("type") not in ("crystools.monitor", "status"):
                    log.debug("ws forward %s for type=%s", r.status, msg.get("type"))
        except Exception:  # noqa: BLE001
            log.debug("ws forward failed (network)", exc_info=True)

    def _on_error(self, _ws, error) -> None:
        log.warning("ws bridge error for %s: %s", self.client_id, error)

    def _on_close(self, _ws, code, reason) -> None:
        log.info("ws bridge closed for %s: %s %s", self.client_id, code, reason)

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
