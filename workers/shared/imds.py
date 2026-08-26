"""IMDSv2, once. The token handshake used to be copied into worker.py,
spot_handler.py and scale_in_protection.py, each with its own pool, timeouts
and refresh rules (only one of them ever refreshed). One token, refreshed on
age or on a 401, shared by every caller. Never raises: IMDS is best-effort for
every use we have (instance type, instance id, spot notice, ASG lifecycle)."""

from __future__ import annotations

import logging
import threading
import time

import urllib3

log = logging.getLogger(__name__)

BASE = "http://169.254.169.254/latest"
TOKEN_TTL_SECONDS = 21600
_REFRESH_AFTER_SECONDS = 3600

_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=1.0, read=2.0))
_lock = threading.Lock()
_token = ""
_token_at = 0.0


def _refresh_locked() -> str:
    global _token, _token_at
    try:
        r = _http.request(
            "PUT", f"{BASE}/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": str(TOKEN_TTL_SECONDS)},
        )
        if r.status == 200:
            _token = r.data.decode().strip()
            _token_at = time.monotonic()
    except Exception:  # noqa: BLE001
        log.warning("imds token refresh failed", exc_info=True)
    return _token


def token() -> str:
    with _lock:
        if not _token or time.monotonic() - _token_at > _REFRESH_AFTER_SECONDS:
            return _refresh_locked()
        return _token


def get(path: str) -> tuple[int, str]:
    """GET <BASE>/<path>. Returns (status, body); (0, '') when IMDS is unreachable.
    One retry with a fresh token on 401."""
    tok = token()
    if not tok:
        return 0, ""
    url = f"{BASE}/{path.lstrip('/')}"
    try:
        r = _http.request("GET", url, headers={"X-aws-ec2-metadata-token": tok})
        if r.status == 401:
            with _lock:
                tok = _refresh_locked()
            r = _http.request("GET", url, headers={"X-aws-ec2-metadata-token": tok})
        return r.status, r.data.decode(errors="replace").strip()
    except Exception:  # noqa: BLE001
        log.debug("imds get %s failed", path, exc_info=True)
        return 0, ""


def text(path: str) -> str:
    """Body of a 200 response, else ''."""
    status, body = get(path)
    return body if status == 200 else ""
