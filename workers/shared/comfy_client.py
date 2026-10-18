"""
HTTP client for the local ComfyUI subprocess.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import urllib3

log = logging.getLogger(__name__)
_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5.0, read=60.0))


class JobCancelled(Exception):
    """Raised by wait_for_completion when the job was cancelled mid-run.

    The worker catches this to mark the job "cancelled" rather than "failed".
    """


def _summarize_node_errors(node_errors: dict) -> str:
    parts = []
    for nid, err in node_errors.items():
        msgs = "; ".join(
            f"{e.get('message', '')}: {e.get('details', '')}" for e in err.get("errors", [])
        )
        parts.append(f"{err.get('class_type', '?')} {nid}: {msgs}")
    return " | ".join(parts)[:500]


def _execution_error(status: dict) -> str:
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            return f"{data.get('node_type', '?')} {data.get('node_id', '?')}: {data.get('exception_message', '').strip()}"
    return "unknown error"


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url.rstrip("/")

    def submit_prompt(self, workflow: dict, client_id: str = "worker") -> str:
        # Drop frontend-only nodes — Notes, rgthree "Fast Groups Bypasser"
        # ("Enable / Disable groups"), labels. They carry no class_type and
        # never execute, but ComfyUI's prompt validator rejects the WHOLE
        # prompt over a single one ("Node ... has no class_type"). A healthy
        # editor strips them client-side; do it here too so a workflow saved
        # while a node pack's JS wasn't loaded still runs.
        if isinstance(workflow, dict):
            workflow = {
                nid: node for nid, node in workflow.items()
                if isinstance(node, dict) and node.get("class_type")
            }
        body = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        r = _http.request(
            "POST",
            f"{self.base_url}/prompt",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        if r.status >= 400:
            raise RuntimeError(f"comfy /prompt {r.status}: {r.data[:300]!r}")
        resp = json.loads(r.data.decode())
        # ComfyUI accepts a prompt when ANY output validates: the outputs that
        # failed come back in node_errors ("Output will be ignored") and the
        # rest run. A bad detector name on a face-detailer branch then
        # "completes" with no outputs and no error, so fail it here instead.
        if resp.get("node_errors"):
            raise RuntimeError(f"comfy /prompt node_errors: {_summarize_node_errors(resp['node_errors'])}")
        return resp["prompt_id"]

    def wait_for_completion(self, prompt_id: str, timeout_seconds: int = 1800,
                            poll_interval: float = 2.0,
                            cancel_event: Any = None) -> dict:
        """Poll /history until prompt completes. Returns the history entry.

        If `cancel_event` (a threading.Event) is set while waiting, raises
        JobCancelled — the caller is expected to have already POSTed
        /interrupt, so an interrupted prompt never reports `completed` and
        would otherwise hang here until the timeout.
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled(f"comfy job {prompt_id} cancelled")
            r = _http.request("GET", f"{self.base_url}/history/{prompt_id}")
            if r.status == 200:
                hist = json.loads(r.data.decode())
                entry = hist.get(prompt_id)
                status = (entry or {}).get("status", {})
                # A prompt that raised mid-run reports status_str "error" and,
                # on current ComfyUI, completed=False, so it would otherwise
                # wait out the full timeout.
                if status.get("status_str") == "error":
                    raise RuntimeError(f"comfy job {prompt_id} failed: {_execution_error(status)}")
                if status.get("completed"):
                    return entry
            time.sleep(poll_interval)
        raise TimeoutError(f"comfy job {prompt_id} did not complete in {timeout_seconds}s")

    def fetch_object_info(self) -> dict:
        r = _http.request("GET", f"{self.base_url}/object_info")
        if r.status >= 400:
            raise RuntimeError(f"comfy /object_info {r.status}")
        return json.loads(r.data.decode())

    def interrupt(self) -> None:
        # Short explicit timeout: the cancel-watch thread calls this and the
        # worker join()s that thread with a 10s budget before claiming the
        # next job. A bounded POST keeps the thread from outliving the join
        # (which could otherwise interrupt the wrong prompt).
        try:
            _http.request("POST", f"{self.base_url}/interrupt",
                          timeout=urllib3.Timeout(connect=3.0, read=5.0))
        except Exception:  # noqa: BLE001
            log.exception("interrupt request failed")
