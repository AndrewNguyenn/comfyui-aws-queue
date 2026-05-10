"""
ComfyUI subprocess supervisor.

Starts ComfyUI as a child process listening on 127.0.0.1:8188. Provides
graceful shutdown, restart on crash (up to N attempts), and a wait-for-ready
helper that polls /system_stats until it responds.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from typing import Optional

import urllib3

log = logging.getLogger(__name__)
http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=2.0, read=2.0))


class ComfySupervisor:
    def __init__(
        self,
        comfy_dir: str = "/opt/comfy",
        port: int = 8188,
        extra_args: tuple[str, ...] = (),
        max_restart_attempts: int = 3,
    ):
        self.comfy_dir = comfy_dir
        self.port = port
        self.extra_args = extra_args
        self.max_restart_attempts = max_restart_attempts
        self._proc: Optional[subprocess.Popen] = None
        self._restart_count = 0
        self._stopping = False
        self._watch_thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        cmd = [
            "python", "main.py",
            "--listen", "127.0.0.1",
            "--port", str(self.port),
            *self.extra_args,
        ]
        log.info("launching comfyui: %s (cwd=%s)", " ".join(cmd), self.comfy_dir)
        self._proc = subprocess.Popen(
            cmd,
            cwd=self.comfy_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._watch_thread = threading.Thread(target=self._watch, daemon=True, name="comfy-watch")
        self._watch_thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stopping = True
        if not self._proc:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning("comfy did not exit in %ss; sending SIGKILL", timeout)
                self._proc.kill()
                self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            log.exception("error stopping comfy")

    def wait_for_ready(self, timeout_seconds: int = 180) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                r = http.request("GET", f"{self.base_url}/system_stats")
                if r.status == 200:
                    log.info("comfyui ready (took %.1fs)", timeout_seconds - (deadline - time.monotonic()))
                    return
            except urllib3.exceptions.HTTPError:
                pass
            time.sleep(2)
        raise RuntimeError(f"comfyui did not become ready within {timeout_seconds}s")

    def _watch(self) -> None:
        """Restart on unexpected exit, up to max_restart_attempts."""
        while not self._stopping and self._proc:
            self._proc.wait()
            if self._stopping:
                return
            log.error(
                "comfyui exited unexpectedly (rc=%s, attempt %d/%d)",
                self._proc.returncode,
                self._restart_count + 1,
                self.max_restart_attempts,
            )
            self._restart_count += 1
            if self._restart_count > self.max_restart_attempts:
                log.error("max restart attempts exceeded; instance will be marked unhealthy")
                # Re-raise SIGTERM to ourselves so the worker exits and ECS replaces.
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(2)
            self.start()
