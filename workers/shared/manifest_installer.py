"""
Custom-node manifest installer.

The manifest at s3://<outputs>/manifests/custom-nodes.json is the single source
of truth for which custom nodes belong on this deployment. Both the metadata
container and every GPU worker run this on boot to sync /opt/comfy/custom_nodes/
against the manifest:

  1. For each entry: ensure git checkout exists at the pinned commit (or HEAD)
  2. If requirements.txt is present, pip install it (best-effort)
  3. Log result to install_log table for visibility

The dispatcher Lambda is what appends entries to this manifest when the user
clicks Install in Manager's UI (see services/dispatcher/handler.py).

Schema:
    {
      "version": 1,
      "updated_at": "2026-05-17T...",
      "nodes": [
        {
          "name": "ComfyUI-Custom-Scripts",
          "url": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git",
          "commit": null,
          "added_at": "2026-05-17T...",
          "source": "manager-ui"
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)
_s3 = boto3.client("s3")

# Same bucket the dispatcher writes object_info to — already in the worker IAM
# grant set. Keeps deploy surface small.
MANIFEST_BUCKET = os.environ.get("OUTPUTS_BUCKET", "")
MANIFEST_KEY = "manifests/custom-nodes.json"
CUSTOM_NODES_DIR = Path(os.environ.get("CUSTOM_NODES_DIR", "/opt/comfy/custom_nodes"))


def load_manifest() -> dict:
    """Return the manifest dict, or an empty default if it doesn't exist yet."""
    if not MANIFEST_BUCKET:
        log.warning("OUTPUTS_BUCKET unset; manifest sync disabled")
        return {"version": 1, "nodes": []}
    try:
        r = _s3.get_object(Bucket=MANIFEST_BUCKET, Key=MANIFEST_KEY)
        return json.loads(r["Body"].read().decode())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return {"version": 1, "nodes": []}
        log.exception("failed to read manifest")
        return {"version": 1, "nodes": []}


def sync() -> dict[str, str]:
    """Sync /opt/comfy/custom_nodes against the manifest.

    Returns a dict of {node_name: "installed" | "skipped" | "failed: <reason>"}.
    Best-effort: a failing node logs the error and moves on rather than killing
    the worker boot.
    """
    results: dict[str, str] = {}
    manifest = load_manifest()
    nodes = manifest.get("nodes") or []
    if not nodes:
        log.info("manifest empty; nothing to install")
        return results

    CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
    log.info("syncing %d custom nodes from manifest", len(nodes))

    for entry in nodes:
        name = entry.get("name") or _name_from_url(entry.get("url", ""))
        url = entry.get("url")
        commit = entry.get("commit")
        if not name or not url:
            log.warning("skip malformed manifest entry: %r", entry)
            continue
        try:
            results[name] = _install_one(name, url, commit)
        except Exception as e:  # noqa: BLE001
            log.exception("install %s crashed", name)
            results[name] = f"failed: {e!r}"
    return results


def _install_one(name: str, url: str, commit: Optional[str]) -> str:
    target = CUSTOM_NODES_DIR / name
    if target.exists():
        if commit:
            _run(["git", "fetch", "--depth=1", "origin", commit], cwd=target)
            _run(["git", "checkout", commit], cwd=target)
            _pip_install_requirements(target)
            return "installed"
        log.info("custom node %s already present; skipping clone", name)
        _pip_install_requirements(target)
        return "skipped"

    log.info("cloning %s from %s", name, url)
    clone_cmd = ["git", "clone", "--depth=1"]
    if commit:
        clone_cmd += ["--branch", commit] if _looks_like_ref(commit) else []
    clone_cmd += [url, str(target)]
    _run(clone_cmd)
    if commit and not _looks_like_ref(commit):
        _run(["git", "fetch", "--depth=1", "origin", commit], cwd=target)
        _run(["git", "checkout", commit], cwd=target)
    _pip_install_requirements(target)
    return "installed"


def _pip_install_requirements(target: Path) -> None:
    req = target / "requirements.txt"
    if not req.is_file():
        return
    log.info("pip install -r %s", req)
    try:
        _run(["pip", "install", "--no-input", "-r", str(req)])
    except subprocess.CalledProcessError as e:
        log.error("pip install failed for %s (rc=%d). Continuing.", target.name, e.returncode)


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    log.debug("run: %s (cwd=%s)", " ".join(cmd), cwd)
    subprocess.run(cmd, cwd=cwd, check=True)


def _name_from_url(url: str) -> str:
    last = url.rstrip("/").rsplit("/", 1)[-1]
    return last[:-4] if last.endswith(".git") else last


def _looks_like_ref(s: str) -> bool:
    # Branch/tag names are short; SHAs are 7-40 hex chars.
    return not all(c in "0123456789abcdef" for c in s.lower()) or len(s) < 7


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    summary = sync()
    for n, st in summary.items():
        print(f"  {n}: {st}")
