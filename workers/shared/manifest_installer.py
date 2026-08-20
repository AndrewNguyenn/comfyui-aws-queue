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
# Lazy S3 client — so this module can be imported at Docker *build* time
# (bake_nodes.py) where there are no AWS creds / region.
_s3 = None


def _get_s3():
    global _s3
    if _s3 is None:
        region = (os.environ.get("AWS_REGION")
                  or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
        _s3 = boto3.client("s3", region_name=region)
    return _s3


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
        r = _get_s3().get_object(Bucket=MANIFEST_BUCKET, Key=MANIFEST_KEY)
        return json.loads(r["Body"].read().decode())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return {"version": 1, "nodes": []}
        log.exception("failed to read manifest")
        return {"version": 1, "nodes": []}


# The metadata container (workers/metadata/Dockerfile) sets COMFY_METADATA=1.
# It runs ComfyUI on a vanilla python:3.12-slim base with a real CPU-built
# torchaudio — NOT the NGC GPU image. The torchaudio re-stub only exists to
# work around NGC shipping no torchaudio; on the metadata image the real CPU
# torchaudio is correct and stubbing it breaks audio nodes. Both the runtime
# sync() below and the build-time bake_nodes.py gate the re-stub on this.
IS_METADATA = os.environ.get("COMFY_METADATA") == "1"

# The video container (workers/video/Dockerfile) sets COMFY_KEEP_TORCHAUDIO=1.
# It deliberately pins a REAL torchaudio 2.5.x matching the NGC torch, because
# ComfyUI's native MiniMax H3 nodes do `import torchaudio` at module load and a
# stub makes every one of them vanish. That pin is the correct state for this
# image, so the repair below must leave it alone.
#
# This is NOT the image fleet's situation: there the only torchaudio available
# is a 2.11 wheel needing a CUDA the base lacks, so stubbing is the fix. The
# distinction is "is there a compatible real wheel installed", not "is a real
# wheel present" — an earlier version of this guard assumed the latter and had
# to be reverted.
KEEP_TORCHAUDIO = IS_METADATA or os.environ.get("COMFY_KEEP_TORCHAUDIO") == "1"


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

    installed_any = False
    for entry in nodes:
        name = entry.get("name") or _name_from_url(entry.get("url", ""))
        url = entry.get("url")
        commit = entry.get("commit")
        if not name or not url:
            log.warning("skip malformed manifest entry: %r", entry)
            continue
        try:
            results[name] = _install_one(name, url, commit)
            if results[name] == "installed":
                installed_any = True
        except Exception as e:  # noqa: BLE001
            log.exception("install %s crashed", name)
            results[name] = f"failed: {e!r}"

    # Only re-repair the foundation if a net-new pack was actually installed
    # — an all-baked boot already has a clean image, no need to churn.
    if installed_any:
        _force_clean_opencv()
        _force_transformers()
        # Image-fleet-only — the metadata and video images both carry a real
        # torchaudio that re-stubbing here (on every boot a net-new pack is
        # installed) would destroy. See KEEP_TORCHAUDIO above.
        if not KEEP_TORCHAUDIO:
            _restore_torchaudio_stub()
        _patch_image_metadata_extension()
    return results


_OPENCV_PKGS = (
    "opencv-python", "opencv-python-headless",
    "opencv-contrib-python", "opencv-contrib-python-headless",
)
_OPENCV_PIN = "opencv-python-headless==4.11.0.86"


def _force_clean_opencv() -> None:
    """Make `import cv2` reliably work. Two failure modes, both fixed here:

    1. Custom-node installs leave MULTIPLE opencv variants (-python /
       -headless / -contrib) clobbering one cv2/ dir → mismatched stub vs
       binary. Fix: uninstall every variant, install exactly one.
    2. Several opencv-python releases ship a cv2/typing/__init__.py that does
       `LayerId = cv2.dnn.DictValue`, but the binary lacks DictValue →
       AttributeError on import → the whole ComfyUI process crashes on
       startup. Fix: patch that file to tolerate the missing attribute."""
    log.info("forcing clean opencv → %s", _OPENCV_PIN)
    try:
        _run(["pip", "uninstall", "-y", *_OPENCV_PKGS])
    except subprocess.CalledProcessError:
        pass
    # Under the constraint file, same as a node install. Without it pip is free
    # to satisfy opencv's own numpy requirement by pulling numpy 2.x, which
    # breaks the C bridge of everything compiled against NumPy 1.x — torch's
    # included ("RuntimeError: Numpy is not available"). This repair runs LAST,
    # so an unconstrained upgrade here silently undoes the pin every other
    # install respected.
    cf = _constraints_file()
    cmd = ["pip", "install", "--no-input", _OPENCV_PIN]
    if cf:
        cmd += ["--constraint", cf]
    try:
        _run(cmd)
    except subprocess.CalledProcessError as e:
        log.error("opencv reinstall failed (rc=%d)", e.returncode)
    _patch_cv2_typing()


def _restore_torchaudio_stub() -> None:
    """ComfyUI v0.20.1 hard-imports torchaudio at startup; NGC ships none and
    a real torchaudio wheel needs a CUDA the image lacks (libcudart.so.13) →
    `import torchaudio` raises OSError → ComfyUI crashes on every boot. The
    Dockerfile installs a no-op stub; custom-node installs clobber it with a
    real torchaudio. Re-stub after installs."""
    import sysconfig
    try:
        _run(["pip", "uninstall", "-y", "torchaudio"])
    except subprocess.CalledProcessError:
        pass
    base = sysconfig.get_path("platlib")
    if not base:
        return
    d = Path(base) / "torchaudio"
    try:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text(
            '"""torchaudio stub — NGC ships no torchaudio; audio workflows are '
            'unused on the image fleet."""\n__version__ = "0.0.0-stub"\n')
        log.info("restored torchaudio stub at %s", d)
    except Exception as e:  # noqa: BLE001
        log.error("torchaudio stub restore failed: %r", e)


_TRANSFORMERS_PIN = "transformers==4.46.3"


def _force_transformers() -> None:
    """transformers >=4.54 ships generation/continuous_batching/distributed.py
    which does `from torch.distributed.tensor.device_mesh import DeviceMesh` —
    a path absent from NGC's torch 2.5. transformers' lazy import machinery
    then fails for AutoModel / BlipProcessor / etc., so every node that
    touches transformers (Easy-Use, WAS, LayerStyle, Florence2…) IMPORT
    FAILS. Node installs keep upgrading transformers; force it back to a
    torch-2.5-era release as the last step, after everything else."""
    log.info("forcing transformers → %s", _TRANSFORMERS_PIN)
    cf = _constraints_file()
    cmd = ["pip", "install", "--no-input", _TRANSFORMERS_PIN]
    if cf:
        cmd += ["--constraint", cf]  # see _force_clean_opencv: unconstrained, this can pull numpy 2.x
    try:
        _run(cmd)
    except subprocess.CalledProcessError as e:
        log.error("transformers pin failed (rc=%d)", e.returncode)


def _patch_cv2_typing() -> None:
    """Neutralise the `cv2.dnn.DictValue` reference in cv2/typing/__init__.py
    so `import cv2` can't crash on it (version-independent)."""
    import sysconfig
    seen = set()
    for key in ("platlib", "purelib"):
        base = sysconfig.get_path(key)
        if not base or base in seen:
            continue
        seen.add(base)
        f = Path(base) / "cv2" / "typing" / "__init__.py"
        try:
            if not f.is_file():
                continue
            txt = f.read_text()
            if "cv2.dnn.DictValue" in txt:
                f.write_text(txt.replace("cv2.dnn.DictValue",
                                         'getattr(cv2.dnn, "DictValue", int)'))
                log.info("patched cv2/typing DictValue reference at %s", f)
        except Exception as e:  # noqa: BLE001
            log.error("cv2 typing patch failed: %r", e)


def _patch_image_metadata_extension() -> None:
    """SaveImageWithMetaData (comfyui_image_metadata_extension) builds its PNG
    metadata by introspecting ComfyUI's execution cache. This ComfyUI version
    made that cache async (`HierarchicalCache.get` is a coroutine), which the
    extension's capture path doesn't await → AttributeError → the exception
    propagates out of save_images and the IMAGE NEVER SAVES.

    Make metadata capture non-fatal: wrap the gen_pnginfo() call so a capture
    failure logs and falls back to empty metadata — the image still saves."""
    f = (CUSTOM_NODES_DIR / "comfyui_image_metadata_extension"
         / "modules" / "nodes" / "node.py")
    old = "pnginfo_dict = pnginfo_dict or self.gen_pnginfo(prompt, prefer_nearest)"
    new = (
        "try:\n"
        "                pnginfo_dict = pnginfo_dict or self.gen_pnginfo(prompt, prefer_nearest)\n"
        "            except Exception as _meta_e:\n"
        "                print(f\"[image_metadata_extension] metadata capture failed:"
        " {_meta_e!r}; saving image without captured metadata\")\n"
        "                pnginfo_dict = pnginfo_dict or {}"
    )
    try:
        if not f.is_file():
            return
        txt = f.read_text()
        if old in txt and "metadata capture failed" not in txt:
            f.write_text(txt.replace(old, new, 1))
            log.info("patched image_metadata_extension: metadata capture is now non-fatal")
    except Exception as e:  # noqa: BLE001
        log.error("image_metadata_extension patch failed: %r", e)


def _install_one(name: str, url: str, commit: Optional[str]) -> str:
    target = CUSTOM_NODES_DIR / name
    # Baked into the worker image at build time — already cloned + deps
    # installed. Skip entirely (no clone, no re-pip) so cold starts are fast
    # and we don't re-run installs that could re-introduce conflicts.
    if (target / ".baked").is_file():
        return "baked"
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


def _constraints_file() -> Optional[str]:
    """Pin the foundational packages at their image-shipped versions so a
    custom node's requirements.txt cannot upgrade/downgrade them. A node that
    pulled a different numpy used to break the WHOLE environment ('numpy.core
    .multiarray failed to import' → every torch/numpy node IMPORT FAILED);
    with a constraint that node simply doesn't get its way instead of taking
    everything down."""
    cache = getattr(_constraints_file, "_cache", "unset")
    if cache != "unset":
        return cache  # type: ignore[return-value]
    import importlib.metadata as md
    lines = []
    # numpy/torch: a wrong version breaks every tensor node. opencv: a wrong
    # version raises 'cv2.dnn has no attribute DictValue' which crashes the
    # ENTIRE ComfyUI process on startup, not just one node.
    # transformers/tokenizers: a too-new transformers needs torch features
    # NGC's torch 2.5 lacks (torch.distributed.tensor.device_mesh) → its lazy
    # imports (AutoModel, BlipProcessor) fail → every transformers-using node
    # IMPORT FAILS. Pin it so a node can't upgrade it.
    for pkg in ("numpy", "torch", "torchvision", "torchaudio",
                "transformers", "tokenizers",
                "opencv-python", "opencv-python-headless",
                "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            lines.append(f"{pkg}=={md.version(pkg)}")
        except Exception:  # noqa: BLE001
            pass
    path: Optional[str] = None
    if lines:
        path = "/tmp/comfy-pip-constraints.txt"
        Path(path).write_text("\n".join(lines) + "\n")
        log.info("pip constraints: %s", ", ".join(lines))
    _constraints_file._cache = path  # type: ignore[attr-defined]
    return path


def _pip_install_requirements(target: Path) -> None:
    req = target / "requirements.txt"
    if not req.is_file():
        return
    log.info("pip install -r %s", req)
    base = ["pip", "install", "--no-input"]
    constraints = _constraints_file()
    if constraints:
        base += ["--constraint", constraints]
    try:
        _run(base + ["-r", str(req)])
        return
    except subprocess.CalledProcessError:
        log.warning("pip -r failed for %s; retrying line-by-line", target.name)
    # Fallback: install each requirement on its own. `pip install -r` is
    # all-or-nothing — one line that conflicts with the constraint (or is
    # otherwise unsatisfiable) drops EVERY dep, which is how impact-pack lost
    # segment-anything. Line-by-line lets the satisfiable deps through.
    for raw in req.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            _run(base + [line])
        except subprocess.CalledProcessError:
            log.error("  skipped unsatisfiable requirement: %s (%s)", line, target.name)


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
