"""
S3-backed model cache for the ComfyUI model directory.

Models live in S3 (~1.4 TB catalog). On-demand they're downloaded to a local
EBS cache. LRU eviction keeps cache under a configured size budget.

Pinned models (defined in extra_models.json) are downloaded at boot in
parallel and never evicted — they cover the common case (e.g., Wan 14B Q8
+ Lightning LoRA + WanVideo VAE).

Cache files are placed under /var/cache/models/<type>/<filename>. The
ComfyUI directory's models/<type>/ contains symlinks pointing into the cache
so ComfyUI sees its expected layout.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

log = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("/var/cache/models")
COMFY_MODELS_ROOT = Path("/opt/comfy/models")


class CacheManager:
    """Thread-safe LRU cache for model files backed by S3."""

    def __init__(
        self,
        models_bucket: str,
        models_table: str,
        max_cache_gb: int,
        region: str,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        comfy_models_root: Path = COMFY_MODELS_ROOT,
    ):
        self.bucket = models_bucket
        self.table = models_table
        self.max_cache_bytes = int(max_cache_gb) * 1024 * 1024 * 1024
        self.cache_root = cache_root
        self.comfy_models_root = comfy_models_root
        self._lock = threading.Lock()
        # Per-model locks prevent two threads from concurrently downloading
        # the same model file (resolves code review N8).
        self._download_locks: dict[str, threading.Lock] = {}
        # Maps cached filename → ref count of in-use jobs
        self._in_use: dict[str, int] = {}
        self._pinned: set[str] = set()

        self._s3 = boto3.client(
            "s3",
            region_name=region,
            config=Config(
                retries={"max_attempts": 5, "mode": "adaptive"},
                max_pool_connections=20,
            ),
        )
        self._ddb = boto3.client("dynamodb", region_name=region)

        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ---------- public ----------
    def warm_pinned(self, pin_list_path: Path) -> None:
        """Download all 'pinned: true' models named in pin_list_path in parallel."""
        if not pin_list_path.exists():
            log.info("no pinned model list at %s — skipping warm-up", pin_list_path)
            return
        with pin_list_path.open() as f:
            pinned_names: list[str] = json.load(f).get("pinned", [])

        log.info("warming %d pinned models", len(pinned_names))
        with self._lock:
            self._pinned.update(pinned_names)

        # Parallel downloads — boto3 client is thread-safe.
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(self._fetch_no_evict, n): n for n in pinned_names}
            for f in as_completed(futs):
                name = futs[f]
                try:
                    f.result()
                    log.info("pinned model %s ready", name)
                except Exception:  # noqa: BLE001
                    log.exception("failed to warm pinned model %s", name)

    def ensure(self, model_name: str) -> Path:
        """Returns local path. Downloads from S3 if missing, evicting LRU.
        Per-model lock prevents concurrent downloads of the same model
        (resolves code review N8)."""
        meta = self._lookup_in_catalog(model_name)
        if not meta:
            raise FileNotFoundError(f"model {model_name} not in catalog")

        local_path = self._cache_path(meta)
        # Fast path: cache hit, no lock needed for symlink.
        if local_path.exists():
            os.utime(local_path, None)
            self._link_into_comfy(meta, local_path)
            return local_path

        # Get-or-create a per-model lock under the global lock.
        with self._lock:
            dl_lock = self._download_locks.setdefault(model_name, threading.Lock())

        # Hold the per-model lock through eviction + download.
        with dl_lock:
            # Re-check after acquiring (another thread may have downloaded it).
            if local_path.exists():
                os.utime(local_path, None)
                self._link_into_comfy(meta, local_path)
                return local_path

            size_bytes = int(meta.get("size_gb", 0) * 1e9)
            with self._lock:
                self._evict_to_fit(size_bytes, exclude=local_path.name)
            self._download(meta, local_path)
            self._link_into_comfy(meta, local_path)
            return local_path

    @contextmanager
    def use(self, model_name: str):
        """Context manager that ref-counts a model so it can't be evicted while in use."""
        path = self.ensure(model_name)
        with self._lock:
            self._in_use[path.name] = self._in_use.get(path.name, 0) + 1
        try:
            yield path
        finally:
            with self._lock:
                self._in_use[path.name] = max(0, self._in_use.get(path.name, 0) - 1)

    # ---------- internals ----------
    def _fetch_no_evict(self, model_name: str) -> Path:
        """Download a pinned model. Skip eviction since pinned takes priority."""
        meta = self._lookup_in_catalog(model_name)
        if not meta:
            raise FileNotFoundError(f"pinned model {model_name} not in catalog")
        local_path = self._cache_path(meta)
        if local_path.exists():
            self._link_into_comfy(meta, local_path)
            return local_path
        self._download(meta, local_path)
        self._link_into_comfy(meta, local_path)
        return local_path

    def _lookup_in_catalog(self, model_name: str) -> dict[str, Any] | None:
        try:
            r = self._ddb.get_item(TableName=self.table, Key={"name": {"S": model_name}})
        except Exception:  # noqa: BLE001
            log.exception("ddb lookup failed for %s", model_name)
            return None
        item = r.get("Item")
        if not item:
            return None
        return {
            "name": item["name"]["S"],
            "type": item["type"]["S"],
            "s3_key": item["s3_key"]["S"],
            "size_gb": float(item.get("size_gb", {"N": "0"})["N"]),
        }

    def _cache_path(self, meta: dict) -> Path:
        type_dir = self.cache_root / meta["type"]
        type_dir.mkdir(parents=True, exist_ok=True)
        # Use the basename of s3_key so multiple models with same display name don't collide.
        filename = Path(meta["s3_key"]).name
        return type_dir / filename

    def _link_into_comfy(self, meta: dict, local_path: Path) -> None:
        """Create symlink at comfy/models/<type>/<filename> → local_path.
        ComfyUI uses its own type dirs (checkpoints/, loras/, vae/, etc.)."""
        comfy_type_dir = self.comfy_models_root / _comfy_type_dir(meta["type"])
        comfy_type_dir.mkdir(parents=True, exist_ok=True)
        link = comfy_type_dir / local_path.name
        if link.is_symlink() or link.exists():
            return
        try:
            link.symlink_to(local_path)
        except FileExistsError:
            pass

    def _download(self, meta: dict, local_path: Path) -> None:
        """Atomic download: write to .part, rename when complete."""
        log.info("downloading model %s (%.2f GB) from s3://%s/%s",
                 meta["name"], meta["size_gb"], self.bucket, meta["s3_key"])
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        try:
            self._s3.download_file(self.bucket, meta["s3_key"], str(tmp))
            tmp.rename(local_path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            raise

    def _evict_to_fit(self, bytes_needed: int, exclude: str) -> None:
        """LRU evict non-pinned, non-in-use files until enough bytes are free."""
        used = self._used_bytes()
        free = self.max_cache_bytes - used
        if free >= bytes_needed:
            return

        # Collect eviction candidates.
        candidates: list[tuple[float, Path]] = []
        for type_dir in self.cache_root.iterdir():
            if not type_dir.is_dir():
                continue
            for f in type_dir.iterdir():
                if f.name == exclude:
                    continue
                if f.name in self._pinned or self._in_use.get(f.name, 0) > 0:
                    continue
                candidates.append((f.stat().st_mtime, f))
        candidates.sort()  # oldest first

        for _mtime, f in candidates:
            if free >= bytes_needed:
                break
            try:
                size = f.stat().st_size
                f.unlink()
                # Also clean the symlink in comfy models dir.
                self._unlink_from_comfy(f)
                free += size
                log.info("evicted %s (freed %d bytes)", f.name, size)
            except Exception:  # noqa: BLE001
                log.exception("eviction of %s failed", f)

        if free < bytes_needed:
            log.warning(
                "could not free %d bytes (only %d available, mostly pinned/in-use)",
                bytes_needed,
                free,
            )

    def _unlink_from_comfy(self, cache_file: Path) -> None:
        for type_dir in self.comfy_models_root.iterdir():
            if not type_dir.is_dir():
                continue
            link = type_dir / cache_file.name
            if link.is_symlink() and link.resolve() == cache_file:
                try:
                    link.unlink()
                except Exception:  # noqa: BLE001
                    pass

    def _used_bytes(self) -> int:
        total = 0
        for type_dir in self.cache_root.iterdir():
            if not type_dir.is_dir():
                continue
            for f in type_dir.iterdir():
                if f.is_file():
                    total += f.stat().st_size
        return total


def _comfy_type_dir(catalog_type: str) -> str:
    """Map our catalog 'type' to ComfyUI's models/ subdirectory name."""
    return {
        "checkpoint": "checkpoints",
        "lora": "loras",
        "vae": "vae",
        "controlnet": "controlnet",
        "clip": "clip",
        "embedding": "embeddings",
        "upscale": "upscale_models",
    }.get(catalog_type, catalog_type)
