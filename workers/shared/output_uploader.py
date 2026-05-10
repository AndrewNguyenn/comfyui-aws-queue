"""
Scan ComfyUI output directory after job completion, upload everything new
to S3 outputs bucket, return list of S3 keys.

ComfyUI custom nodes (especially WanVideoWrapper) sometimes write output
files directly with hardcoded paths instead of using ComfyUI's standard
output mechanism. By scanning the output dir for anything modified after
job start, we catch all of them. (v3 N11)
"""

from __future__ import annotations

import logging
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

log = logging.getLogger(__name__)

# File extensions we consider as "outputs" — defensively narrow.
OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mkv"}


class OutputUploader:
    def __init__(self, bucket: str, region: str, output_dir: Path = Path("/opt/comfy/output")):
        self.bucket = bucket
        self.output_dir = output_dir
        self._s3 = boto3.client("s3", region_name=region)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_marker(self) -> float:
        """Capture pre-job timestamp. Use as `since` for upload_new_outputs."""
        return time.time() - 1  # tiny offset to be inclusive

    def upload_new_outputs(self, job_id: str, since: float) -> list[str]:
        """Find all output files modified after `since`, upload to S3, return keys."""
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        keys: list[str] = []

        for path in self.output_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in OUTPUT_EXTS:
                continue
            try:
                if path.stat().st_mtime < since:
                    continue
            except FileNotFoundError:
                continue

            key = f"outputs/{date_prefix}/{job_id}/{path.relative_to(self.output_dir)}"
            content_type, _ = mimetypes.guess_type(path.name)
            log.info("uploading %s -> s3://%s/%s", path, self.bucket, key)
            try:
                self._s3.upload_file(
                    str(path),
                    self.bucket,
                    key,
                    ExtraArgs={"ContentType": content_type or "application/octet-stream"},
                )
                keys.append(key)
            except Exception:  # noqa: BLE001
                log.exception("failed to upload %s", path)

        return keys

    def cleanup_outputs(self, since: float) -> None:
        """Remove output files modified after `since` to free disk for next job."""
        for path in self.output_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < since:
                    continue
            except FileNotFoundError:
                continue
            try:
                path.unlink()
            except Exception:  # noqa: BLE001
                log.warning("could not unlink %s", path)
