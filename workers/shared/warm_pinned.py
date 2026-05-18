"""Background pre-warmer for pinned models.

Run as `python -m warm_pinned` from the worker entrypoint after mount-s3 has
mounted the model dirs. Reads pinned=true rows from the DDB catalog and
streams each file through the mountpoint with a chunked read, which forces
mount-s3 to fetch it from S3 and populate its NVMe cache. By the time the
first job needing a pinned model lands, the file is hot.

Pin policy lives in the DDB models table as a boolean attribute on each row
(`pinned`). Setting/clearing pins is done by the editor's model browser or
manually via the catalog Lambda.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import boto3

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s warm: %(message)s")

from model_types import comfy_dir

REGION = os.environ.get("AWS_REGION", "us-west-2")
MODELS_TABLE = os.environ.get("MODELS_TABLE", "")
COMFY_MODELS = Path(os.environ.get("COMFY_MODELS", "/opt/comfy/models"))


def main() -> int:
    if not MODELS_TABLE:
        log.warning("MODELS_TABLE unset; skipping warm")
        return 0

    ddb = boto3.client("dynamodb", region_name=REGION)

    log.info("scanning DDB %s for pinned=true", MODELS_TABLE)
    # attribute_exists guard makes the intent explicit even though our two
    # write paths (services/downloader/worker.py:_add_to_catalog and
    # services/catalog/handler.py:_create_model) both initialize pinned=False.
    # Safer than relying purely on `pinned = :t` filtering against rows that
    # might lack the attribute from a future write path.
    scanned = 0
    pinned: list[tuple[str, int, str]] = []  # (type, size_bytes, path)
    paginator = ddb.get_paginator("scan")
    for page in paginator.paginate(
        TableName=MODELS_TABLE,
        FilterExpression="attribute_exists(pinned) AND pinned = :t",
        ExpressionAttributeValues={":t": {"BOOL": True}},
    ):
        for item in page.get("Items", []):
            scanned += 1
            t = item.get("type", {}).get("S", "")
            s3_key = item.get("s3_key", {}).get("S", "")
            if not t or not s3_key:
                continue
            size_gb = float(item.get("size_gb", {}).get("N", "0"))
            filename = s3_key.rsplit("/", 1)[-1]
            target = COMFY_MODELS / comfy_dir(t) / filename
            pinned.append((t, int(size_gb * 1e9), str(target)))

    if not pinned:
        log.info("scanned %d pinned rows, nothing to warm", scanned)
        return 0

    # Warm smallest first — unblocks more jobs sooner if a job lands mid-warmup.
    pinned.sort(key=lambda x: x[1])

    log.info("warming %d pinned models (smallest-first)", len(pinned))
    warmed = 0
    for t, _size, path in pinned:
        if not Path(path).exists():
            log.warning("pinned but not visible in mount: %s (mount-s3 may still be initializing)", path)
            continue
        size_mb = 0
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    size_mb += len(chunk) // (1024 * 1024)
            log.info("warmed %s (%s, %d MB)", path, t, size_mb)
            warmed += 1
        except Exception as e:  # noqa: BLE001
            log.exception("warm failed for %s: %s", path, e)

    log.info("warmup complete: %d warmed, %d skipped (not visible)", warmed, len(pinned) - warmed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
