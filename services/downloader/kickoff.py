"""
Download Kickoff Lambda — fast, sync. Returns a download_id and async-invokes
the long-running worker Lambda.

Routes (handled here):
  POST /models/download   → kick off a CivitAI download
  GET  /downloads/{id}    → return progress

The actual work happens in `worker.py`, invoked async (Event invocation type).
That allows API GW to return in <1s while the download runs up to 15 min.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

ddb = boto3.client("dynamodb")
lam = boto3.client("lambda")

DOWNLOADS_TABLE = os.environ["DOWNLOADS_TABLE"]
DOWNLOAD_WORKER_FN = os.environ["DOWNLOAD_WORKER_FN"]

# Catalog model types. Each maps to a directory under ComfyUI's models/ tree.
# When unsure: use 'diffusion_models' for raw UNet/transformer checkpoints
# (Wan 2.x, Flux, etc.) and 'text_encoders' for the matching text encoders
# (T5/UMT5/CLIP-L). Standard SD1.5/SDXL fully-baked checkpoints go in 'checkpoint'.
ALLOWED_TYPES = (
    "checkpoint",        # → models/checkpoints — full SD1.5/SDXL/SD3 checkpoints
    "diffusion_models",  # → models/diffusion_models — Wan, Flux, raw UNet/DiT (incl. GGUF)
    "text_encoders",     # → models/text_encoders — UMT5 (Wan), T5 (Flux), etc.
    "lora",              # → models/loras
    "vae",               # → models/vae
    "vae_approx",        # → models/vae_approx — TAESD-style fast preview VAEs
    "controlnet",        # → models/controlnet
    "clip",              # → models/clip — text encoder (legacy SDXL clip-L/G)
    "clip_vision",       # → models/clip_vision — image conditioning (IPAdapter, I2V)
    "embedding",         # → models/embeddings — textual inversion
    "upscale",           # → models/upscale_models — ESRGAN/etc.
    "style_models",      # → models/style_models — style transfer
    "gligen",            # → models/gligen — GLIGEN bbox-conditioned
    "hypernetworks",     # → models/hypernetworks — old-style fine-tuning
    "photomaker",        # → models/photomaker — face-conditioned generation
    "audio_encoders",    # → models/audio_encoders — for audio-conditioned video
    "model_patches",     # → models/model_patches
    "unet",              # → models/unet — deprecated alias for diffusion_models
    "ultralytics",       # → models/ultralytics — Impact-Pack detector models
    "sams",              # → models/sams — Segment Anything models (FaceDetailer)
    "ipadapter",         # → models/ipadapter — IPAdapter model weights
    "insightface",       # → models/insightface — InsightFace face-analysis sets (FaceID)
    "wildcards",         # → models/wildcards — prompt wildcard .txt files (a .zip
                         #   pack is unzipped by the download worker)
)
DOWNLOAD_TTL = timedelta(hours=24)


def lambda_handler(event: dict, _context: Any) -> dict:
    method = event["httpMethod"]
    path = event["resource"]
    try:
        if method == "POST" and path == "/models/download":
            return _kickoff(event)
        if method == "GET" and path == "/downloads/{id}":
            return _status(event)
        return _resp(404, {"error": "not found"})
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {method} {path}: {e!r}")
        return _resp(500, {"error": "internal error"})


def _kickoff(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")
    civitai_url = body.get("civitai_url", "").strip()
    model_type = body.get("model_type", "").strip().lower()
    # Optional: select a SPECIFIC file from a multi-file CivitAI version by name
    # (case-insensitive, matched in the worker). Default (absent) keeps the
    # primary-file behavior. Needed for versions that bundle non-primary files —
    # e.g. an Anima checkpoint's separate text encoder / VAE alongside the UNET.
    file_name = body.get("file_name", "").strip()

    if not civitai_url or not _looks_like_civitai_url(civitai_url):
        return _resp(400, {"error": "invalid civitai_url"})
    if model_type not in ALLOWED_TYPES:
        return _resp(400, {"error": f"model_type must be one of {ALLOWED_TYPES}"})

    download_id = str(uuid.uuid4())
    expire_at = int((datetime.now(timezone.utc) + DOWNLOAD_TTL).timestamp())

    ddb.put_item(
        TableName=DOWNLOADS_TABLE,
        Item={
            "download_id": {"S": download_id},
            "civitai_url": {"S": civitai_url},
            "model_type": {"S": model_type},
            "status": {"S": "queued"},
            "bytes_done": {"N": "0"},
            "total_bytes": {"N": "0"},
            "created_at": {"S": datetime.now(timezone.utc).isoformat()},
            "expire_at": {"N": str(expire_at)},
            **({"file_name": {"S": file_name}} if file_name else {}),
        },
    )

    # Async invoke (Event = fire-and-forget). Returns immediately.
    payload = {"download_id": download_id, "civitai_url": civitai_url, "model_type": model_type}
    if file_name:
        payload["file_name"] = file_name
    lam.invoke(
        FunctionName=DOWNLOAD_WORKER_FN,
        InvocationType="Event",
        Payload=json.dumps(payload).encode(),
    )

    return _resp(202, {"download_id": download_id, "status": "queued"})


def _status(event: dict) -> dict:
    download_id = event["pathParameters"]["id"]
    r = ddb.get_item(TableName=DOWNLOADS_TABLE, Key={"download_id": {"S": download_id}})
    if "Item" not in r:
        return _resp(404, {"error": "download not found (may have expired after 24h)"})

    item = r["Item"]
    bytes_done = int(item.get("bytes_done", {"N": "0"})["N"])
    total_bytes = int(item.get("total_bytes", {"N": "0"})["N"])
    pct = (bytes_done / total_bytes * 100.0) if total_bytes > 0 else 0.0
    return _resp(
        200,
        {
            "download_id": download_id,
            "status": item.get("status", {"S": "unknown"})["S"],
            "bytes_done": bytes_done,
            "total_bytes": total_bytes,
            "percent": round(pct, 2),
            "model_name": item.get("model_name", {"S": ""})["S"],
            "error": item.get("error", {"S": ""})["S"],
        },
    )


# Allow civitai.com and civitai.red (the latter mirrors content the main site
# filters from default views — same backend in practice for many models).
_CIVITAI_RE = re.compile(r"^https?://(www\.)?civitai\.(com|red)/")


def _looks_like_civitai_url(url: str) -> bool:
    return bool(_CIVITAI_RE.match(url))


def _resp(status: int, body: Any) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body),
    }
