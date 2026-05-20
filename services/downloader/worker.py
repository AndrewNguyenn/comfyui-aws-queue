"""
Download Worker Lambda — long-running. Streams CivitAI → S3 multipart upload.

Invoked asynchronously by `kickoff.py`. Reports progress to DDB throughout.
On completion, inserts the model into the catalog so it shows up immediately
in /object_info dropdowns.

Lambda config (set in CDK ApiStack):
  - Memory: 1 GB (sufficient for 8 MB chunked streaming + headroom)
  - Timeout: 15 min (Lambda max)
  - Architecture: arm64
  - Ephemeral storage: default 512 MB (no big disk needed; streaming)
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import boto3
import urllib3

s3 = boto3.client("s3")
ddb = boto3.client("dynamodb")
sm = boto3.client("secretsmanager")

MODELS_BUCKET = os.environ["MODELS_BUCKET"]
MODELS_TABLE = os.environ["MODELS_TABLE"]
DOWNLOADS_TABLE = os.environ["DOWNLOADS_TABLE"]
CIVITAI_SECRET = "civitai/api-token"

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks for S3 multipart
PROGRESS_UPDATE_BYTES = 256 * 1024 * 1024  # v3 N13: throttle progress updates to every 256 MB

http = urllib3.PoolManager(
    timeout=urllib3.Timeout(connect=10.0, read=600.0),
    retries=urllib3.Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504]),
)


def lambda_handler(event: dict, _context: Any) -> dict:
    download_id = event["download_id"]
    civitai_url = event["civitai_url"]
    model_type = event["model_type"]

    try:
        _set_status(download_id, "resolving")
        token = _civitai_token()
        version_id = _parse_version_id(civitai_url, token)
        meta = _fetch_metadata(version_id, token)
        file_meta = _pick_primary_file(meta)
        s3_key = f"{model_type}/{file_meta['name']}"
        total_bytes = int(file_meta["sizeKB"] * 1024)

        _set_total(download_id, total_bytes, file_meta["name"])
        _set_status(download_id, "downloading")

        download_url = file_meta["downloadUrl"]
        bytes_done = _stream_to_s3(download_url, token, s3_key, total_bytes, download_id)

        _add_to_catalog(file_meta, model_type, s3_key, bytes_done, version_id, meta)
        _set_status(download_id, "complete")
        return {"ok": True, "download_id": download_id, "s3_key": s3_key}

    except Exception as e:  # noqa: BLE001
        print(f"download {download_id} failed: {e!r}")
        _set_error(download_id, str(e))
        return {"ok": False, "download_id": download_id, "error": str(e)}


def _civitai_token() -> str:
    try:
        r = sm.get_secret_value(SecretId=CIVITAI_SECRET)
        token = json.loads(r["SecretString"]).get("token", "")
        return "" if token in ("", "PLACEHOLDER_NOT_SET") else token
    except Exception as e:  # noqa: BLE001
        print(f"civitai token unavailable: {e!r}")
        return ""


def _parse_version_id(url: str, token: str) -> int:
    """Resolves a CivitAI URL to a modelVersionId.

    Accepts:
      https://civitai.com/models/12345?modelVersionId=67890   (explicit version)
      https://civitai.com/api/v1/model-versions/67890         (api path)
      https://civitai.com/models/12345/some-slug              (model id only - fetch latest version)
      https://civitai.red/models/12345/some-slug              (mirror domain, same backend)

    For URLs without an explicit version, looks up the model and uses
    modelVersions[0] (CivitAI orders newest first).
    """
    m = re.search(r"modelVersionId=(\d+)", url)
    if m:
        return int(m.group(1))
    m = re.search(r"/model-versions/(\d+)", url)
    if m:
        return int(m.group(1))

    m = re.search(r"/models/(\d+)", url)
    if not m:
        raise ValueError(f"could not extract a model or version id from URL: {url}")

    model_id = int(m.group(1))
    headers = {"User-Agent": "comfyui-aws-queue/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = http.request(
        "GET",
        f"https://civitai.com/api/v1/models/{model_id}",
        headers=headers,
    )
    if r.status >= 400:
        raise RuntimeError(
            f"civitai api {r.status} looking up model {model_id}: {r.data[:200]!r}"
        )
    data = json.loads(r.data.decode())
    versions = data.get("modelVersions") or []
    if not versions:
        raise RuntimeError(f"model {model_id} has no published versions")
    return int(versions[0]["id"])


def _fetch_metadata(version_id: int, token: str) -> dict:
    headers = {"User-Agent": "comfyui-aws-queue/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = http.request(
        "GET",
        f"https://civitai.com/api/v1/model-versions/{version_id}",
        headers=headers,
    )
    if r.status >= 400:
        raise RuntimeError(f"civitai api {r.status}: {r.data[:200]!r}")
    return json.loads(r.data.decode())


def _pick_primary_file(meta: dict) -> dict:
    files = meta.get("files", [])
    if not files:
        raise RuntimeError("no files in CivitAI metadata")
    # CivitAI flags one file `primary: true` — the canonical download (e.g.
    # the pruned fp16 checkpoint). Honour it; picking "largest" instead grabs
    # the heavier full/fp32 sibling, which is bigger, slower, and not what the
    # editor's file picker defaults to.
    primary = [f for f in files if f.get("primary")]
    if primary:
        return primary[0]
    # No primary flag: prefer .safetensors, then largest.
    safetensors = [f for f in files if f.get("name", "").endswith(".safetensors")]
    candidates = safetensors or files
    return max(candidates, key=lambda f: f.get("sizeKB", 0))


_STREAM_MAX_ATTEMPTS = 6


def _stream_to_s3(
    url: str, token: str, s3_key: str, total_bytes: int, download_id: str
) -> int:
    """Stream a CivitAI download into an S3 multipart upload.

    Resilient to mid-stream connection drops: CivitAI's CDN closes long
    connections on multi-GB files (seen as IncompleteRead / ProtocolError).
    On a break we re-request with an HTTP Range header and resume from the
    byte we got to, rather than failing the whole download.
    """
    base_headers = {"User-Agent": "comfyui-aws-queue/1.0"}
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    # S3 multipart upload, 8 MB parts. Min part size is 5 MB except for last.
    mpu = s3.create_multipart_upload(Bucket=MODELS_BUCKET, Key=s3_key)
    upload_id = mpu["UploadId"]
    parts: list[dict] = []
    part_number = 1
    bytes_done = 0  # bytes committed to completed S3 parts
    last_progress_at_bytes = 0
    buffer = bytearray()  # accumulates between part flushes; survives retries

    def _flush_full_parts() -> None:
        nonlocal part_number, bytes_done, last_progress_at_bytes
        while len(buffer) >= CHUNK_SIZE:
            part_bytes = bytes(buffer[:CHUNK_SIZE])
            del buffer[:CHUNK_SIZE]
            resp = s3.upload_part(
                Bucket=MODELS_BUCKET, Key=s3_key, PartNumber=part_number,
                UploadId=upload_id, Body=part_bytes,
            )
            parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
            part_number += 1
            bytes_done += len(part_bytes)
            if bytes_done - last_progress_at_bytes >= PROGRESS_UPDATE_BYTES:
                _set_bytes_done(download_id, bytes_done)
                last_progress_at_bytes = bytes_done

    try:
        attempt = 0
        while True:
            resume_at = bytes_done + len(buffer)
            headers = dict(base_headers)
            if resume_at:
                headers["Range"] = f"bytes={resume_at}-"
            r = None
            try:
                # CivitAI redirects to a signed CDN URL — follow.
                r = http.request(
                    "GET", url, headers=headers, preload_content=False, redirect=True
                )
                if resume_at and r.status == 200:
                    # CDN ignored Range and is resending from byte 0 — splicing
                    # that onto our existing parts would corrupt the file.
                    raise RuntimeError("CDN ignored Range header; cannot resume")
                if r.status >= 400:
                    raise RuntimeError(f"download GET {r.status}")
                for chunk in r.stream(CHUNK_SIZE):
                    buffer.extend(chunk)
                    _flush_full_parts()
                break  # stream drained cleanly
            except Exception as e:  # noqa: BLE001
                got = bytes_done + len(buffer)
                if total_bytes and got >= total_bytes:
                    break  # connection dropped but we already have every byte
                attempt += 1
                if attempt >= _STREAM_MAX_ATTEMPTS:
                    raise
                print(
                    f"stream interrupted at {got}/{total_bytes} ({e!r}); "
                    f"retry {attempt}/{_STREAM_MAX_ATTEMPTS} with Range resume"
                )
                time.sleep(min(2 ** attempt, 30))
            finally:
                if r is not None:
                    try:
                        r.release_conn()
                    except Exception:  # noqa: BLE001
                        pass

        # Flush remaining buffer as the last part.
        if buffer:
            resp = s3.upload_part(
                Bucket=MODELS_BUCKET, Key=s3_key, PartNumber=part_number,
                UploadId=upload_id, Body=bytes(buffer),
            )
            parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
            bytes_done += len(buffer)
            buffer.clear()

        s3.complete_multipart_upload(
            Bucket=MODELS_BUCKET,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        _set_bytes_done(download_id, bytes_done)
        return bytes_done

    except Exception:
        try:
            s3.abort_multipart_upload(Bucket=MODELS_BUCKET, Key=s3_key, UploadId=upload_id)
        except Exception:  # noqa: BLE001
            pass
        raise


def _add_to_catalog(
    file_meta: dict,
    model_type: str,
    s3_key: str,
    size_bytes: int,
    version_id: int,
    meta: dict,
) -> None:
    name = file_meta["name"].rsplit(".", 1)[0]
    preview_url = ""
    images = meta.get("images") or []
    if images:
        preview_url = images[0].get("url", "") or ""

    item: dict[str, Any] = {
        "name": {"S": name},
        "type": {"S": model_type},
        "s3_key": {"S": s3_key},
        "size_gb": {"N": str(round(size_bytes / 1e9, 3))},
        "pinned": {"BOOL": False},
        "civitai_version_id": {"N": str(version_id)},
        "preview_url": {"S": preview_url},
        "added_at": {"S": datetime.now(timezone.utc).isoformat()},
    }
    try:
        ddb.put_item(TableName=MODELS_TABLE, Item=item)
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not add to catalog: {e!r}")


def _set_status(download_id: str, status: str) -> None:
    ddb.update_item(
        TableName=DOWNLOADS_TABLE,
        Key={"download_id": {"S": download_id}},
        UpdateExpression="SET #s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": {"S": status}},
    )


def _set_total(download_id: str, total_bytes: int, model_name: str) -> None:
    ddb.update_item(
        TableName=DOWNLOADS_TABLE,
        Key={"download_id": {"S": download_id}},
        UpdateExpression="SET total_bytes = :t, model_name = :n",
        ExpressionAttributeValues={
            ":t": {"N": str(total_bytes)},
            ":n": {"S": model_name},
        },
    )


def _set_bytes_done(download_id: str, bytes_done: int) -> None:
    ddb.update_item(
        TableName=DOWNLOADS_TABLE,
        Key={"download_id": {"S": download_id}},
        UpdateExpression="SET bytes_done = :b",
        ExpressionAttributeValues={":b": {"N": str(bytes_done)}},
    )


def _set_error(download_id: str, error: str) -> None:
    ddb.update_item(
        TableName=DOWNLOADS_TABLE,
        Key={"download_id": {"S": download_id}},
        # 'error' is a DynamoDB reserved keyword — it must be aliased, or this
        # UpdateItem itself fails and the failure is never recorded, leaving
        # the download stuck showing 'downloading' forever.
        UpdateExpression="SET #s = :s, #e = :e",
        ExpressionAttributeNames={"#s": "status", "#e": "error"},
        ExpressionAttributeValues={
            ":s": {"S": "failed"},
            ":e": {"S": error[:500]},
        },
    )
