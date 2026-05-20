"""
Dispatcher Lambda — entry point for ComfyUI workflow submission.

Routes:
  POST /prompt              → enqueue to image-jobs or video-jobs SQS queue
  GET  /history/{prompt_id} → return job state from DDB (ComfyUI-compatible shape)
  GET  /object_info         → merged ComfyUI /object_info from cached fleet data,
                               with model dropdowns sourced from DDB catalog
  POST /internal/object_info→ worker pushes its /object_info on boot (cached in DDB)

Authentication:
  All routes are guarded by API Gateway's Cognito authorizer. Verified claims
  arrive in event["requestContext"]["authorizer"]["claims"]. We don't need to
  verify the JWT ourselves.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from workflow_router import classify_workflow

# AWS clients are created at module load (re-used across warm invocations).
sqs = boto3.client("sqs")
ddb = boto3.client("dynamodb")
s3 = boto3.client("s3")
sm = boto3.client("secretsmanager")

JOBS_TABLE = os.environ["JOBS_TABLE"]
MODELS_TABLE = os.environ["MODELS_TABLE"]
OBJECT_INFO_TABLE = os.environ["OBJECT_INFO_TABLE"]
IMAGE_QUEUE_URL = os.environ["IMAGE_QUEUE_URL"]
VIDEO_QUEUE_URL = os.environ["VIDEO_QUEUE_URL"]
OUTPUTS_BUCKET = os.environ.get("OUTPUTS_BUCKET", "")  # Storage for userdata + settings
WS_TICKET_SECRET_ARN = os.environ.get("WS_TICKET_SECRET_ARN", "")
WS_API_ENDPOINT = os.environ.get("WS_API_ENDPOINT", "")  # wss://<id>.execute-api.<region>.amazonaws.com/v1

_ws_ticket_secret_cache: bytes | None = None

# Job records expire from DDB after 30 days (TTL attribute).
JOB_TTL = timedelta(days=30)

# In-memory cache for /object_info responses to avoid hammering DDB on every
# poll. TTL is 60s — when a model is added via the catalog Lambda, that Lambda
# also bumps a version key in DDB so we can detect changes proactively.
_object_info_cache: dict[str, Any] = {"data": None, "fetched_at": 0.0}
OBJECT_INFO_CACHE_TTL_SECONDS = 60


def lambda_handler(event: dict, context: Any) -> dict:
    method = event["httpMethod"]
    path = event["resource"]

    try:
        if method == "POST" and path == "/prompt":
            return _post_prompt(event)
        if method == "GET" and path == "/history":
            return _get_history_list(event)
        if method == "GET" and path == "/history/{id}":
            return _get_history(event)
        if method == "GET" and path == "/object_info":
            return _get_object_info(event)
        if method == "POST" and path == "/internal/object_info":
            return _post_internal_object_info(event)
        if method == "GET" and path == "/ws-ticket":
            return _get_ws_ticket(event)

        # `path` above is event["resource"] (the API GW route pattern, e.g.
        # "/customnode/{proxy+}"). For routing decisions below we want the
        # actual URL the client hit ("/customnode/install/git_url"), which
        # API GW puts in event["path"].
        url_path = event.get("path", "")

        # ComfyUI-Manager: model installs through Manager UI use a different
        # path than custom-node installs. Translate to our /models/download
        # Lambda so the file lands in S3 + DDB + editor dropdown.
        if method == "POST" and url_path == "/manager/queue/install_model":
            return _manager_install_model(event)

        # Proxy ComfyUI-Manager runtime endpoints to the metadata instance.
        # API GW resources for these are configured in ApiStack.
        for proxy_base in ("/manager/", "/customnode/", "/snapshot/", "/model-manager/"):
            if url_path.startswith(proxy_base) and method in ("GET", "POST"):
                # For install routes: write manifest first, then proxy.
                # /customnode/install* is the legacy Manager install path;
                # /manager/queue/install is ComfyUI-Manager V3's queue-based
                # install (what the current editor actually calls). Both must
                # be recorded to the manifest or the node won't survive a
                # restart — the metadata container and GPU workers reinstall
                # custom nodes from the manifest on boot.
                if url_path in (
                    "/customnode/install",
                    "/customnode/install/git_url",
                    "/manager/queue/install",
                ):
                    return _customnode_install_then_record(event, proxy_base)
                return _proxy_to_metadata(event, proxy_base)

        # Stub endpoints that ComfyUI's frontend polls but we don't need full impl
        if method == "GET" and path == "/queue":
            return _resp(200, {"queue_running": [], "queue_pending": []})
        if method == "GET" and path == "/system_stats":
            return _resp(200, {
                "system": {"comfyui_version": "remote", "ram_total": 0, "ram_free": 0},
                "devices": [],
            })
        if method == "GET" and path == "/embeddings":
            return _resp(200, [])
        if method == "GET" and path == "/extensions":
            return _get_extensions()
        if method == "POST" and path == "/internal/extensions":
            return _post_internal_extensions(event)

        # New ComfyUI editor (Comfy-Org/ComfyUI_frontend v1.x) expects these.
        # Most are stubs — return defaults so the editor loads cleanly.
        if method == "GET" and path == "/users":
            return _resp(200, _users_response(event))
        if method == "POST" and path == "/users":
            return _resp(200, {"username": _claim_username(event)})
        if method == "GET" and path == "/i18n":
            return _resp(200, {})
        if method == "GET" and path == "/i18n/{locale}":
            return _resp(200, {})
        if method == "POST" and path == "/free":
            return _resp(200, {"freed": True})
        if method == "GET" and path == "/workflow_templates":
            return _resp(200, {})
        if method == "GET" and path == "/global_subgraphs":
            return _resp(200, {})
        if method == "GET" and path == "/experiment/models":
            return _resp(200, _experiment_models())
        if method == "GET" and path == "/experiment/models/{folder}":
            return _resp(200, _experiment_models_for(event["pathParameters"]["folder"]))

        # Storage-backed: settings + userdata. Each user gets a prefix in S3.
        if method == "GET" and path == "/settings":
            return _get_settings(event)
        if method == "POST" and path == "/settings":
            return _post_settings(event)
        if method == "GET" and path == "/settings/{id}":
            return _get_setting(event)
        if method == "POST" and path == "/settings/{id}":
            return _post_setting(event)
        if method == "DELETE" and path == "/settings/{id}":
            return _delete_setting(event)
        if method == "GET" and path == "/userdata":
            return _list_userdata(event)
        if method == "GET" and path == "/userdata/{file+}":
            return _get_userdata(event)
        if method == "POST" and path == "/userdata/{file+}":
            return _post_userdata(event)
        if method == "DELETE" and path == "/userdata/{file+}":
            return _delete_userdata(event)
        if method == "POST" and path == "/userdata/{file+}/move/{dest+}":
            return _move_userdata(event)

        return _resp(404, {"error": "not found", "method": method, "path": path})
    except Exception as e:  # noqa: BLE001 — return JSON to client, log to CW
        print(f"ERROR {method} {path}: {e!r}")
        return _resp(500, {"error": "internal error"})


# ----- POST /prompt -----
def _post_prompt(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")

    workflow = body.get("prompt") or body.get("workflow")
    if not workflow or not isinstance(workflow, dict):
        return _resp(400, {"error": "missing or invalid 'prompt' (workflow JSON)"})

    explicit_type = body.get("type")
    if explicit_type in ("image", "video"):
        job_type = explicit_type
    else:
        # v3 C3 fallback: if frontend didn't specify, sniff workflow class_types.
        job_type = classify_workflow(workflow)

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expire_at = int((now + JOB_TTL).timestamp())

    queue_url = IMAGE_QUEUE_URL if job_type == "image" else VIDEO_QUEUE_URL

    # Write job record FIRST so even if SQS send fails the job is recoverable.
    ddb.put_item(
        TableName=JOBS_TABLE,
        Item={
            "job_id": {"S": job_id},
            "type": {"S": job_type},
            "status": {"S": "queued"},
            "workflow_json": {"S": json.dumps(workflow)},
            "client_id": {"S": body.get("client_id", "")},
            "created_at": {"S": now.isoformat()},
            "expire_at": {"N": str(expire_at)},
            "attempt_count": {"N": "0"},
        },
    )

    # SQS message: small (just job_id + type). Worker reads full workflow from DDB.
    # Pure SQS body avoids hitting the 256 KB SQS message limit on big workflows.
    # If SQS send fails, mark the job failed so user gets a clear signal in /jobs/{id}
    # (resolves code review N13).
    try:
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps({"job_id": job_id, "type": job_type}),
            MessageAttributes={
                "job_id": {"DataType": "String", "StringValue": job_id},
                "type": {"DataType": "String", "StringValue": job_type},
            },
        )
    except Exception as e:  # noqa: BLE001
        ddb.update_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
            UpdateExpression="SET #s = :s, #e = :e",
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
            ExpressionAttributeValues={
                ":s": {"S": "failed"},
                ":e": {"S": f"sqs send failed: {e}"[:500]},
            },
        )
        return _resp(503, {"error": "queue temporarily unavailable; job marked failed"})

    # Match ComfyUI's native /prompt response shape so existing client libraries work.
    return _resp(
        200,
        {
            "prompt_id": job_id,
            "number": 1,
            "node_errors": {},
        },
    )


# ----- GET /history/{id} -----
def _get_history(event: dict) -> dict:
    job_id = event["pathParameters"]["id"]
    try:
        result = ddb.get_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
        )
    except ClientError as e:
        print(f"ddb error: {e}")
        return _resp(500, {})

    item = result.get("Item")
    if not item:
        # ComfyUI returns empty {} for unknown prompt_ids
        return _resp(200, {})

    status = item["status"]["S"]
    output_keys = json.loads(item.get("output_keys", {"S": "[]"})["S"])

    # ComfyUI history shape: { "<prompt_id>": { "outputs": {...}, "status": {...} } }
    history_entry = {
        "outputs": _format_outputs(output_keys),
        "status": {
            "status_str": _comfyui_status_str(status),
            "completed": status in ("complete", "failed"),
            "messages": [],
        },
    }
    if status == "failed":
        history_entry["status"]["messages"].append(
            ["execution_error", {"message": item.get("error", {"S": ""})["S"]}]
        )

    return _resp(200, {job_id: history_entry})


def _format_outputs(s3_keys: list[str]) -> dict:
    """Convert worker's S3 output keys into ComfyUI's expected output shape.
    Each entry is grouped under a node_id; ComfyUI's frontend uses the
    'images'/'videos' arrays to render results. The frontend will then call
    /view?filename=...&type=output for the actual bytes.
    """
    images: list[dict] = []
    videos: list[dict] = []
    for key in s3_keys:
        filename = key.rsplit("/", 1)[-1]
        ref = {"filename": filename, "subfolder": "", "type": "output", "s3_key": key}
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("mp4", "webm", "gif", "webp", "mkv", "mov"):
            videos.append(ref)
        else:
            images.append(ref)

    # ComfyUI expects outputs keyed by node_id. We use a synthetic "0" since the
    # worker doesn't preserve per-node mappings; this is a known limitation.
    out: dict[str, Any] = {"0": {}}
    if images:
        out["0"]["images"] = images
    if videos:
        out["0"]["gifs"] = videos
    return out


def _comfyui_status_str(internal_status: str) -> str:
    return {
        "queued": "executing",
        "running": "executing",
        "complete": "success",
        "failed": "error",
    }.get(internal_status, "executing")


# ----- GET /object_info -----
def _get_object_info(_event: dict) -> dict:
    """Return a merged /object_info from cached fleet data, with model
    dropdowns swapped for the live DDB catalog.
    """
    now = time.time()
    if (
        _object_info_cache["data"] is None
        or now - _object_info_cache["fetched_at"] > OBJECT_INFO_CACHE_TTL_SECONDS
    ):
        _object_info_cache["data"] = _build_object_info()
        _object_info_cache["fetched_at"] = now

    return _resp(200, _object_info_cache["data"])


def _build_object_info() -> dict:
    """Read worker-pushed /object_info from S3 (large), union image + video,
    swap model name lists for live DDB catalog values.

    DDB row stores object_info_s3_key (the S3 location of the full JSON);
    object_info JSON is too big for the 400 KB DDB item limit (~1-5 MB for
    a typical install).
    """
    merged: dict[str, Any] = {}
    for fleet in ("image", "video"):
        try:
            r = ddb.get_item(TableName=OBJECT_INFO_TABLE, Key={"fleet": {"S": fleet}})
            item = r.get("Item")
            if not item:
                continue
            # Prefer S3-stored key; fall back to legacy inline json for older rows
            s3_key = item.get("object_info_s3_key", {}).get("S")
            if s3_key and OUTPUTS_BUCKET:
                try:
                    s3_resp = s3.get_object(Bucket=OUTPUTS_BUCKET, Key=s3_key)
                    fleet_oi = json.loads(s3_resp["Body"].read().decode())
                    merged.update(fleet_oi)
                except (ClientError, json.JSONDecodeError):
                    pass
            else:
                inline = item.get("object_info_json", {}).get("S")
                if inline:
                    try:
                        fleet_oi = json.loads(inline)
                        merged.update(fleet_oi)
                    except json.JSONDecodeError:
                        pass
        except ClientError:
            pass

    # Replace model dropdowns with current catalog.
    #
    # The metadata instance has no model files on disk, so ComfyUI reports
    # empty lists ([]) for every model-name dropdown. We swap whenever the
    # input *spec* is a list (the ComfyUI signature for a dropdown) AND the
    # input name maps to a known model type — including the empty case, so
    # the editor sees the real catalog instead of "no choices".
    catalog_by_type = _scan_catalog_by_type()
    for node_class, schema in merged.items():
        inputs = schema.get("input", {}).get("required", {})
        for input_name, spec in inputs.items():
            if not isinstance(spec, list) or not spec:
                continue
            choices_or_type = spec[0]
            if not isinstance(choices_or_type, list):
                continue
            # If non-empty, require all-strings (a real dropdown, not e.g.
            # a tuple type like ["INT", {...}]).
            if choices_or_type and not all(isinstance(x, str) for x in choices_or_type):
                continue
            model_type = _guess_model_type_from_input(node_class, input_name)
            if model_type and model_type in catalog_by_type:
                spec[0] = sorted(catalog_by_type[model_type])

    return merged


_INPUT_TO_MODEL_TYPE = {
    "ckpt_name": "checkpoint",
    "lora_name": "lora",
    "vae_name": "vae",
    "control_net_name": "controlnet",
    "controlnet_name": "controlnet",
    "clip_name": "clip",
    "clip_vision_name": "clip_vision",
    "text_encoder_name": "text_encoders",
    "embedding_name": "embedding",
    "unet_name": "diffusion_models",      # ComfyUI deprecated 'unet/' → 'diffusion_models/'
    "diffusion_model_name": "diffusion_models",
    "style_model_name": "style_models",
    "gligen_name": "gligen",
    "hypernetwork_name": "hypernetworks",
    "photomaker_name": "photomaker",
    "upscale_model_name": "upscale",
    # 'model_name' is ambiguous — handled by _guess_model_type_from_input below
    # so we can use node_class to disambiguate Wan/Flux/etc.
}


def _guess_model_type_from_input(node_class: str, input_name: str) -> str | None:
    """Map ComfyUI input names (and node-class hints) to our DDB catalog 'type'.
    Falls back to substring heuristics when there's no direct mapping."""
    direct = _INPUT_TO_MODEL_TYPE.get(input_name)
    if direct:
        return direct

    name_lc = input_name.lower()
    class_lc = node_class.lower()

    # Wan / Hunyuan / Flux / SD3 — modern transformer-based models live in diffusion_models/
    if "model" in name_lc and any(
        k in class_lc for k in ("wan", "hunyuan", "flux", "sd3", "ltx", "cog", "pyramid")
    ):
        return "diffusion_models"
    # Wan/Flux text encoders
    if "text_encoder" in name_lc or ("encoder" in name_lc and "audio" not in name_lc):
        return "text_encoders"
    if "lora" in name_lc:
        return "lora"
    if "vae" in name_lc:
        return "vae"
    if "clip_vision" in name_lc or "clipvision" in name_lc:
        return "clip_vision"
    if "controlnet" in name_lc or "control_net" in name_lc:
        return "controlnet"
    if "upscale" in name_lc:
        return "upscale"
    return None


def _scan_catalog_by_type() -> dict[str, list[str]]:
    """Pull all models from DDB catalog grouped by type.

    Returns filenames AS THEY APPEAR ON DISK (i.e., with .safetensors / .ckpt
    extension), because ComfyUI's prompt validator compares ckpt_name against
    folder_paths.get_filename_list() which returns full filenames. If we
    returned the bare catalog name, the editor's workflow stores it bare,
    worker submits, ComfyUI rejects ("value_not_in_list").

    Filename = basename of the S3 key (which always carries the extension).
    Falls back to the catalog name if no s3_key (shouldn't happen).
    """
    grouped: dict[str, list[str]] = {}
    paginator = ddb.get_paginator("scan")
    for page in paginator.paginate(TableName=MODELS_TABLE):
        for item in page.get("Items", []):
            t = item.get("type", {}).get("S", "")
            n = item.get("name", {}).get("S", "")
            s3_key = item.get("s3_key", {}).get("S", "")
            if not t or not n:
                continue
            filename = s3_key.rsplit("/", 1)[-1] if s3_key else n
            grouped.setdefault(t, []).append(filename)
    return grouped


# ----- Manager passthrough proxy -----
# Forwards /manager/* /customnode/* /snapshot/* /model-manager/* HTTP requests
# to the metadata instance's ComfyUI process (which has Manager installed).
# Without this, Manager's frontend init aborts on CORS preflight 403.
import urllib.request
import urllib.error

_metadata_url_cache: Optional[str] = None  # type: ignore[name-defined]


def _get_metadata_url() -> str:
    global _metadata_url_cache
    if _metadata_url_cache:
        return _metadata_url_cache
    try:
        ssm = boto3.client("ssm")
        r = ssm.get_parameter(Name="/comfy/metadata/url")
        _metadata_url_cache = r["Parameter"]["Value"]
        return _metadata_url_cache
    except Exception as e:  # noqa: BLE001
        print(f"failed to fetch /comfy/metadata/url: {e!r}")
        return ""


def _proxy_to_metadata(event: dict, _base: str, timeout: int = 8) -> dict:
    metadata_base = _get_metadata_url()
    if not metadata_base:
        return _resp(503, {"error": "metadata instance URL unknown"})

    # event["resource"] = "/manager/{proxy+}"; pathParameters['proxy'] has actual path
    pp = event.get("pathParameters") or {}
    proxy_path = pp.get("proxy", "")
    # event["path"] is the actual matched URL (e.g. "/manager/version")
    full_path = event.get("path", "")
    qs = event.get("queryStringParameters") or {}
    qs_str = ""
    if qs:
        from urllib.parse import urlencode
        qs_str = "?" + urlencode(qs)

    target_url = f"{metadata_base.rstrip('/')}{full_path}{qs_str}"
    method = event["httpMethod"]
    body = event.get("body") or ""
    body_bytes = body.encode() if body else None
    headers = {"Content-Type": event.get("headers", {}).get("Content-Type", "application/json")}

    req = urllib.request.Request(target_url, data=body_bytes, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "application/json")
            try:
                body_str = data.decode("utf-8")
            except UnicodeDecodeError:
                body_str = ""
            return {
                "statusCode": resp.status,
                "headers": {
                    "Content-Type": ctype,
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,Authorization,Comfy-User",
                },
                "body": body_str,
            }
    except urllib.error.HTTPError as e:
        return {
            "statusCode": e.code,
            "headers": {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"},
            "body": json.dumps({"error": e.reason}),
        }
    except urllib.error.URLError as e:
        # Differentiate "timed out reading response" (504) from "couldn't
        # connect at all" (502) so the install path can return success on
        # timeout. urllib raises URLError with reason=socket.timeout for both.
        msg = str(e.reason)
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            return _resp(504, {"error": f"metadata timeout: {msg}"})
        return _resp(502, {"error": f"metadata unreachable: {msg}"})
    except Exception as e:  # noqa: BLE001
        return _resp(500, {"error": str(e)})


# ----- Manager UI installers (intercepts that update the S3 manifest) -----
#
# Architecture:
#   1. User clicks Install in Manager UI → POST /customnode/install (or /install/git_url)
#   2. We forward to the metadata instance so Manager actually clones the repo
#      and the new node shows up in the editor's /object_info ASAP.
#   3. If the metadata-side install succeeds, append an entry to the S3 manifest
#      so any GPU worker launched after this also installs the node on boot.
#
# Models work differently — Manager's downloader writes to local disk which is
# useless to us. We translate /manager/queue/install_model to our existing
# /models/download Lambda flow which streams CivitAI → S3 → DDB → dropdowns.

MANIFEST_BUCKET = OUTPUTS_BUCKET  # use the same bucket as object_info storage
MANIFEST_KEY = "manifests/custom-nodes.json"


def _customnode_install_then_record(event: dict, base: str) -> dict:
    """Append entry to S3 manifest FIRST, then proxy to metadata.

    Order matters: heavy installs (pip-installing scipy, etc.) can run for
    minutes on the metadata container — longer than API GW's 29-sec limit.
    By writing the manifest first, the install is durable even when our
    proxy times out: workers will git-clone + pip-install it on next boot.
    """
    raw_body = event.get("body") or ""
    if event["path"].endswith("/git_url") and raw_body and not raw_body.lstrip().startswith("{"):
        body: Any = {"url": raw_body.strip()}
    else:
        try:
            body = json.loads(raw_body or "{}")
        except json.JSONDecodeError:
            body = {}
    entry = _normalize_install_body(event["path"], body)
    if entry:
        try:
            _append_to_manifest(entry)
        except Exception:  # noqa: BLE001
            print(f"WARN: failed to append to manifest before proxy: {entry!r}")

    resp = _proxy_to_metadata(event, base, timeout=25)
    # If the metadata-side install timed out, still report success — the
    # manifest entry is in place so workers will install it on next boot.
    if resp.get("statusCode") == 504 and entry:
        return _resp(200, {"status": "queued-via-manifest"})
    return resp


def _normalize_install_body(path: str, body: dict) -> Optional[dict]:
    """Coerce the various Manager install request shapes into our manifest entry."""
    now = datetime.now(timezone.utc).isoformat()
    # /customnode/install/git_url: body = {"url": "<git url>"}
    if path.endswith("/git_url"):
        url = body.get("url") or body.get("git_url")
        if not url:
            return None
        return {
            "name": _name_from_git_url(url),
            "url": url,
            "commit": None,
            "added_at": now,
            "source": "manager-git-url",
        }
    # /manager/queue/install: ComfyUI-Manager V3 sends the node-pack object —
    # {id, title, files: ["<git_url>"], repository, reference, version,
    #  selected_version, install_type, ...}. `id` is the cnr id and also the
    # directory name Manager clones into, so it's the correct manifest `name`.
    if path.endswith("/queue/install"):
        files = body.get("files") or []
        url = body.get("repository") or (files[0] if files else None) or body.get("url")
        if not url:
            # `reference` may be a registry URL (registry.comfy.org/...) which
            # is not git-cloneable — only accept it if it's a real git host.
            ref = body.get("reference") or ""
            if "github.com" in ref or "gitlab.com" in ref or ref.endswith(".git"):
                url = ref
        if not url:
            return None
        return {
            # commit pinned to None: Manager's cnr `selected_version` (e.g.
            # "1.0.9") is not necessarily a git ref/tag, so track HEAD.
            "name": body.get("id") or body.get("title") or _name_from_git_url(url),
            "url": url,
            "commit": None,
            "added_at": now,
            "source": "manager-queue-install",
        }
    # /customnode/install: body = {title, files: ["<git_url>", ...], install_type, ...}
    title = body.get("title") or body.get("name")
    files = body.get("files") or []
    url = files[0] if files else body.get("reference") or body.get("url")
    if not url:
        return None
    return {
        "name": title or _name_from_git_url(url),
        "url": url,
        "commit": body.get("commit") or None,
        "added_at": now,
        "source": "manager-install",
    }


def _name_from_git_url(url: str) -> str:
    last = url.rstrip("/").rsplit("/", 1)[-1]
    return last[:-4] if last.endswith(".git") else last


def _append_to_manifest(entry: dict) -> None:
    """Read-modify-write the manifest. Idempotent: if an entry with the same
    name+url already exists, update its added_at + source rather than dupe."""
    if not MANIFEST_BUCKET:
        return
    try:
        r = s3.get_object(Bucket=MANIFEST_BUCKET, Key=MANIFEST_KEY)
        manifest = json.loads(r["Body"].read().decode())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            manifest = {"version": 1, "nodes": []}
        else:
            raise
    nodes = manifest.setdefault("nodes", [])
    existing = next(
        (n for n in nodes if n.get("name") == entry["name"] and n.get("url") == entry["url"]),
        None,
    )
    if existing:
        existing.update(entry)
    else:
        nodes.append(entry)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    s3.put_object(
        Bucket=MANIFEST_BUCKET,
        Key=MANIFEST_KEY,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"manifest updated: {entry['name']} from {entry['url']}")


def _manager_install_model(event: dict) -> dict:
    """Adapt Manager's model-install request to our /models/download Lambda.

    Manager body shape (varies; tolerate both):
      {name, type, url, save_path, filename?, base?, description?}
    Our /models/download body:
      {url, model_type, name?, force?}
    """
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        body = {}
    url = body.get("url")
    if not url:
        return _resp(400, {"error": "missing url"})

    model_type = (
        body.get("type")
        or body.get("model_type")
        or _guess_model_type_from_save_path(body.get("save_path", ""))
        or "checkpoint"
    )
    name = (
        body.get("filename")
        or body.get("name")
        or url.rstrip("/").rsplit("/", 1)[-1]
    )

    # Invoke the download-kickoff Lambda asynchronously via API GW path is
    # simpler than direct lambda invoke (keeps cred handling out of dispatcher).
    # We're already inside the API; just call the same code path that
    # POST /models/download uses by constructing the equivalent body and
    # invoking the worker.
    download_fn = os.environ.get("DOWNLOAD_KICKOFF_FN")
    # Our kickoff Lambda expects 'civitai_url' and only validates civitai.com /
    # civitai.red. Manager may send HuggingFace or direct .safetensors URLs;
    # for now we forward as civitai_url and let kickoff reject non-civitai
    # cleanly (returns 400 invalid_civitai_url). Direct-URL support is a
    # follow-up — needs a generic streaming branch in services/downloader/worker.py.
    payload = {
        "body": json.dumps({
            "civitai_url": url,
            "model_type": model_type,
            "name": name,
        }),
        "httpMethod": "POST",
        "resource": "/models/download",
        "headers": event.get("headers", {}),
        "requestContext": event.get("requestContext", {}),
    }
    if download_fn:
        try:
            lam = boto3.client("lambda")
            r = lam.invoke(
                FunctionName=download_fn,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode(),
            )
            data = json.loads(r["Payload"].read().decode())
            # Translate to Manager's expected shape: it just cares that it's a
            # 200 with a non-error body, so passing through is fine.
            return data
        except Exception as e:  # noqa: BLE001
            return _resp(500, {"error": f"download invoke failed: {e!r}"})
    return _resp(503, {"error": "DOWNLOAD_KICKOFF_FN env not configured"})


_SAVE_PATH_TO_MODEL_TYPE = {
    "checkpoints": "checkpoint",
    "loras": "lora",
    "vae": "vae",
    "controlnet": "controlnet",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "embeddings": "embedding",
    "diffusion_models": "diffusion_models",
    "text_encoders": "text_encoders",
    "upscale_models": "upscale",
    "unet": "diffusion_models",
}


def _guess_model_type_from_save_path(save_path: str) -> Optional[str]:
    """save_path looks like 'custom_nodes/.../models/checkpoints/' or just
    'checkpoints/'. Find the last segment that maps to a model type."""
    if not save_path:
        return None
    parts = [p for p in save_path.strip("/").split("/") if p]
    for p in reversed(parts):
        if p in _SAVE_PATH_TO_MODEL_TYPE:
            return _SAVE_PATH_TO_MODEL_TYPE[p]
    return None


# ----- GET /ws-ticket (Cognito-authed) -----
# Returns a short-lived HMAC-signed ticket the browser uses to open a WS
# connection to the WebSocket API. Browser flow:
#   1. fetch /ws-ticket (with Cognito Authorization header)
#   2. Receive {ws_url, ticket, client_id}
#   3. Open new WebSocket(ws_url + "?ticket=" + ticket + "&client_id=" + client_id)
def _get_ws_ticket(event: dict) -> dict:
    if not WS_TICKET_SECRET_ARN or not WS_API_ENDPOINT:
        return _resp(503, {"error": "ws not configured"})
    username = _claim_username(event)
    # Use the client's provided client_id if any; else generate one. Returning
    # it lets the browser use the same id when submitting prompts so events
    # route back through.
    qs = event.get("queryStringParameters") or {}
    client_id = qs.get("client_id") or str(uuid.uuid4())

    # 60-sec ticket: just enough to open the WS, short enough to limit replay.
    payload = {
        "sub": username,
        "client_id": client_id,
        "exp": int(time.time()) + 60,
    }
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).rstrip(b"=").decode()
    sig = hmac.new(_get_ws_ticket_secret(), payload_b64.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    ticket = f"{payload_b64}.{sig_b64}"
    return _resp(200, {"ws_url": WS_API_ENDPOINT, "ticket": ticket, "client_id": client_id})


def _get_ws_ticket_secret() -> bytes:
    global _ws_ticket_secret_cache
    if _ws_ticket_secret_cache is None:
        r = sm.get_secret_value(SecretId=WS_TICKET_SECRET_ARN)
        _ws_ticket_secret_cache = r["SecretString"].encode("utf-8")
    return _ws_ticket_secret_cache


# ----- /extensions and /internal/extensions -----
# Workers POST their list of extension JS file paths on boot. Dispatcher
# returns the merged list to the editor. Files themselves live in the
# frontend S3 bucket (uploaded by extensions_publisher) and are loaded by
# the editor directly from there (same-origin, no api shim involved).
def _get_extensions() -> dict:
    """Return merged extension list from both fleets."""
    merged: list[str] = []
    seen: set[str] = set()
    for fleet in ("image", "video"):
        try:
            r = ddb.get_item(TableName=OBJECT_INFO_TABLE, Key={"fleet": {"S": fleet}})
            item = r.get("Item")
            if not item:
                continue
            ext_json = item.get("extensions_json", {}).get("S", "[]")
            for path in json.loads(ext_json):
                if path not in seen:
                    seen.add(path)
                    merged.append(path)
        except (ClientError, json.JSONDecodeError):
            continue
    return _resp(200, merged)


def _post_internal_extensions(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")
    fleet = body.get("fleet")
    extensions = body.get("extensions")
    if fleet not in ("image", "video") or not isinstance(extensions, list):
        return _resp(400, {"error": "fleet must be image|video, extensions must be list"})
    # Update the fleet's row in the object_info table with the extensions list.
    # We use UpdateItem rather than PutItem so we don't clobber the
    # object_info_json attribute.
    ddb.update_item(
        TableName=OBJECT_INFO_TABLE,
        Key={"fleet": {"S": fleet}},
        UpdateExpression="SET extensions_json = :e, ext_updated_at = :t",
        ExpressionAttributeValues={
            ":e": {"S": json.dumps(extensions)},
            ":t": {"S": datetime.now(timezone.utc).isoformat()},
        },
    )
    return _resp(200, {"ok": True, "count": len(extensions)})


# ----- POST /internal/object_info (worker → dispatcher) -----
# object_info from a real install is 1-5 MB JSON (way past DDB's 400 KB item
# limit). Store the JSON in S3 (outputs bucket, metadata/ prefix), keep just
# the S3 key in DDB. _build_object_info reads via the S3 key.
def _post_internal_object_info(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")
    fleet = body.get("fleet")
    object_info = body.get("object_info")
    if fleet not in ("image", "video") or not isinstance(object_info, dict):
        return _resp(400, {"error": "fleet must be image|video, object_info must be dict"})
    if not OUTPUTS_BUCKET:
        return _resp(500, {"error": "OUTPUTS_BUCKET unset"})

    s3_key = f"metadata/object_info/{fleet}.json"
    try:
        s3.put_object(
            Bucket=OUTPUTS_BUCKET,
            Key=s3_key,
            Body=json.dumps(object_info).encode(),
            ContentType="application/json",
        )
    except ClientError as e:
        print(f"S3 put_object failed: {e!r}")
        return _resp(500, {"error": "s3 write failed"})

    # UpdateItem (not PutItem) preserves the extensions_json attribute that
    # _post_internal_extensions writes on the same row.
    ddb.update_item(
        TableName=OBJECT_INFO_TABLE,
        Key={"fleet": {"S": fleet}},
        UpdateExpression="SET object_info_s3_key = :k, updated_at = :t REMOVE object_info_json",
        ExpressionAttributeValues={
            ":k": {"S": s3_key},
            ":t": {"S": datetime.now(timezone.utc).isoformat()},
        },
    )
    # Invalidate this Lambda's in-memory cache so next /object_info re-fetches.
    _object_info_cache["data"] = None
    return _resp(200, {"ok": True, "s3_key": s3_key, "size_bytes": len(json.dumps(object_info))})


def _resp(status: int, body: Any, content_type: str = "application/json") -> dict:
    if content_type == "application/json":
        body_str = json.dumps(body) if not isinstance(body, str) else body
    else:
        body_str = body if isinstance(body, str) else str(body)
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": content_type,
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization,Comfy-User",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        },
        "body": body_str,
    }


# ============================================================================
# ComfyUI editor compatibility endpoints
# ============================================================================
#
# The new Comfy-Org/ComfyUI_frontend (v1.x) calls these on startup. Most are
# stubs returning empty defaults; settings + userdata are storage-backed using
# S3 prefixes under the outputs bucket (path: userdata/<username>/...).
#
# Username comes from the Cognito ID token claims that API GW passes through
# in event["requestContext"]["authorizer"]["claims"]["cognito:username"].

def _claim_username(event: dict) -> str:
    """Pull the Cognito username from the verified JWT claims."""
    try:
        claims = event["requestContext"]["authorizer"]["claims"]
        return claims.get("cognito:username") or claims.get("sub", "anonymous")
    except (KeyError, TypeError):
        return "anonymous"


def _userdata_prefix(event: dict, suffix: str = "") -> str:
    user = _claim_username(event)
    base = f"userdata/{user}/"
    return base + suffix.lstrip("/")


# ----- /users -----
def _users_response(event: dict) -> dict:
    """Single-user-per-account model. Editor expects {storage, users}.

    Why "browser": Comfy-Org/ComfyUI_frontend's multi-user picker triggers
    when storage="server" with one-or-more users — the editor redirects to
    /user-select on load, blocking the canvas. For a one-Cognito-user
    deployment that's pure friction. "browser" tells the editor to skip the
    picker entirely; per-user *settings* end up in localStorage rather than
    server-side, but our /userdata routes still work for actual file storage
    (workflows, custom CSS).
    """
    del event
    return {"storage": "browser"}


# ----- /settings -----
# All settings stored as a single JSON object at userdata/<user>/.settings.json.
# GET /settings → whole object. POST /settings → replace whole object.
# GET /settings/{id} → one value. POST /settings/{id} → set one value.
def _settings_key(event: dict) -> str:
    return _userdata_prefix(event, ".settings.json")


def _load_settings(event: dict) -> dict:
    try:
        r = s3.get_object(Bucket=OUTPUTS_BUCKET, Key=_settings_key(event))
        return json.loads(r["Body"].read().decode())
    except s3.exceptions.NoSuchKey:
        return {}
    except (ClientError, json.JSONDecodeError):
        return {}


def _save_settings(event: dict, settings: dict) -> None:
    s3.put_object(
        Bucket=OUTPUTS_BUCKET,
        Key=_settings_key(event),
        Body=json.dumps(settings).encode(),
        ContentType="application/json",
    )


def _get_settings(event: dict) -> dict:
    return _resp(200, _load_settings(event))


def _post_settings(event: dict) -> dict:
    body = json.loads(event.get("body") or "{}")
    if isinstance(body, dict):
        _save_settings(event, body)
    return _resp(200, {})


def _get_setting(event: dict) -> dict:
    setting_id = event["pathParameters"]["id"]
    settings = _load_settings(event)
    return _resp(200, settings.get(setting_id))


def _post_setting(event: dict) -> dict:
    setting_id = event["pathParameters"]["id"]
    body_raw = event.get("body") or "null"
    try:
        value = json.loads(body_raw)
    except json.JSONDecodeError:
        value = body_raw
    settings = _load_settings(event)
    settings[setting_id] = value
    _save_settings(event, settings)
    return _resp(200, {})


def _delete_setting(event: dict) -> dict:
    setting_id = event["pathParameters"]["id"]
    settings = _load_settings(event)
    settings.pop(setting_id, None)
    _save_settings(event, settings)
    return _resp(200, {})


# ----- /userdata -----
# File-like storage. Each user gets a prefix; nested paths supported via
# {file+} proxy. ComfyUI saves workflow JSON, default workflow, etc. here.
def _userdata_filepath(event: dict) -> str:
    raw = event["pathParameters"].get("file") or event["pathParameters"].get("file+", "")
    return _userdata_prefix(event, raw)


def _get_userdata(event: dict) -> dict:
    key = _userdata_filepath(event)
    qs = event.get("queryStringParameters") or {}
    full = qs.get("full") in ("true", "1")
    try:
        r = s3.get_object(Bucket=OUTPUTS_BUCKET, Key=key)
        body = r["Body"].read().decode("utf-8", errors="replace")
        ctype = r.get("ContentType", "application/octet-stream")
        if not full and ctype == "application/json":
            return _resp(200, json.loads(body))
        return _resp(200, body, content_type=ctype)
    except s3.exceptions.NoSuchKey:
        return _resp(404, {"error": "not found"})


def _post_userdata(event: dict) -> dict:
    key = _userdata_filepath(event)
    raw_body = event.get("body") or ""
    is_b64 = event.get("isBase64Encoded", False)
    if is_b64:
        import base64
        body_bytes = base64.b64decode(raw_body)
    else:
        body_bytes = raw_body.encode("utf-8")
    qs = event.get("queryStringParameters") or {}
    overwrite = qs.get("overwrite", "true") not in ("false", "0")
    if not overwrite:
        try:
            s3.head_object(Bucket=OUTPUTS_BUCKET, Key=key)
            return _resp(409, {"error": "exists"})
        except ClientError:
            pass
    s3.put_object(Bucket=OUTPUTS_BUCKET, Key=key, Body=body_bytes)
    return _resp(200, {"path": key})


def _delete_userdata(event: dict) -> dict:
    key = _userdata_filepath(event)
    s3.delete_object(Bucket=OUTPUTS_BUCKET, Key=key)
    return _resp(200, {})


def _move_userdata(event: dict) -> dict:
    src_key = _userdata_filepath(event)
    dest_raw = event["pathParameters"].get("dest") or event["pathParameters"].get("dest+", "")
    dest_key = _userdata_prefix(event, dest_raw)
    try:
        s3.copy_object(
            Bucket=OUTPUTS_BUCKET,
            Key=dest_key,
            CopySource={"Bucket": OUTPUTS_BUCKET, "Key": src_key},
        )
        s3.delete_object(Bucket=OUTPUTS_BUCKET, Key=src_key)
        return _resp(200, {"moved": dest_key})
    except ClientError as e:
        return _resp(404, {"error": str(e)})


def _list_userdata(event: dict) -> dict:
    qs = event.get("queryStringParameters") or {}
    sub_dir = qs.get("dir", "")
    recurse = qs.get("recurse") in ("true", "1")
    prefix = _userdata_prefix(event, sub_dir)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    paginator = s3.get_paginator("list_objects_v2")
    files: list[dict] = []
    user_prefix = _userdata_prefix(event)
    for page in paginator.paginate(Bucket=OUTPUTS_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(user_prefix):] if key.startswith(user_prefix) else key
            if not recurse and "/" in relative.lstrip("/"):
                continue
            files.append({
                "path": relative,
                "size": obj["Size"],
                "modified": obj["LastModified"].timestamp(),
            })
    return _resp(200, files)


# ----- /experiment/models -----
def _experiment_models() -> dict:
    """Return the model catalog grouped by type, in the editor's expected
    folder structure shape."""
    grouped = _scan_catalog_by_type()
    return {folder: list(names) for folder, names in grouped.items()}


def _experiment_models_for(folder: str) -> list:
    grouped = _scan_catalog_by_type()
    return list(grouped.get(folder, []))


# ----- /history (no id, list mode) -----
def _get_history_list(event: dict) -> dict:
    """Return a list of recent jobs in ComfyUI's history shape."""
    qs = event.get("queryStringParameters") or {}
    max_items = min(int(qs.get("max_items", "200")), 500)
    paginator = ddb.get_paginator("query")
    history: dict[str, Any] = {}
    try:
        for page in paginator.paginate(
            TableName=JOBS_TABLE,
            IndexName="status-index",
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": {"S": "complete"}},
            ScanIndexForward=False,
            Limit=max_items,
        ):
            for item in page.get("Items", []):
                job_id = item["job_id"]["S"]
                output_keys = json.loads(item.get("output_keys", {"S": "[]"})["S"])
                history[job_id] = {
                    "outputs": _format_outputs(output_keys),
                    "status": {"status_str": "success", "completed": True, "messages": []},
                }
            if len(history) >= max_items:
                break
    except ClientError:
        pass
    return _resp(200, history)
