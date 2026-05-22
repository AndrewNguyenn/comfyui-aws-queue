"""
Status Lambda — read-only job lookup + presigned S3 URL generation.

Routes:
  GET  /jobs             → list jobs filtered by status (editor queue/history)
  GET  /jobs/{id}        → job record from DDB
  DELETE /jobs/{id}      → delete a completed generation (S3 outputs + DDB record)
  POST /jobs/{id}/cancel → cancel a queued or running job (viewer pending strip)
  GET  /view?key=<s3>    → 302 redirect to presigned GET URL on outputs bucket
  POST /upload/image     → returns presigned PUT URL for direct browser upload
                            to uploads bucket (sidesteps Lambda 6 MB body limit)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ddb = boto3.client("dynamodb")
s3 = boto3.client("s3", config=Config(signature_version="s3v4"))

JOBS_TABLE = os.environ["JOBS_TABLE"]
OUTPUTS_BUCKET = os.environ["OUTPUTS_BUCKET"]
UPLOADS_BUCKET = os.environ["UPLOADS_BUCKET"]

PRESIGNED_GET_TTL = 3600  # 1 h — lets the viewer reuse a URL (browser image
# cache stays warm) across reloads instead of re-presigning every render.
PRESIGNED_PUT_TTL = 600  # 10 min for upload


def lambda_handler(event: dict, _context: Any) -> dict:
    method = event["httpMethod"]
    path = event["resource"]

    try:
        if method == "GET" and path == "/jobs":
            return _list_jobs(event)
        if method == "GET" and path == "/jobs/{id}":
            return _get_job(event)
        if method == "DELETE" and path == "/jobs/{id}":
            return _delete_job(event)
        if method == "POST" and path == "/jobs/{id}/cancel":
            return _cancel_job(event)
        if method == "GET" and path == "/view":
            return _view(event)
        if method == "POST" and path == "/upload/image":
            return _upload(event)
        return _resp(404, {"error": "not found"})
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {method} {path}: {e!r}")
        return _resp(500, {"error": "internal error"})


def _parse_workflow(workflow_json: str) -> dict:
    """Parse the stored workflow JSON into a node dict, or {} if unusable."""
    if not workflow_json:
        return {}
    try:
        wf = json.loads(workflow_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return wf if isinstance(wf, dict) else {}


def _extract_model(wf: dict) -> str:
    """Best-effort: the primary model a workflow used.

    A workflow loads many models (LoRA, VAE, CLIP, ControlNet…). The main
    model is either a checkpoint (CheckpointLoaderSimple → ckpt_name) or, for
    Flux/Wan-style graphs, a standalone diffusion model (UNETLoader /
    UnetLoaderGGUF / a 'Load Diffusion Model' node → unet_name). We scan for
    either, skipping the auxiliary loaders, and prefer a checkpoint."""
    ckpt = diffusion = None
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").lower()
        inp = node.get("inputs", {}) or {}
        # Skip auxiliary loaders — none of these is "the model".
        if any(x in ct for x in ("lora", "vae", "clip", "controlnet",
                                 "upscale", "ipadapter", "style", "embedding")):
            continue
        if "checkpoint" in ct and ckpt is None:
            ckpt = inp.get("ckpt_name") or inp.get("model_name")
        elif ("unet" in ct or "diffusion" in ct) and diffusion is None:
            diffusion = inp.get("unet_name") or inp.get("model_name") or inp.get("model")
    name = ckpt or diffusion
    if not name or not isinstance(name, str):
        return ""
    # strip directory + extension for a clean display name
    return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]


_PROMPT_MAX = 2000  # cap each prompt — keeps the /jobs list response bounded

# Prompt-specific text input keys — safe to read on any node: a CLIPTextEncode
# (`text`), an SDXL encoder (`text_g`/`text_l`), a String Literal (`string`),
# a wildcard processor (`wildcard`/`populated_text`).
_TEXT_KEYS = ("text", "text_g", "text_l", "string", "wildcard", "populated_text")
# Generic value keys (a primitive's `value`, a passthrough `prompt`). Only
# trusted on a node reached by *following a text link* — i.e. a confirmed
# string-provider — never on an arbitrary conditioning node, where a stray
# string-typed `value` would be mistaken for a prompt.
_PROVIDER_KEYS = ("value", "prompt")


def _generators(wf: dict) -> tuple[list, list]:
    """Split the nodes that actually sample/detail into (samplers, detailers).

    A generator must satisfy *both* signals: a sampler/guider/detailer class
    name AND a positive/negative/conditioning input link. The class check
    rejects routing nodes like rgthree's "Context Big" (which carry
    conditioning links but don't sample); the link check rejects KSamplerSelect
    and SamplerCustomAdvanced (sampler-named but bearing no prompt). Lists keep
    workflow order, so samplers[0] is the primary pass."""
    samplers, detailers = [], []
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").lower()
        is_detailer = "detailer" in ct
        if not (is_detailer or "sampler" in ct or "guider" in ct):
            continue
        inp = node.get("inputs", {}) or {}
        if not any(isinstance(inp.get(k), list)
                   for k in ("positive", "negative", "conditioning")):
            continue
        (detailers if is_detailer else samplers).append((nid, node))
    return samplers, detailers


def _extract_prompts(wf: dict) -> list[dict]:
    """Every distinct text prompt a workflow used, each with a label.

    Returns a list of {"label": str, "text": str}. A plain txt2img graph
    yields [Positive, Negative]; a graph with detailer nodes (FaceDetailer,
    DetailerForEach, …) carries its own prompts, so each detailer that uses a
    *different* prompt adds its own labelled section. Ordered primary-sampler
    first; identical prompt text is shown only once.

    ComfyUI's API-format graph wires CONDITIONING chains into a sampler's
    `positive` / `negative` inputs (or a guider's `conditioning`); the text
    lives in a CLIPTextEncode `inputs.text`. We walk each conditioning input
    upstream — following only conditioning-named inputs so a ControlNet/Concat
    node can't leak the other polarity's text — until we hit a text string.
    The text input may itself be *wired* from a string-provider node (a
    "String Literal", a wildcard processor), so a text-link is followed too.
    """

    def _node(ref: Any) -> dict | None:
        # A link is [node_id, output_index]; node_id is a str in API format.
        if not (isinstance(ref, list) and ref):
            return None
        n = wf.get(str(ref[0]))
        return n if isinstance(n, dict) else None

    def _resolve(ref: Any, polarity: str, seen: set,
                 depth: int = 0, via_link: bool = False) -> str:
        node = _node(ref)
        if node is None or depth > 16:
            return ""
        nid = str(ref[0])
        if nid in seen:
            return ""
        seen.add(nid)
        inp = node.get("inputs", {}) or {}
        # A direct string in a text input. The generic value/prompt keys count
        # only when this node was reached by following a text link (via_link)
        # — i.e. it is a confirmed string provider — so a stray string-typed
        # `value` on an unrelated node isn't mistaken for a prompt.
        for key in (_TEXT_KEYS + _PROVIDER_KEYS if via_link else _TEXT_KEYS):
            val = inp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # A text input wired from a provider node — follow that link. Only the
        # fixed text-content keys are followed (none is a polarity-named input),
        # so this branch structurally cannot cross from positive to negative.
        for key in _TEXT_KEYS:
            got = _resolve(inp.get(key), polarity, set(seen), depth + 1, via_link=True)
            if got:
                return got
        # Otherwise walk upstream, but only through conditioning-named inputs so
        # a ControlNetApplyAdvanced (has both `positive` and `negative`) can't
        # cross polarities. `seen` is copied per branch so it guards against
        # path cycles without letting one branch starve a sibling that shares
        # an upstream node (diamond-shaped conditioning graphs).
        for key, val in inp.items():
            if "cond" not in key.lower() and key != polarity:
                continue
            got = _resolve(val, polarity, set(seen), depth + 1)
            if got:
                return got
        return ""

    def _title(node: dict, nid: str) -> str:
        meta = node.get("_meta") or {}
        t = meta.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
        return node.get("class_type") or f"node {nid}"

    def _cap(text: str) -> str:
        return (text or "").strip()[:_PROMPT_MAX]

    # Each generator (sampler / guider / detailer) contributes a positive +
    # negative pair. Dedupe at the *pair* level: a detailer whose whole prompt
    # set matches an earlier generator's is dropped, but a detailer with any
    # distinct prompt keeps its full pair — so a section never ends up orphaned
    # from its other half.
    samplers, detailers = _generators(wf)
    sections: list[dict] = []
    seen_pairs: set = set()
    primary_id = samplers[0][0] if samplers else None
    for nid, node in samplers + detailers:
        inp = node.get("inputs", {}) or {}
        pos = _cap(_resolve(inp.get("positive") or inp.get("conditioning"), "positive", set()))
        neg = _cap(_resolve(inp.get("negative"), "negative", set()))
        if (not pos and not neg) or (pos, neg) in seen_pairs:
            continue
        seen_pairs.add((pos, neg))
        if nid == primary_id:
            p_label, n_label = "Positive", "Negative"
        else:
            title = _title(node, nid)
            p_label, n_label = f"{title} · Positive", f"{title} · Negative"
        if pos:
            sections.append({"label": p_label, "text": pos})
        if neg:
            sections.append({"label": n_label, "text": neg})

    # Fallback: no generator resolved — read CLIPTextEncode nodes in graph
    # order (first = positive, second = negative, rest numbered).
    if not sections:
        seen: set = set()
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            if "cliptextencode" not in (node.get("class_type") or "").lower():
                continue
            t = (node.get("inputs", {}) or {}).get("text")
            if not (isinstance(t, str) and t.strip()):
                continue
            t = _cap(t)
            if t in seen:
                continue
            seen.add(t)
            i = len(sections)
            sections.append({
                "label": "Positive" if i == 0 else "Negative" if i == 1 else f"Prompt {i + 1}",
                "text": t,
            })

    return sections


_PARAM_KEYS = ("steps", "cfg", "sampler_name", "scheduler", "denoise")


def _extract_params(wf: dict) -> dict:
    """Best-effort generation parameters, anchored on the primary sampler.

    KSampler carries steps/cfg/sampler_name/scheduler/denoise inline, so a
    single node yields a coherent set. Custom-sampler (Flux-style) graphs split
    them across BasicScheduler / CFGGuider / KSamplerSelect nodes — so any key
    the primary sampler lacks is then filled from the other sampler/guider/
    scheduler nodes. Detailer nodes are skipped throughout: we want the main
    run's settings, not a face/hand pass."""

    def _scalar(v: Any) -> Any:
        return v if isinstance(v, (int, float, str)) and v != "" else None

    samplers, _ = _generators(wf)
    primary = samplers[0][1] if samplers else None
    # Fill-in candidates for keys the primary node doesn't carry inline.
    cands = [
        node for node in wf.values()
        if isinstance(node, dict)
        and "detailer" not in (node.get("class_type") or "").lower()
        and any(t in (node.get("class_type") or "").lower()
                for t in ("sampler", "guider", "scheduler"))
    ]

    def _value(ref: Any, *keys: str) -> Any:
        """A scalar from `ref` directly, or one hop upstream. Many graphs wire
        steps/cfg/seed from a shared parameter-provider node (an "Input
        Parameters" node, a primitive) rather than typing them on the sampler;
        `keys` are the input names to try on that upstream node. Exactly one
        hop — a value behind a chain of relay/reroute nodes is left
        unresolved rather than chased (best-effort, keeps this bounded)."""
        v = _scalar(ref)
        if v is not None:
            return v
        up = wf.get(str(ref[0])) if isinstance(ref, list) and ref else None
        if isinstance(up, dict):
            ui = up.get("inputs", {}) or {}
            for k in keys:
                v = _scalar(ui.get(k))
                if v is not None:
                    return v
        return None

    params: dict = {}
    # Resolve every key from the primary sampler first (following a one-hop
    # link to a param-provider node) so the set stays coherent; only keys the
    # primary genuinely lacks fall through to the other sampler nodes — the
    # Flux custom-sampler case, where the params live in separate nodes.
    for key in _PARAM_KEYS:
        alts = (key, "sampler", "value") if key == "sampler_name" else (key, "value")
        if primary is not None:
            v = _value((primary.get("inputs", {}) or {}).get(key), *alts)
            if v is not None:
                params[key] = v
                continue
        for node in cands:
            v = _value((node.get("inputs", {}) or {}).get(key), *alts)
            if v is not None:
                params[key] = v
                break

    # Seed — primary first, then any sampler node; it is often wired from a
    # Seed / primitive node.
    for node in ([primary] if primary else []) + cands:
        inp = node.get("inputs", {}) or {}
        for sk in ("seed", "noise_seed"):
            v = _value(inp.get(sk), "seed", "noise_seed", "value")
            if v is not None:
                params["seed"] = v
                break
        if "seed" in params:
            break

    return params


def _query_newest(status: str, n: int) -> list:
    """The `n` newest job records in `status`, via the status-index GSI.

    A single DynamoDB Query returns only one ~1 MB page — and because each job
    record carries the full workflow_json, that page can hold far fewer than
    `n` items (≈50). This is the bug that hid every generation past the newest
    ~50: the query was issued once and never followed past the first page. So
    here we page through LastEvaluatedKey until we have `n` items or the
    status is exhausted."""
    items: list = []
    kwargs: dict = {}
    while len(items) < n:
        r = ddb.query(
            TableName=JOBS_TABLE,
            IndexName="status-index",
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": {"S": status}},
            ScanIndexForward=False,  # newest (highest created_at) first
            Limit=n - len(items),
            **kwargs,
        )
        items.extend(r.get("Items", []))
        lek = r.get("LastEvaluatedKey")
        if not lek:
            break
        kwargs["ExclusiveStartKey"] = lek
    return items[:n]


def _list_jobs(event: dict) -> dict:
    """GET /jobs?status=complete,failed,cancelled&limit=64&offset=0

    Queries the status-index GSI (paginated — see _query_newest) once per
    requested status, merges, sorts by submission time (created_at)
    newest-first, then applies offset/limit in Python.
    """
    qs = event.get("queryStringParameters") or {}
    statuses = [s.strip() for s in (qs.get("status") or "").split(",") if s.strip()]
    if not statuses:
        # Match the values the worker actually writes (see worker.py
        # _set_job_status / _claim_job): complete / failed / cancelled /
        # running / queued.
        statuses = ["complete", "failed", "cancelled", "running", "queued"]

    try:
        limit = max(1, min(int(qs.get("limit") or 64), 500))
    except ValueError:
        limit = 64
    try:
        offset = max(0, int(qs.get("offset") or 0))
    except ValueError:
        offset = 0

    # Each status contributes its newest (offset+limit) records by created_at
    # (the GSI sort key). Since the merge below also sorts by created_at, the
    # fetch order and sort order match, so the top (offset+limit) of the merge
    # is exactly the global newest however the statuses interleave.
    items: list[dict] = []
    for status in statuses:
        items.extend(_query_newest(status, offset + limit))

    # Newest first by submission time (created_at).
    items.sort(key=lambda it: it.get("created_at", {}).get("S", ""), reverse=True)
    page = items[offset : offset + limit]

    jobs = [_serialize_job(it) for it in page]
    # `total` is the size of the fetched candidate pool (capped per status at
    # offset+limit), not a true count of all matching jobs. Fine for the
    # current callers — none of them page on it.
    return _resp(200, {"jobs": jobs, "limit": limit, "offset": offset, "total": len(items)})


def _serialize_job(it: dict) -> dict:
    """Shape a raw DDB job item into the API job object. `model`, the prompt
    sections and the generation params are all derived on the fly from the
    stored `workflow_json` (the job record itself never carries them)."""
    wf = _parse_workflow(it.get("workflow_json", {}).get("S", ""))
    return {
        "job_id": it["job_id"]["S"],
        "type": it.get("type", {"S": ""})["S"],
        "status": it.get("status", {"S": ""})["S"],
        "created_at": it.get("created_at", {"S": ""})["S"],
        "started_at": it.get("started_at", {"S": ""})["S"],
        "completed_at": it.get("completed_at", {"S": ""})["S"],
        "output_keys": json.loads(it.get("output_keys", {"S": "[]"})["S"]),
        "error": it.get("error", {"S": ""})["S"],
        "model": _extract_model(wf),
        # All distinct prompts (txt2img → Positive/Negative; detailer graphs
        # add their own sections) + best-effort generation params.
        "prompts": _extract_prompts(wf),
        "params": _extract_params(wf),
        "progress": it.get("progress", {}).get("S", ""),
        # The GPU instance type the worker recorded when it claimed the job.
        # Both fleets run mixed instance types (a spot-capacity ASG), so this
        # is only accurate from the worker — blank on jobs claimed before the
        # worker started reporting it.
        "instance_type": it.get("instance_type", {}).get("S", ""),
        # True once a cancel has been requested on a still-running job — lets
        # the viewer keep the row in a "cancelling" state until the worker
        # actually stops it (a running cancel is not instantaneous).
        "cancel_requested": it.get("cancel_requested", {}).get("BOOL", False),
    }


def _get_job(event: dict) -> dict:
    job_id = event["pathParameters"]["id"]
    r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    if "Item" not in r:
        return _resp(404, {"error": "job not found"})

    item = r["Item"]
    output = _serialize_job(item)
    output["attempt_count"] = int(item.get("attempt_count", {"N": "0"})["N"])
    # Raw workflow graph — the viewer's "copy workflow" button needs the full
    # node dict. Deliberately not in the /jobs list response (too large ×N);
    # callers fetch it per-job here.
    output["workflow"] = _parse_workflow(item.get("workflow_json", {}).get("S", ""))
    return _resp(200, output)


def _delete_job(event: dict) -> dict:
    """DELETE /jobs/{id} — delete a generation: its output objects from the
    outputs bucket, then the job record from DynamoDB so it drops out of the
    viewer. Used by the viewer's per-tile delete button."""
    job_id = event["pathParameters"]["id"]
    r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    if "Item" not in r:
        return _resp(404, {"error": "job not found"})
    keys = json.loads(r["Item"].get("output_keys", {"S": "[]"})["S"])
    deleted = 0
    for key in keys:
        try:
            s3.delete_object(Bucket=OUTPUTS_BUCKET, Key=key)
            deleted += 1
        except ClientError as e:  # noqa: PERF203
            print(f"delete object {key} failed: {e!r}")
    ddb.delete_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    return _resp(200, {"deleted": job_id, "objects": deleted})


def _cancel_job(event: dict) -> dict:
    """POST /jobs/{id}/cancel — cancel a queued or running job.

    Two paths, because the API Lambda has no inbound channel to a busy worker:

      queued  — the job hasn't been picked up. Flip status → "cancelled" so it
                drops out of the viewer's pending strip immediately; the SQS
                message lingers but the worker discards it on dequeue (it only
                runs jobs still in "queued"). The status write is conditional
                on the job still being "queued" so we don't clobber a worker
                that grabbed it microseconds ago — if that race is lost we
                fall through to the running path.

      running — a worker is mid-sampling. We can't reach it, so we set a
                `cancel_requested` flag; the worker polls it, calls ComfyUI's
                /interrupt, and writes the final "cancelled" status itself.

    Terminal jobs (complete / failed / already cancelled) → 409.
    """
    job_id = event["pathParameters"]["id"]
    r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    if "Item" not in r:
        return _resp(404, {"error": "job not found"})
    status = r["Item"].get("status", {"S": ""})["S"]

    if status == "queued":
        try:
            ddb.update_item(
                TableName=JOBS_TABLE,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET #s = :cancelled, cancelled_at = :t",
                ConditionExpression="#s = :queued",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":cancelled": {"S": "cancelled"},
                    ":queued": {"S": "queued"},
                    ":t": {"S": datetime.now(timezone.utc).isoformat()},
                },
            )
            return _resp(200, {"job_id": job_id, "state": "cancelled"})
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
            # A worker claimed the job between our read and write — re-read and
            # fall through to the running path below.
            r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
            status = r.get("Item", {}).get("status", {"S": ""})["S"]

    if status == "running":
        ddb.update_item(
            TableName=JOBS_TABLE,
            Key={"job_id": {"S": job_id}},
            UpdateExpression="SET cancel_requested = :true",
            ExpressionAttributeValues={":true": {"BOOL": True}},
        )
        return _resp(200, {"job_id": job_id, "state": "cancelling"})

    return _resp(409, {"error": f"job is {status or 'unknown'}, cannot cancel"})


def _view(event: dict) -> dict:
    """Return a presigned URL for an output S3 key.

    Default: 302 redirect — an <img>/<video> whose src points here follows it
    straight to S3.
    With ?json=1: 200 + {"url": ...}. A cross-origin fetch() cannot read a
    302's Location (opaque-redirect responses hide headers, and the 302
    carries no CORS header), so JS callers that fetch this — e.g. the
    standalone viewer page — must use the JSON form and set the <img> src
    to the returned URL themselves.
    """
    qs = event.get("queryStringParameters") or {}
    key = qs.get("key")
    if not key:
        return _resp(400, {"error": "missing key parameter"})

    # SECURITY: only allow keys under outputs bucket. The key is opaque to us
    # but we don't expose models/uploads buckets through this path.
    try:
        url = s3.generate_presigned_url(
            "get_object",
            # ResponseCacheControl makes S3 return a real Cache-Control on the
            # image response (the objects themselves carry none). Output PNGs
            # are immutable — a generation's file never changes — so the
            # browser can cache hard and never revalidate within the URL's
            # life. Without this the browser only heuristic-caches, which is
            # near-useless for freshly-generated images.
            Params={
                "Bucket": OUTPUTS_BUCKET,
                "Key": key,
                "ResponseCacheControl": "public, max-age=31536000, immutable",
            },
            ExpiresIn=PRESIGNED_GET_TTL,
        )
    except ClientError as e:
        print(f"presign error: {e}")
        return _resp(500, {})
    if qs.get("json") in ("1", "true"):
        return _resp(200, {"url": url})
    return {
        "statusCode": 302,
        "headers": {"Location": url, "Cache-Control": "no-store"},
        "body": "",
    }


def _upload(event: dict) -> dict:
    """Return a presigned PUT URL for the browser to upload directly to S3.
    Avoids passing image bytes through Lambda (6 MB limit)."""
    body = json.loads(event.get("body") or "{}")
    content_type = body.get("content_type", "image/png")
    upload_id = str(uuid.uuid4())
    key = f"uploads/{datetime.now(timezone.utc):%Y/%m/%d}/{upload_id}"

    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": UPLOADS_BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=PRESIGNED_PUT_TTL,
    )
    return _resp(200, {"upload_url": url, "key": key, "bucket": UPLOADS_BUCKET})


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
