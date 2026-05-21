"""
Main worker loop.

Lifecycle:
  1. Boot: warm pinned models, start ComfyUI subprocess, publish object_info
  2. Loop: poll SQS, run job, upload outputs, update DDB
  3. Shutdown on spot interruption: release in-flight job to queue, exit clean
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

# Model files appear under /opt/comfy/models/<type>/ via mount-s3 (see
# workers/image/entrypoint.sh). No download path needed here.
from comfy_client import ComfyClient
from comfy_supervisor import ComfySupervisor
from object_info_publisher import publish_object_info
from output_uploader import OutputUploader
from spot_handler import SpotHandler, make_default_on_terminate
from ws_bridge import WsBridge

# extensions_publisher is optional — older container images don't ship it.
# Make import non-fatal so the worker doesn't crash on an old image during
# rolling deploys.
try:
    from extensions_publisher import publish_extensions
except ImportError:
    publish_extensions = None  # type: ignore

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Silence chatty libs (resolves v3 N1 — log spend control)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

# ---------- env ----------
FLEET = os.environ["FLEET"]
QUEUE_URL = os.environ["QUEUE_URL"]
MODELS_BUCKET = os.environ["MODELS_BUCKET"]
OUTPUTS_BUCKET = os.environ["OUTPUTS_BUCKET"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
MODELS_TABLE = os.environ["MODELS_TABLE"]
REGION = os.environ["AWS_REGION"]
VISIBILITY_TIMEOUT_SEC = int(os.environ.get("VISIBILITY_TIMEOUT_SECONDS", "900"))
HEARTBEAT_INTERVAL_SEC = 60
COMFY_OUTPUT_DIR = Path("/opt/comfy/output")

sqs = boto3.client("sqs", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)


def main() -> int:
    log.info("worker starting: fleet=%s, queue=%s", FLEET, QUEUE_URL)

    # Install custom nodes from the S3 manifest BEFORE starting ComfyUI so
    # they register on the first /object_info publish. This is what makes
    # Manager-installed nodes actually work on GPU workers (otherwise the
    # node only lives on the metadata container).
    try:
        from manifest_installer import sync as sync_custom_nodes
        results = sync_custom_nodes()
        for name, status in results.items():
            log.info("custom node %s: %s", name, status)
    except Exception:  # noqa: BLE001
        log.exception("manifest sync failed (non-fatal; continuing without custom nodes)")

    # NOTE: model files appear under /opt/comfy/models/<type>/ via mount-s3,
    # mounted in workers/image/entrypoint.sh. No download-on-demand needed here.
    # Pinned-model warmup runs in background, also kicked off by the entrypoint.

    # ComfyUI subprocess.
    extra_args = ["--use-sage-attention"] if FLEET == "video" else []
    comfy = ComfySupervisor(extra_args=tuple(extra_args))

    # Spot handler: wires DDB+SQS reset on termination notice.
    # The handler now receives (job_id, receipt_handle) tuple so it can reset
    # SQS visibility, not just update DDB (resolves code review C4).
    on_terminate = make_default_on_terminate(QUEUE_URL, JOBS_TABLE, REGION)

    def _terminate_chain(in_flight):
        on_terminate(in_flight)
        comfy.stop(timeout=10)
        sys.exit(0)

    spot = SpotHandler(on_terminate=_terminate_chain)
    spot.start()

    # SIGTERM (from ECS draining or manual stop) — graceful shutdown
    def _on_sigterm(_signo, _frame):
        log.warning("received SIGTERM; draining")
        spot.terminating = True
        comfy.stop(timeout=30)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    # Start ComfyUI and wait for ready.
    comfy.start()
    comfy.wait_for_ready(timeout_seconds=180)

    # ComfyUI HTTP client — used for object_info AND for submitting every job
    # below, so it must always be created (not just when publishing).
    client = ComfyClient(comfy.base_url)

    # Publish /object_info + JS extensions to the frontend — ONLY for fleets
    # without a dedicated metadata instance. The image fleet has the always-on
    # metadata instance as the canonical publisher: it runs full-time with
    # every custom node loaded. A GPU image worker booting would overwrite
    # that complete node list with its own — often partial, because on a cold
    # worker custom nodes can still be importing when wait_for_ready returns —
    # which makes the editor lose nodes. So image workers skip publishing
    # (the worker still *installs* every manifest node, so it can run jobs);
    # video workers (no metadata box) keep publishing.
    if FLEET != "image":
        try:
            oi = client.fetch_object_info()
            publish_object_info(FLEET, oi)
        except Exception:  # noqa: BLE001
            log.exception("object_info publish failed (non-fatal)")
        if publish_extensions is not None:
            try:
                publish_extensions(FLEET)
            except Exception:  # noqa: BLE001
                log.exception("extensions publish failed (non-fatal)")

    uploader = OutputUploader(OUTPUTS_BUCKET, REGION, COMFY_OUTPUT_DIR)

    log.info("worker ready, entering SQS poll loop")
    while not spot.terminating:
        msgs = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            WaitTimeSeconds=20,
            MaxNumberOfMessages=1,
            VisibilityTimeout=VISIBILITY_TIMEOUT_SEC,
            MessageAttributeNames=["All"],
        )
        messages = msgs.get("Messages", [])
        if not messages:
            continue

        msg = messages[0]
        receipt = msg["ReceiptHandle"]
        body = json.loads(msg["Body"])
        job_id = body["job_id"]
        log.info("received job %s", job_id)

        # Idempotency: skip if another worker already running it (v3 N9)
        job_record = _read_job(job_id)
        if not job_record:
            log.warning("job %s not in DDB; deleting message", job_id)
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt)
            continue
        if job_record.get("status") == "running":
            log.warning("job %s already running on another worker; deleting duplicate delivery", job_id)
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt)
            continue

        spot.set_in_flight(job_id, receipt)
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(receipt, job_id, heartbeat_stop),
            daemon=True,
        )
        heartbeat_thread.start()

        success = False
        try:
            workflow_str = job_record["workflow_json"]
            workflow = json.loads(workflow_str)

            # Mark running BEFORE starting work so a duplicate delivery skips.
            _set_job_status(job_id, "running",
                            extra={"started_at": datetime.now(timezone.utc).isoformat()})

            # Background-fetch every referenced model so mount-s3 starts
            # streaming it from S3 while ComfyUI is still parsing the workflow.
            # Without this, the first read of an uncached model blocks the
            # whole sampling pipeline on the S3 download. cat-to-/dev/null is
            # the same trick warm_pinned uses; threading just parallelizes the
            # I/O for multi-LoRA workflows.
            _prefetch_referenced_models(workflow)

            # Snapshot output dir BEFORE starting the job so we can scan for
            # new files after (handles custom nodes that bypass standard outputs).
            since = uploader.snapshot_marker()

            # Use the browser-supplied client_id so events route back to the
            # right WS connection. Fallback to a synthetic id if missing
            # (older clients that didn't send one — events get dropped at
            # the forward Lambda since no WS connection matches).
            ws_client_id = job_record.get("client_id") or f"worker-{FLEET}-{job_id[:8]}"
            # Pass job_id so the bridge also records sampling progress to the
            # job's DDB record — drives the viewer's live progress bar.
            ws_bridge = WsBridge(client_id=ws_client_id, job_id=job_id)
            ws_bridge.start()

            try:
                # Submit + wait
                prompt_id = client.submit_prompt(workflow, client_id=ws_client_id)
                client.wait_for_completion(prompt_id, timeout_seconds=VISIBILITY_TIMEOUT_SEC - 60)
            finally:
                ws_bridge.stop()

            # Upload anything new from the output dir.
            output_keys = uploader.upload_new_outputs(job_id, since)
            uploader.cleanup_outputs(since)

            _set_job_status(
                job_id,
                "complete",
                extra={
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "output_keys": json.dumps(output_keys),
                },
            )
            success = True
            log.info("job %s complete: %d outputs", job_id, len(output_keys))

        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            _set_job_status(job_id, "failed", extra={"error": str(e)[:500]})

        finally:
            heartbeat_stop.set()
            spot.clear_in_flight()
            if success:
                # Successful jobs delete from queue so they don't redeliver.
                try:
                    sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt)
                except Exception:  # noqa: BLE001
                    log.exception("failed to delete sqs message for %s", job_id)
            else:
                # Failed jobs: delete to avoid retry storms (v3 N9). They're in DDB
                # for the user to inspect. SQS DLQ catches accidental retries.
                try:
                    sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt)
                except Exception:  # noqa: BLE001
                    pass

    log.info("worker exiting cleanly")
    return 0


def _read_job(job_id: str) -> dict | None:
    r = ddb.get_item(TableName=JOBS_TABLE, Key={"job_id": {"S": job_id}})
    item = r.get("Item")
    if not item:
        return None
    return {
        "status": item.get("status", {"S": ""})["S"],
        "type": item.get("type", {"S": ""})["S"],
        "workflow_json": item.get("workflow_json", {"S": "{}"})["S"],
        "client_id": item.get("client_id", {"S": ""})["S"],
    }


def _set_job_status(job_id: str, status: str, extra: dict | None = None) -> None:
    expr_names = {"#s": "status"}
    expr_values = {":s": {"S": status}}
    sets = ["#s = :s"]
    if extra:
        for i, (k, v) in enumerate(extra.items()):
            expr_names[f"#k{i}"] = k
            expr_values[f":v{i}"] = {"S": str(v)}
            sets.append(f"#k{i} = :v{i}")

    ddb.update_item(
        TableName=JOBS_TABLE,
        Key={"job_id": {"S": job_id}},
        UpdateExpression="SET " + ", ".join(sets),
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )


def _heartbeat_loop(receipt_handle: str, job_id: str, stop_event: threading.Event) -> None:
    """Extend SQS visibility every 60s. v3 N9: prevents duplicate delivery on long jobs."""
    while not stop_event.wait(HEARTBEAT_INTERVAL_SEC):
        try:
            sqs.change_message_visibility(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=VISIBILITY_TIMEOUT_SEC,
            )
            ddb.update_item(
                TableName=JOBS_TABLE,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET last_heartbeat = :h",
                ExpressionAttributeValues={
                    ":h": {"S": datetime.now(timezone.utc).isoformat()}
                },
            )
        except Exception:  # noqa: BLE001
            log.exception("heartbeat failed for %s", job_id)


_MODEL_INPUT_KEYS = {
    "ckpt_name", "lora_name", "vae_name", "control_net_name", "controlnet_name",
    "clip_name", "clip_name1", "clip_name2", "clip_name3", "clip_vision_name",
    "text_encoder_name", "embedding_name", "unet_name", "diffusion_model_name",
    "style_model_name", "gligen_name", "hypernetwork_name", "photomaker_name",
    "upscale_model_name", "model_name",
}
_MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".gguf", ".pt", ".bin", ".pth")


def _extract_model_filenames(workflow: dict) -> set[str]:
    """Return the set of model filenames the workflow references. We pick up
    values whose input key matches a model-input name (ckpt_name, lora_name,
    ...) — those carry the on-disk filename verbatim, matching what mount-s3
    surfaces under /opt/comfy/models/<type>/."""
    found: set[str] = set()
    for _node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        for k, v in node.get("inputs", {}).items():
            if not isinstance(v, str) or not v:
                continue
            if k in _MODEL_INPUT_KEYS or v.lower().endswith(_MODEL_EXTENSIONS):
                found.add(v)
    return found


_COMFY_MODELS_ROOT = Path("/opt/comfy/models")


def _prefetch_referenced_models(workflow: dict) -> None:
    """Kick off background reads of every referenced model file so mount-s3
    starts streaming from S3 in parallel with ComfyUI's workflow parsing.
    The cat-to-/dev/null trick is the same one warm_pinned uses to seed the
    mount-s3 disk cache. Best-effort: anything that doesn't resolve to a
    real file (custom-node input that isn't actually a model) is skipped."""
    import threading

    filenames = _extract_model_filenames(workflow)
    if not filenames:
        return

    def _stream(path: str) -> None:
        try:
            with open(path, "rb") as f:
                while f.read(16 * 1024 * 1024):
                    pass
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("prefetch failed for %s", path)

    started = 0
    for filename in filenames:
        # Catalog type is unknown without a DDB lookup, so check each models/*/
        # dir. Cheap — mount-s3 caches readdir results, and we stop at first hit.
        for type_dir in _COMFY_MODELS_ROOT.iterdir() if _COMFY_MODELS_ROOT.is_dir() else []:
            if not type_dir.is_dir():
                continue
            candidate = type_dir / filename
            if candidate.exists():
                threading.Thread(target=_stream, args=(str(candidate),), daemon=True).start()
                started += 1
                break
    if started:
        log.info("prefetching %d referenced models in background", started)


if __name__ == "__main__":
    sys.exit(main())
