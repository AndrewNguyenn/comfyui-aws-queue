#!/usr/bin/env python3
"""Rebuild comfy-jobs records from the images left in the outputs bucket.

WHY THIS EXISTS
---------------
The dispatcher used to stamp every job record with a 30-day `expire_at`, and
the comfy-jobs table had TTL enabled on it. The viewer's gallery is driven by
that table (GET /jobs?status=complete), so the user's *visible* history was
capped at 30 days while every generated image sat retained in S3 forever. By
the time it was noticed (2026-08-19) ~18.5k job rows had been reaped against
18,798 surviving PNGs. The TTL is gone now (see the dispatcher + storage.ts);
this script recovers the rows that were already lost.

HOW RECOVERY IS POSSIBLE
------------------------
ComfyUI embeds the full API-format prompt graph in every PNG it saves, as a
`prompt` tEXt chunk (and the UI graph as a `workflow` chunk). That is byte-for
-byte the same thing the dispatcher stored in `workflow_json` / `workflow_ui`.
Combined with the `outputs/YYYY/MM/DD/<job_id>/<file>` key layout, every field
the viewer needs can be reconstructed from S3 alone:

    job_id       <- key path component
    output_keys  <- every object under that job's prefix
    created_at   <- S3 LastModified of the job's earliest output
    completed_at <- S3 LastModified of the job's latest output
    workflow_json<- PNG `prompt` tEXt chunk
    workflow_ui  <- PNG `workflow` tEXt chunk
    model        \
    character     >- derived from the graph via services/dispatcher/extract.py,
    subject      /   the same module the dispatcher denormalizes with, so the
                     values match what a live job would have written.

Only the PNG *header* is read (a ranged GET), not the whole image — the tEXt
chunks sit before IDAT.

Reconstructed rows carry `backfilled: true` so they are distinguishable from
rows written by a real dispatch.

USAGE
    ./scripts/backfill-job-history.py                      # dry run (default)
    ./scripts/backfill-job-history.py --limit 20 --apply   # small live sample
    ./scripts/backfill-job-history.py --apply              # full run

Idempotent: job_ids already present in the table are skipped, so a partial or
interrupted run can simply be re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "services", "dispatcher")
)
import extract  # noqa: E402  (services/dispatcher/extract.py)

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = os.environ.get("COMFY_JOBS_TABLE", "comfy-jobs")


def _outputs_bucket() -> str:
    """comfy-outputs-<account>-<region>, with the account resolved from STS.

    Matches scripts/backfill-job-attrs.sh — no account id baked into the repo.
    """
    override = os.environ.get("COMFY_OUTPUTS_BUCKET")
    if override:
        return override
    acct = os.environ.get("COMFY_ACCOUNT_ID") or (
        boto3.client("sts", region_name=REGION).get_caller_identity()["Account"])
    return f"comfy-outputs-{acct}-{REGION}"


BUCKET = ""  # resolved in main() so --help never needs credentials

# The tEXt chunks live before IDAT, so the header alone carries the graphs.
# 256 KB covers every graph seen in this deployment; if a chunk is truncated we
# re-fetch the object in full rather than store a broken graph.
HEAD_BYTES = 256 * 1024
# DynamoDB caps an item at 400 KB. The dispatcher guards workflow_ui at 350 KB;
# hold the *combined* graphs under the same ceiling, dropping the UI graph first
# (it is a convenience for "Copy JSON", where workflow_json drives the viewer).
ITEM_GUARD = 350_000

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".gif", ".mp4", ".webm", ".mkv")

# Why a job produced no record — reported as a breakdown so "N unrecoverable"
# is actionable rather than opaque.
SKIP_NO_MEDIA = "no media outputs"
SKIP_NO_GRAPH = "no prompt chunk in any candidate"
SKIP_BAD_GRAPH = "prompt chunk did not parse"
SKIP_TOO_BIG = "item over the DynamoDB size guard"
SKIP_S3_ERROR = "S3 header read failed"

_boto_cfg = Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=64)
_local = threading.local()


def _s3():
    if not hasattr(_local, "s3"):
        _local.s3 = boto3.client("s3", region_name=REGION, config=_boto_cfg)
    return _local.s3


def _png_text_chunks(data: bytes) -> tuple[dict[str, str], bool]:
    """Parse PNG tEXt/iTXt chunks out of `data`.

    Returns (chunks, complete) — `complete` is False when the buffer ran out
    mid-chunk before IDAT, meaning the caller should re-fetch more bytes.
    """
    chunks: dict[str, str] = {}
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return chunks, True  # not a PNG (jpg/webp) — nothing to parse, don't refetch
    i = 8
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        ctype = data[i + 4 : i + 8]
        if ctype == b"IDAT":
            return chunks, True  # past the metadata; whatever we have is all there is
        end = i + 8 + length
        if end > len(data):
            return chunks, False  # truncated mid-chunk — need a bigger read
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            raw = data[i + 8 : end]
            keyword, _, rest = raw.partition(b"\x00")
            key = keyword.decode("latin-1", "replace")
            if ctype == b"iTXt":
                # iTXt layout: keyword \0 compFlag compMethod lang \0 translated \0 text
                # compFlag/compMethod are raw bytes, NOT delimiters — splitting
                # on NUL without dropping them prefixes the text with junk and
                # the graph fails to parse. PIL's PngInfo.add_text silently
                # falls back to iTXt for any non-latin-1 value, so this path is
                # reachable from any node that writes the prompt with
                # ensure_ascii=False.
                if rest[:1] == b"\x01":
                    i = end + 4
                    continue  # zlib-compressed iTXt — skip, don't store garbage
                body = rest[2:]  # drop compFlag + compMethod
                _lang, _, r2 = body.partition(b"\x00")
                _translated, _, text = r2.partition(b"\x00")
                chunks[key] = text.decode("utf-8", "replace")
            elif ctype == b"zTXt":
                # zTXt: keyword \0 compMethod <zlib data>
                try:
                    chunks[key] = zlib.decompress(rest[1:]).decode("utf-8", "replace")
                except Exception:  # noqa: BLE001 — corrupt/unknown compression
                    pass
            else:
                chunks[key] = rest.decode("utf-8", "replace")
        i = end + 4  # skip the CRC
    return chunks, False


def _fetch_graphs(key: str) -> dict[str, str]:
    """Ranged-GET a PNG header and return its embedded graphs."""
    s3 = _s3()
    obj = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes=0-{HEAD_BYTES - 1}")
    data = obj["Body"].read()
    chunks, complete = _png_text_chunks(data)
    if not complete:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        chunks, _ = _png_text_chunks(data)
    return chunks


def _list_jobs_in_s3() -> dict[str, dict]:
    """Group every object under outputs/ into {job_id: {keys, first, last}}."""
    jobs: dict[str, dict] = defaultdict(lambda: {"keys": [], "first": None, "last": None})
    paginator = _s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="outputs/"):
        for o in page.get("Contents", []):
            parts = o["Key"].split("/")
            # outputs/YYYY/MM/DD/<job_id>/<file...>
            if len(parts) < 6 or parts[0] != "outputs":
                continue
            job_id = parts[4]
            j = jobs[job_id]
            j["keys"].append(o["Key"])
            lm = o["LastModified"]
            if j["first"] is None or lm < j["first"]:
                j["first"] = lm
            if j["last"] is None or lm > j["last"]:
                j["last"] = lm
    return jobs


def _existing_job_ids(ddb) -> set[str]:
    """Every job_id already in the table (so the backfill stays idempotent).

    Queries the jobs-by-status GSI rather than scanning the base table.
    ProjectionExpression limits what is RETURNED, not what is BILLED — DynamoDB
    charges on the stored item size, so a base-table scan pays for every row's
    ~13 KB workflow_json. The GSI's INCLUDE projection excludes that blob, so
    the same answer costs ~1 KB/row (this is the same read-cost reasoning that
    drove the ALL->INCLUDE reproject; see infra/lib/stacks/storage.ts).
    """
    ids: set[str] = set()
    paginator = ddb.get_paginator("query")
    for status in ("complete", "failed", "cancelled", "running", "queued"):
        for page in paginator.paginate(
            TableName=TABLE,
            IndexName="jobs-by-status",
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": {"S": status}},
            ProjectionExpression="job_id",
        ):
            ids.update(i["job_id"]["S"] for i in page.get("Items", []))
    return ids


def _item_bytes(item: dict) -> int:
    """Approximate the DynamoDB item size — attribute names plus UTF-8 values.

    len() on a str counts characters; DynamoDB bills UTF-8 bytes. That is the
    same today (ComfyUI escapes to ASCII) but stops being true for any graph
    written with ensure_ascii=False, which the iTXt path above now recovers.
    """
    total = 0
    for k, v in item.items():
        total += len(k.encode())
        val = next(iter(v.values()))
        total += len(val.encode()) if isinstance(val, str) else 8
    return total


def _build_item(job_id: str, info: dict) -> tuple[dict | None, str]:
    """Reconstruct one job record. Returns (item, skip_reason)."""
    keys = sorted(info["keys"])
    images = [k for k in keys if k.lower().endswith(IMAGE_EXTS)]
    videos = [k for k in keys if k.lower().endswith(VIDEO_EXTS)]
    if not images and not videos:
        return None, SKIP_NO_MEDIA

    # Only PNGs carry the embedded graph, and not every PNG does (a node may
    # save without metadata). Probe PNGs first, then the rest, until one yields
    # a prompt chunk — taking only images[0] drops a job whose first key
    # happens to be a .webp or a metadata-less save.
    candidates = ([k for k in images if k.lower().endswith(".png")] +
                  [k for k in images if not k.lower().endswith(".png")])
    prompt_json = ""
    graphs: dict[str, str] = {}
    saw_s3_error = False
    for cand in candidates:
        try:
            graphs = _fetch_graphs(cand)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {job_id}: header read failed on {cand}: {e}", file=sys.stderr)
            saw_s3_error = True
            continue
        if graphs.get("prompt"):
            prompt_json = graphs["prompt"]
            break
    if not prompt_json:
        return None, (SKIP_S3_ERROR if saw_s3_error else SKIP_NO_GRAPH)
    wf = extract._parse_workflow(prompt_json)
    if not wf:
        return None, SKIP_BAD_GRAPH

    item = {
        "job_id": {"S": job_id},
        "type": {"S": "video" if videos and not images else "image"},
        "status": {"S": "complete"},
        "created_at": {"S": info["first"].isoformat()},
        "completed_at": {"S": info["last"].isoformat()},
        "output_keys": {"S": json.dumps(keys)},
        "workflow_json": {"S": prompt_json},
        "attempt_count": {"N": "0"},
        # Marks this row as reconstructed from S3 rather than written at dispatch.
        "backfilled": {"BOOL": True},
    }

    # Denormalized attrs — the gallery list reads these off the GSI, which does
    # not project workflow_json, so without them every recovered row shows a
    # blank model label. Derived with the dispatcher's own module.
    model = extract._extract_model(wf)
    character, subject = extract._extract_subject(wf)
    if model:
        item["model"] = {"S": model}
    if character:
        item["character"] = {"S": character}
    if subject:
        item["subject"] = {"S": subject}

    ui_json = graphs.get("workflow", "")
    if ui_json and _item_bytes(item) + len(ui_json.encode()) <= ITEM_GUARD:
        item["workflow_ui"] = {"S": ui_json}
    # The UI graph is optional, but workflow_json alone can exceed the cap on a
    # pathological graph. DynamoDB rejects an oversized item with a
    # ValidationException against the WHOLE BatchWriteItem call, so screen it
    # here rather than letting one row take a 25-item batch down.
    size = _item_bytes(item)
    if size > ITEM_GUARD:
        print(f"  ! {job_id}: item {size} B over the {ITEM_GUARD} B guard — skipping",
              file=sys.stderr)
        return None, SKIP_TOO_BIG
    return item, ""


def _write_batches(ddb, items: list[dict]) -> tuple[int, int]:
    """BatchWriteItem in chunks of 25. Returns (written, failed).

    BatchWriteItem returns HTTP 200 with an UnprocessedItems list when it
    throttles, so botocore's retry layer never sees a failure — the backoff has
    to live here. Only items DynamoDB actually accepted are counted; a batch
    that never drains is reported, not silently tallied as written.
    """
    written = 0
    failed = 0
    for i in range(0, len(items), 25):
        batch = items[i : i + 25]
        remaining = [{"PutRequest": {"Item": it}} for it in batch]
        try:
            for attempt in range(8):
                r = ddb.batch_write_item(RequestItems={TABLE: remaining})
                remaining = (r.get("UnprocessedItems") or {}).get(TABLE) or []
                if not remaining:
                    break
                time.sleep(0.05 * (2 ** attempt))
        except Exception as e:  # noqa: BLE001
            # A malformed/oversized item raises ValidationException for the
            # WHOLE call. Don't let one bad batch abort the remaining writes.
            print(f"  ! batch at offset {i} failed: {e}", file=sys.stderr)
            failed += len(batch)
            continue
        written += len(batch) - len(remaining)
        if remaining:
            failed += len(remaining)
            print(f"  ! {len(remaining)} items still unprocessed after 8 attempts",
                  file=sys.stderr)
    return written, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Dry run by default, matching scripts/backfill-job-attrs.sh — a bare
    # invocation of a bulk-write script must never write.
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process N missing jobs (the NEWEST N, so a sample is easy to eyeball)")
    ap.add_argument("--workers", type=int, default=32, help="concurrent S3 header reads")
    args = ap.parse_args()

    global BUCKET
    BUCKET = _outputs_bucket()
    ddb = boto3.client("dynamodb", region_name=REGION, config=_boto_cfg)

    print(f"bucket={BUCKET} table={TABLE} region={REGION}")
    print(f"mode={'APPLY (writes)' if args.apply else 'DRY RUN (no writes)'}")
    print("listing outputs/ ...")
    jobs = _list_jobs_in_s3()
    print(f"  {len(jobs)} job folders, {sum(len(j['keys']) for j in jobs.values())} objects")

    print("reading existing job ids ...")
    existing = _existing_job_ids(ddb)
    print(f"  {len(existing)} already in {TABLE}")

    missing = sorted(set(jobs) - existing, key=lambda j: jobs[j]["first"])
    print(f"  {len(missing)} missing -> to reconstruct")
    if args.limit:
        # Newest N: the oldest rows sit at the bottom of a 19k-row gallery and
        # are the hardest possible sample to verify by eye.
        missing = missing[-args.limit:]
        print(f"  limited to the newest {len(missing)}")
    if not missing:
        print("nothing to do")
        return 0

    built: list[dict] = []
    skips: Counter = Counter()
    done = 0
    lock = threading.Lock()

    def work(job_id: str):
        nonlocal done
        try:
            it, reason = _build_item(job_id, jobs[job_id])
        except Exception as e:  # noqa: BLE001
            # One unexpected throw must not kill all `workers` threads —
            # ex.map re-raises out of the pool and would abort the whole run.
            print(f"  ! {job_id}: unexpected error: {e}", file=sys.stderr)
            it, reason = None, "unexpected error"
        with lock:
            done += 1
            if it is None:
                skips[reason] += 1
            else:
                built.append(it)
            if done % 500 == 0:
                print(f"  parsed {done}/{len(missing)} (skipped {sum(skips.values())})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, missing))

    sizes = [_item_bytes(i) for i in built]
    print(f"\nreconstructed {len(built)} records ({sum(skips.values())} unrecoverable)")
    if skips:
        for reason, n in skips.most_common():
            print(f"    {n:>6}  {reason}")
    if sizes:
        print(f"  payload {sum(sizes) / 1e6:.1f} MB  (largest item {max(sizes):,} B "
              f"of {ITEM_GUARD:,} B guard)")
    with_model = sum(1 for i in built if "model" in i)
    print(f"  with model label: {with_model}/{len(built)}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to write.")
        for i in built[:3]:
            print(
                f"  sample {i['job_id']['S']} created={i['created_at']['S']} "
                f"model={i.get('model', {}).get('S', '-')} "
                f"outputs={len(json.loads(i['output_keys']['S']))}"
            )
        return 0

    print(f"\nwriting {len(built)} records to {TABLE} ...")
    written, failed = _write_batches(ddb, built)
    print(f"done: {written} written, {failed} failed, {sum(skips.values())} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
