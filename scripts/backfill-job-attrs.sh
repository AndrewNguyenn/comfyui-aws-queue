#!/usr/bin/env bash
# One-time backfill of the denormalized model/character/subject attrs onto
# existing comfy-jobs rows, so the /jobs list (which reads them off the
# reprojected status-index GSI) shows model labels on historical rows and
# cancel-group matches old in-flight rows.
#
# Derivation uses services/dispatcher/extract.py — the SAME logic the dispatcher
# now writes at job time and the status Lambda reads — so backfilled values
# match new-write and read-time derivation exactly.
#
# Idempotent: only SETs an attr that is missing AND non-empty, so re-runs are
# safe and it skips rows already done. Covers ALL statuses (incl queued/running)
# so old in-flight rows get attrs for cancel-group.
#
# Dry-run by default (reports what it would change). Pass --apply to write.
#   ./scripts/backfill-job-attrs.sh            # dry run
#   ./scripts/backfill-job-attrs.sh --apply    # perform the updates
#
# Region is forced to us-east-1 (the deployment region — note replay-dlq.sh
# defaults to us-west-2; do NOT inherit that here).
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
TABLE="${JOBS_TABLE:-comfy-jobs}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Backfill model/character/subject on $TABLE ($REGION) — $([ $APPLY -eq 1 ] && echo APPLY || echo 'DRY RUN')"
if [ $APPLY -eq 1 ]; then
  read -p "This will UpdateItem rows in $TABLE. Proceed? [y/N] " -n 1 -r; echo
  [[ $REPLY =~ ^[Yy]$ ]] || exit 0
fi

REGION="$REGION" TABLE="$TABLE" APPLY="$APPLY" REPO_ROOT="$REPO_ROOT" python3 - <<'PY'
import json, os, sys
import boto3

sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "services", "dispatcher"))
import extract  # the dispatcher's verbatim mirror of the status extract helpers

region = os.environ["REGION"]
table = os.environ["TABLE"]
apply = os.environ["APPLY"] == "1"
ddb = boto3.client("dynamodb", region_name=region)

paginator = ddb.get_paginator("scan")
scanned = updated = skipped = no_change = errors = 0

for page in paginator.paginate(
    TableName=table,
    # Only the attrs we need: job_id (key) + workflow_json (to derive from) +
    # the three targets (to detect what's already set). Skip workflow_ui (~350 KB).
    ProjectionExpression="job_id, workflow_json, #m, #c, subject",
    ExpressionAttributeNames={"#m": "model", "#c": "character"},
):
    for it in page.get("Items", []):
        scanned += 1
        job_id = it["job_id"]["S"]
        wf_raw = it.get("workflow_json", {}).get("S", "")
        try:
            wf = json.loads(wf_raw) if wf_raw else {}
        except (json.JSONDecodeError, TypeError):
            wf = {}
        if not isinstance(wf, dict) or not wf:
            skipped += 1  # no usable graph to derive from
            continue
        desired = {}
        model = extract._extract_model(wf)
        character, subject = extract._extract_subject(wf)
        if model:
            desired["model"] = model
        if character:
            desired["character"] = character
        if subject:
            desired["subject"] = subject
        # Only set attrs that are missing or empty on the row (idempotent).
        to_set = {k: v for k, v in desired.items()
                  if not it.get(k, {}).get("S", "")}
        if not to_set:
            no_change += 1
            continue
        if not apply:
            updated += 1
            if updated <= 8:
                print(f"  would set {job_id}: {to_set}")
            continue
        names = {f"#k{i}": k for i, k in enumerate(to_set)}
        values = {f":v{i}": {"S": v} for i, (k, v) in enumerate(to_set.items())}
        expr = "SET " + ", ".join(f"#k{i} = :v{i}" for i in range(len(to_set)))
        try:
            ddb.update_item(TableName=table, Key={"job_id": {"S": job_id}},
                            UpdateExpression=expr,
                            ExpressionAttributeNames=names,
                            ExpressionAttributeValues=values)
            updated += 1
            if updated % 500 == 0:
                print(f"  ...updated {updated}")
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  ERROR updating {job_id}: {e!r}")

verb = "updated" if apply else "would update"
print(f"\nscanned={scanned}  {verb}={updated}  already-set={no_change}  "
      f"no-graph={skipped}  errors={errors}")
if not apply:
    print("DRY RUN — re-run with --apply to write.")
PY
