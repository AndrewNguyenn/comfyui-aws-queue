#!/usr/bin/env bash
# Show current month spend by Component tag for the project.
# Requires cost-allocation tags activated in Billing console (one-time, ~24h to populate).
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
START="$(date -u +%Y-%m-01)"
END="$(date -u -v+1m +%Y-%m-01 2>/dev/null || date -u -d '+1 month' +%Y-%m-01)"

echo "==> Cost by Component for $START → $END (Project=comfyui-aws-queue)"
aws ce get-cost-and-usage \
  --time-period "Start=$START,End=$END" \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --filter '{"Tags":{"Key":"Project","Values":["comfyui-aws-queue"]}}' \
  --group-by 'Type=TAG,Key=Component' \
  --output table 2>&1 || echo "  (no data — cost allocation tags may not be activated yet)"

echo
echo "==> Cost by Service (all project resources)"
aws ce get-cost-and-usage \
  --time-period "Start=$START,End=$END" \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --filter '{"Tags":{"Key":"Project","Values":["comfyui-aws-queue"]}}' \
  --group-by 'Type=DIMENSION,Key=SERVICE' \
  --output table 2>&1 || true
