#!/usr/bin/env bash
# Drain a DLQ back to its main queue with a confirmation.
#   ./scripts/replay-dlq.sh image
#   ./scripts/replay-dlq.sh video
set -euo pipefail

target="${1:-}"
if [ "$target" != "image" ] && [ "$target" != "video" ]; then
  echo "Usage: $0 {image|video}" >&2
  exit 1
fi

REGION="${AWS_REGION:-us-west-2}"
DLQ_NAME="comfy-${target}-jobs-dlq"
MAIN_NAME="comfy-${target}-jobs"

dlq_url="$(aws sqs get-queue-url --queue-name "$DLQ_NAME" --region "$REGION" --query 'QueueUrl' --output text)"
main_url="$(aws sqs get-queue-url --queue-name "$MAIN_NAME" --region "$REGION" --query 'QueueUrl' --output text)"

count="$(aws sqs get-queue-attributes --queue-url "$dlq_url" --attribute-names ApproximateNumberOfMessages \
  --region "$REGION" --query 'Attributes.ApproximateNumberOfMessages' --output text)"

echo "DLQ $DLQ_NAME has ~$count messages."
read -p "Replay all to $MAIN_NAME? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

while true; do
  resp="$(aws sqs receive-message --queue-url "$dlq_url" --region "$REGION" \
    --max-number-of-messages 10 --wait-time-seconds 1 --output json)"
  msgs="$(echo "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("Messages", [])))')"
  if [ "$msgs" = "[]" ]; then break; fi
  echo "$msgs" | REGION="$REGION" MAIN_URL="$main_url" DLQ_URL="$dlq_url" python3 - <<'PY'
import json, os, sys
import boto3
sqs = boto3.client("sqs", region_name=os.environ["REGION"])
main = os.environ["MAIN_URL"]
dlq = os.environ["DLQ_URL"]
msgs = json.loads(sys.stdin.read())
for m in msgs:
    sqs.send_message(QueueUrl=main, MessageBody=m["Body"])
    sqs.delete_message(QueueUrl=dlq, ReceiptHandle=m["ReceiptHandle"])
    print(f"replayed {m.get('MessageId')}")
PY
done

echo "✓ DLQ drain complete."
