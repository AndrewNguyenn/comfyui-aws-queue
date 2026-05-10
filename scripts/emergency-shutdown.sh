#!/usr/bin/env bash
#
# Manual kill-switch. Run this if you see runaway spend or unexpected behavior.
#
# Actions:
#   1. Set both ASGs max=0 (terminates all GPU instances immediately)
#   2. Purge both SQS queues (no more pending jobs)
#   3. Disable the Cognito user pool client (no more API calls)
#
# Recovery: ./scripts/restore-after-killswitch.sh
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"

red(){ printf "\033[31m%s\033[0m\n" "$*"; }
green(){ printf "\033[32m%s\033[0m\n" "$*"; }

red "=== EMERGENCY SHUTDOWN ==="
echo "This will:"
echo "  - Set ASG comfy-image-asg and comfy-video-asg to min=0, max=0 (terminates running instances)"
echo "  - Purge SQS queues comfy-image-jobs and comfy-video-jobs"
echo "  - Disable the Cognito user pool app client (signs everyone out)"
echo
read -p "Type 'KILL' to confirm: " -r confirm
if [ "$confirm" != "KILL" ]; then
  echo "aborted"
  exit 1
fi

for asg in comfy-image-asg comfy-video-asg; do
  echo "==> ASG $asg → min=0 max=0"
  aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name "$asg" \
    --min-size 0 --max-size 0 --desired-capacity 0 \
    --region "$REGION" || red "  warning: failed (ASG may not exist yet)"
done

for q in comfy-image-jobs comfy-video-jobs; do
  url="$(aws sqs get-queue-url --queue-name "$q" --region "$REGION" --query 'QueueUrl' --output text 2>/dev/null || true)"
  if [ -n "$url" ]; then
    echo "==> Purging SQS $q"
    aws sqs purge-queue --queue-url "$url" --region "$REGION" || red "  warning: purge failed"
  fi
done

# Disable Cognito user pool client (find by name)
pool_id="$(aws cognito-idp list-user-pools --max-results 50 --region "$REGION" \
  --query "UserPools[?Name=='comfyui-aws-queue-users'].Id | [0]" --output text 2>/dev/null || true)"
if [ -n "$pool_id" ] && [ "$pool_id" != "None" ]; then
  client_id="$(aws cognito-idp list-user-pool-clients --user-pool-id "$pool_id" --region "$REGION" \
    --query "UserPoolClients[0].ClientId" --output text 2>/dev/null || true)"
  if [ -n "$client_id" ] && [ "$client_id" != "None" ]; then
    echo "==> Disabling Cognito client $client_id"
    # Setting all auth flows to false effectively disables sign-in.
    aws cognito-idp update-user-pool-client \
      --user-pool-id "$pool_id" \
      --client-id "$client_id" \
      --explicit-auth-flows '[]' \
      --region "$REGION" || red "  warning: failed"
  fi
fi

green "✓ Kill-switch complete. Run scripts/restore-after-killswitch.sh to bring it back up."
