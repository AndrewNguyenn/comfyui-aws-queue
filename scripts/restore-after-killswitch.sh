#!/usr/bin/env bash
# Reverse emergency-shutdown.sh. Restores ASG capacity and re-enables Cognito client.
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"

green(){ printf "\033[32m%s\033[0m\n" "$*"; }

echo "==> Restoring ASG capacity (image: max=1, video: max=3, both at min=0)"
aws autoscaling update-auto-scaling-group --auto-scaling-group-name comfy-image-asg \
  --min-size 0 --max-size 1 --desired-capacity 0 --region "$REGION" || true
aws autoscaling update-auto-scaling-group --auto-scaling-group-name comfy-video-asg \
  --min-size 0 --max-size 3 --desired-capacity 0 --region "$REGION" || true

pool_id="$(aws cognito-idp list-user-pools --max-results 50 --region "$REGION" \
  --query "UserPools[?Name=='comfyui-aws-queue-users'].Id | [0]" --output text 2>/dev/null || true)"
if [ -n "$pool_id" ] && [ "$pool_id" != "None" ]; then
  client_id="$(aws cognito-idp list-user-pool-clients --user-pool-id "$pool_id" --region "$REGION" \
    --query "UserPoolClients[0].ClientId" --output text 2>/dev/null || true)"
  if [ -n "$client_id" ] && [ "$client_id" != "None" ]; then
    echo "==> Re-enabling Cognito client $client_id"
    aws cognito-idp update-user-pool-client \
      --user-pool-id "$pool_id" \
      --client-id "$client_id" \
      --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_USER_SRP_AUTH ALLOW_REFRESH_TOKEN_AUTH \
      --region "$REGION" || true
  fi
fi

green "✓ Restored. ASGs at min=0 (no instances yet) — next job submission scales up normally."
