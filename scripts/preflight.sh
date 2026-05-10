#!/usr/bin/env bash
#
# Pre-flight checks before any deploy or build.
#
# - Verifies AWS CLI is configured and points at the EXPECTED account
#   (refuses to run if EXPECTED_ACCOUNT_ID env var is set and doesn't match)
# - Verifies region is us-west-2
# - Checks for required tooling (node, npm, cdk)
# - Optionally checks the SSM parameter for budget alert email
#
# Exits non-zero if anything is wrong. Run this before deploy-stacks.sh.
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }

err=0

echo "==> Pre-flight checks"

# AWS account
caller="$(aws sts get-caller-identity --output json 2>&1)" || {
  red "✗ AWS CLI cannot get caller identity. Run 'aws configure' or set credentials."
  err=1
}
if [ "$err" = "0" ]; then
  account="$(echo "$caller" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])')"
  arn="$(echo "$caller" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')"
  echo "    AWS account: $account"
  echo "    ARN: $arn"

  if [ -n "${EXPECTED_ACCOUNT_ID:-}" ] && [ "$account" != "$EXPECTED_ACCOUNT_ID" ]; then
    red "✗ Account mismatch: got $account, expected $EXPECTED_ACCOUNT_ID"
    red "  This is a deployment-safety check. If you really mean to deploy to $account,"
    red "  unset EXPECTED_ACCOUNT_ID or set it to match."
    err=1
  fi
fi

# Region
configured_region="$(aws configure get region 2>/dev/null || echo unset)"
if [ "$configured_region" != "$REGION" ]; then
  yellow "⚠  Configured region is '$configured_region'; this project deploys to '$REGION'"
  yellow "    Either run 'aws configure set region $REGION' or pass --region on CLI calls."
fi

# Tooling
for tool in node npm aws; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    red "✗ Missing tool: $tool"
    err=1
  fi
done

if ! command -v cdk >/dev/null 2>&1; then
  yellow "⚠  AWS CDK CLI not installed. Run: npm install -g aws-cdk"
fi

# SSM parameter: budget email
if ! aws ssm get-parameter --name /comfy/alerts/budget-email --region "$REGION" >/dev/null 2>&1; then
  yellow "⚠  SSM parameter /comfy/alerts/budget-email is missing. Set it before deploying:"
  yellow "    aws ssm put-parameter --name /comfy/alerts/budget-email \\"
  yellow "        --value 'you@example.com' --type String --region $REGION"
fi

# Spot vCPU quota — informational
quota="$(aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-3819A6DF \
  --region "$REGION" \
  --query 'Quota.Value' --output text 2>/dev/null || echo unknown)"
if [ "$quota" = "0.0" ] || [ "$quota" = "0" ]; then
  yellow "⚠  G/VT Spot vCPU quota is 0 in $REGION. Compute stack will fail to launch instances."
  yellow "    Request increase via console: Service Quotas → EC2 → 'All G and VT Spot Instance Requests'"
  yellow "    Recommend ≥40 vCPU. Lead time 24-72h."
elif [ "$quota" != "unknown" ]; then
  green "✓ G/VT Spot vCPU quota: $quota in $REGION"
fi

if [ "$err" -ne 0 ]; then
  red "Pre-flight FAILED. Fix the issues above before deploying."
  exit 1
fi

green "✓ Pre-flight OK"
