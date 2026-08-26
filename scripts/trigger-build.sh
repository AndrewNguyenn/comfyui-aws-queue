#!/usr/bin/env bash
# Manually trigger a CodeBuild project. v1 has no GitHub webhook.
#   ./scripts/trigger-build.sh image
#   ./scripts/trigger-build.sh video
set -euo pipefail

target="${1:-}"
case "$target" in
  image|video)      PROJECT="comfy-build-${target}-worker" ;;
  video-blackwell)  PROJECT="comfy-build-video-worker-blackwell" ;;
  *)
    echo "Usage: $0 {image|video|video-blackwell}" >&2
    exit 1 ;;
esac
REGION="${AWS_REGION:-us-east-1}"

echo "==> Starting CodeBuild project: $PROJECT"
build_id=$(
  aws codebuild start-build \
    --project-name "$PROJECT" \
    --region "$REGION" \
    --query 'build.id' \
    --output text
)
echo "    build id: $build_id"
echo "    Watch:    aws codebuild batch-get-builds --ids '$build_id' --region $REGION --query 'builds[0].{phase:currentPhase,status:buildStatus,logs:logs.deepLink}' --output table"
echo "    Or in console: https://console.aws.amazon.com/codesuite/codebuild/projects/$PROJECT"
