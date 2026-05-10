#!/usr/bin/env bash
#
# Staged deploy with cost gates. Phases:
#   phase1: free stacks (network, storage, queue, monitoring)  — $0/mo idle
#   phase2: auth + api                                          — <$1/mo idle
#   phase3: ci (CodeBuild + ECR repos)                          — $0 until builds run
#   phase4: compute (ECS cluster, ASGs at min=0)                — $0 until first job
#   phase5: frontend (S3 static hosting)                        — $0
#
# Each phase prompts for confirmation before deploying. Run preflight.sh first.
#
# Usage:  ./scripts/deploy-stacks.sh phase1
#         ./scripts/deploy-stacks.sh all          # all phases sequentially with prompts
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase="${1:-}"

confirm() {
  read -p "$1 [y/N] " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

deploy_stacks() {
  local label="$1"; shift
  echo
  echo "==> $label"
  echo "    stacks: $*"
  if ! confirm "    Proceed?"; then
    echo "    skipped"
    return
  fi
  ( cd infra && npx cdk deploy "$@" --require-approval never )
}

case "$phase" in
  phase1)
    deploy_stacks "Phase 1: free stacks (network, storage, queue, monitoring)" \
      ComfyNetworkStack ComfyStorageStack ComfyQueueStack ComfyMonitoringStack
    ;;
  phase2)
    deploy_stacks "Phase 2: auth + api" \
      ComfyAuthStack ComfyApiStack
    ;;
  phase3)
    deploy_stacks "Phase 3: ci" \
      ComfyCiStack
    ;;
  phase4)
    echo "WARNING: Phase 4 deploys ECS+ASGs. They start at min=0 so no instances launch"
    echo "         until a job is submitted. But: if G/VT spot vCPU quota is 0, even the first"
    echo "         job submission will sit unfulfilled."
    deploy_stacks "Phase 4: compute" \
      ComfyComputeStack
    ;;
  phase5)
    echo "Note: Phase 5 needs frontend/dist/ to exist. Run ./frontend/build.sh first."
    if [ ! -d "frontend/dist" ]; then
      echo "ERROR: frontend/dist/ not found. Run ./frontend/build.sh first."
      exit 1
    fi
    deploy_stacks "Phase 5: frontend" \
      ComfyFrontendStack
    ;;
  all)
    "$0" phase1
    "$0" phase2
    "$0" phase3
    "$0" phase4
    "$0" phase5
    ;;
  *)
    echo "Usage: $0 {phase1|phase2|phase3|phase4|phase5|all}"
    exit 1
    ;;
esac

echo
echo "Deploy complete. Verify with: aws cloudformation list-stacks --query 'StackSummaries[?contains(StackName, \`Comfy\`)].[StackName,StackStatus]' --output table"
