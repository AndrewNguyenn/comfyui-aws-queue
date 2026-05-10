# comfyui-aws-queue

Cost-effective AWS deployment of ComfyUI with spot-based GPU workers, queue-driven job execution, and per-fleet specialization for image vs video generation.

> ⚠️ **Personal-project scope.** Single-user, single-region (us-west-2), spot-only compute. Not designed for multi-tenant or production SLA workloads.

## Architecture (one-paragraph version)

A vanilla ComfyUI frontend hosted on S3 lets the user build workflows in the browser. Submitting a workflow hits an API Gateway → Lambda dispatcher that classifies the workflow as image or video and enqueues it to the appropriate SQS queue. Two ECS-on-EC2 spot fleets — one g4dn (image, max=1) and one g5 (video, max=3) — pull jobs, run ComfyUI locally, stream models from S3 to a per-instance EBS cache, and write outputs back to S3. Cognito guards the API; CloudWatch alarms + a kill-switch Lambda guard the bill. CodeBuild builds the worker images and pushes to ECR. See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the full design.

## Cost expectation

**~$95-110/mo** at the projected usage of ~65 hours/month, ~100 images/hr + ~20 videos/hr. Spot prices in us-west-2 fluctuate; the figure assumes prices at deploy time. A $100/mo AWS Budget alarm + automatic kill-switch protect against runaway spend.

## Repository layout

```
infra/             CDK app (TypeScript) — all AWS resources
workers/           Container images for image and video workers (Python + Dockerfile)
services/          Lambda functions (dispatcher, downloader, catalog, status)
frontend/          Vanilla ComfyUI bundle + standalone /models page + viewer
codebuild/         CodeBuild buildspecs for worker images
scripts/           Operations: bootstrap, deploy, kill-switch restore, DLQ replay
examples/          Sample workflows for smoke-testing
```

## Prerequisites

1. **AWS account + CLI configured** with admin or CDK-deploy permissions
2. **Service quota raised** for "All G and VT Spot Instance Requests" in your region (default is 0; request ≥40 vCPU). Lead time 24-72 hours.
3. **Node.js ≥20**, **Python ≥3.10**, **AWS CDK CLI** (`npm install -g aws-cdk`)
4. **GitHub CLI** (for the public push, optional)
5. *(Optional)* CivitAI API token if you want to download gated models

## Deployment

Deploys are staged with cost gates between phases. Read `scripts/deploy-stacks.sh` before running.

```bash
# Phase 0: pre-flight checks
./scripts/preflight.sh

# Phase 1: free stacks (network, storage, queues, monitoring with $100 budget)
./scripts/deploy-stacks.sh phase1

# Phase 2: API + Cognito + Lambdas (cents at idle)
./scripts/deploy-stacks.sh phase2

# Phase 3: CodeBuild + first container build (~$0.50 one-off)
./scripts/deploy-stacks.sh phase3
./scripts/trigger-build.sh image
./scripts/trigger-build.sh video

# Phase 4: ECS + ASGs (instances launched on first job, not at deploy)
./scripts/deploy-stacks.sh phase4

# Phase 5: frontend (S3 static, $0)
./scripts/deploy-stacks.sh phase5

# Phase 6: smoke test (this triggers the first real GPU spend)
./scripts/smoke-test.sh
```

## Operations

- **View costs:** `./scripts/check-costs.sh` (current month spend by tag)
- **Kill-switch (manual):** `./scripts/emergency-shutdown.sh` — drains queues, sets ASGs to max=0, disables Cognito user
- **Restore after kill-switch:** `./scripts/restore-after-killswitch.sh`
- **Replay DLQ:** `./scripts/replay-dlq.sh image|video`
- **Manual AMI rollback:** see `docs/runbook.md`

## Security model

- **Public API** behind **Cognito user pool** (single user, JWT-authenticated).
- **No SSH** to workers. EC2 instance profile only. Workers initiate all communication outbound.
- **S3 buckets** for outputs/uploads/models have Block Public Access enforced. Only the frontend bucket is public-read, and only for static assets.
- **API Gateway throttling** at 5 req/sec per source IP, 50 burst. CloudWatch alarm at >100 req/5min triggers kill-switch.
- **`gitleaks` pre-commit hook** + CI check prevents secrets from landing in git.
- **Never set ComfyUI to `--listen 0.0.0.0`.** Workers have public IPs (security group blocks inbound, but a misconfigured SG would expose ComfyUI to the world).

## Known limitations

- Single AZ (us-west-2a) — AZ outage = full outage
- Spot interruption mid-generation loses that one in-flight job (auto-requeue handles the rest)
- First-job-after-cold-start: 3 min for image, 5-7 min for video (model fetch + ComfyUI startup)
- No live-preview WebSocket frames in v1 (would need Fargate dispatcher); status updates via polling
- No automated GitHub→CodeBuild webhook in v1 (manual `trigger-build.sh`)

## License

MIT — see [LICENSE](./LICENSE).

## Disclaimer

This project provisions GPU resources that incur real charges on the deployer's AWS account. Read the implementation plan and run `./scripts/preflight.sh` before deploying. The author of this code is not responsible for AWS bills incurred by users of this project.
