# CodeBuild

Two CodeBuild projects (`comfy-build-image-worker`, `comfy-build-video-worker`)
defined in `infra/lib/stacks/ci.ts`. They build the Docker images defined under
`workers/{image,video}/` and push to ECR.

## Manual trigger

```
./scripts/trigger-build.sh image    # ~30 min cold, <10 min warm
./scripts/trigger-build.sh video    # ~45 min cold (sage CUDA build), <15 min warm
```

## Watching progress

```
aws codebuild list-builds-for-project --project-name comfy-build-image-worker
aws codebuild batch-get-builds --ids <build-id> --query 'builds[0].logs'
```

Or in console: CodeBuild → Build projects → comfy-build-image-worker → Build history.

## Cost

- Cold build: ~$0.30 image, ~$0.50 video (BUILD_GENERAL1_LARGE × build minutes × $0.01/min)
- Warm build (cache hit): ~$0.10 each
- Storage in ECR: included in the project cost estimate ($1-2/mo for both repos)

## Image rollout to running workers

After a successful build, the new image is in ECR but running workers continue
using their cached image. To pick up the new version:

```
aws ecs update-service --cluster comfy-cluster --service comfy-image --force-new-deployment
aws ecs update-service --cluster comfy-cluster --service comfy-video --force-new-deployment
```

This terminates running tasks and lets ECS launch new ones with the latest tag.
With `min=0` and 1-user usage, this typically just affects the next-job cold start.
