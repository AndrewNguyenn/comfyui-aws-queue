# comfyui-aws-queue — Implementation Plan

**Status:** v3 (post-code-reviewer grilling)
**Owner:** Andrew Nguyen
**Last updated:** 2026-05-10

> **Read sections 15 (self-grilling), 16 (iterations), 19 (independent review findings), and 20 (final iterations) before approving this plan.** Several v1/v2 assumptions were wrong and have been corrected. The code-reviewer agent caught 5 Critical issues missed in self-grilling (log spend, scaling overshoot, missing Secrets Manager, public-repo leakage, world-readable API key).

---

## 1. Goal & Constraints (recap)

Build a cost-effective ComfyUI generation service on AWS for a single user with these characteristics:

- **Throughput target:** ~100 images/hr, ~20 videos/hr (Wan 2.2 14B I2V + Lightning LoRA)
- **Active usage:** ~15 hrs/week (~65 hrs/mo)
- **GPUs:** g4dn.xlarge spot (image), g5.xlarge spot (video), with g5.2xlarge / g6e.xlarge fallbacks
- **Image fleet:** max=1 (no scaling beyond a single instance)
- **Video fleet:** scales 0→5 on SQS backlog (target burst capacity ~150/hr)
- **Storage:** S3 with Intelligent-Tiering for ~1.4 TB model catalog
- **No EFS, no CDN, no AMI snapshots** (per user's cost constraints)
- **Spot interruption recovery:** SQS visibility timeout, in-flight job auto-requeue
- **Auth:** API key in custom header (API Gateway built-in keys + usage plan)
- **Region:** us-west-2
- **Budget envelope:** ~$74/mo at projected usage

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser (laptop)                                                    │
│  - vanilla ComfyUI frontend (Vite bundle from S3 static hosting)     │
│  - models.html standalone page (CivitAI URL paste, catalog browse)   │
│  - viewer (image/video gallery from S3 presigned URLs)               │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ HTTPS, x-api-key header
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  API Gateway (REST API, API key required)                            │
└────────────────────┬─────────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┬──────────────────┬───────────────┐
        ▼                         ▼                  ▼               ▼
   ┌──────────┐             ┌──────────┐      ┌──────────┐    ┌──────────┐
   │Dispatcher│             │Downloader│      │ Catalog  │    │ Status   │
   │ Lambda   │             │ Lambda   │      │ Lambda   │    │ Lambda   │
   │          │             │          │      │          │    │          │
   │POST/prompt             │POST/down │      │GET /models│   │GET /jobs/│
   │GET /history             │GET /down│      │POST /modls│   │GET /view │
   │GET /object_info         │/{id}    │      │          │    │/upload   │
   └─────┬────┘             └─────┬────┘      └─────┬────┘    └─────┬────┘
         │                        │                 │               │
         │ SendMessage            │ Streaming PUT   │ Read/Write    │ Read/Write
         ▼                        │ to S3           ▼               ▼
   ┌─────────┐                    ▼            ┌──────┐         ┌──────┐
   │SQS      │              ┌──────────┐       │ DDB  │         │  S3  │
   │image-jobs                │   S3   │       │models│         │outputs│
   │video-jobs              │  models  │       │ jobs │         │uploads│
   └────┬────┘              │ (1.4 TB) │       └──────┘         └──────┘
        │                   └──────────┘
        │
   ┌────▼─────────────────────────────────────────┐
   │ ECS Cluster                                  │
   │                                              │
   │ ┌─────────────────┐    ┌──────────────────┐  │
   │ │ image-fleet     │    │ video-fleet      │  │
   │ │ ASG g4dn.xlarge │    │ ASG g5.xlarge    │  │
   │ │ spot, min=0,max=1   │ spot, min=0,max=5   │
   │ │                 │    │                  │  │
   │ │ Capacity        │    │ Capacity         │  │
   │ │ Provider:       │    │ Provider:        │  │
   │ │ cp-image        │    │ cp-video         │  │
   │ │                 │    │                  │  │
   │ │ Service:        │    │ Service:         │  │
   │ │ comfy-image     │    │ comfy-video      │  │
   │ │                 │    │                  │  │
   │ │ Task: 1 per     │    │ Task: 1 per      │  │
   │ │ instance        │    │ instance         │  │
   │ │                 │    │                  │  │
   │ │ Container:      │    │ Container:       │  │
   │ │ comfy-image:tag │    │ comfy-video:tag  │  │
   │ │ (ECR)           │    │ (ECR)            │  │
   │ └─────────────────┘    └──────────────────┘  │
   └───────────────────────────────────────────────┘

   ┌─────────────────────────────────────────┐
   │ CodeBuild                               │
   │ - buildspec-image-worker.yml            │
   │ - buildspec-video-worker.yml            │
   │ Triggered on git push (CodePipeline)    │
   │ Builds Docker images, pushes to ECR     │
   └─────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
comfyui-aws-queue/
├── README.md
├── IMPLEMENTATION_PLAN.md          (this file)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                  (lint, cdk synth on PRs)
│
├── infra/                          (CDK app, TypeScript)
│   ├── bin/
│   │   └── comfyui-aws-queue.ts    (app entrypoint)
│   ├── lib/
│   │   ├── stacks/
│   │   │   ├── network.ts          (VPC, public subnets, SG)
│   │   │   ├── storage.ts          (S3 buckets, DDB tables)
│   │   │   ├── queue.ts            (SQS queues + DLQ)
│   │   │   ├── compute.ts          (ECS cluster, ASGs, capacity providers)
│   │   │   ├── api.ts              (API GW, dispatcher, downloader, catalog Lambdas)
│   │   │   ├── frontend.ts         (S3 static hosting)
│   │   │   ├── ci.ts               (CodeBuild project, ECR repos)
│   │   │   └── monitoring.ts       (CW dashboards, alarms, cost budget)
│   │   ├── constructs/
│   │   │   ├── spot-asg.ts         (mixed-instance spot ASG with capacity rebalance)
│   │   │   └── lambda-fn.ts        (Python Lambda factory)
│   │   └── config.ts               (env-specific config: instance types, AMI lookups)
│   ├── package.json
│   ├── tsconfig.json
│   └── cdk.json
│
├── workers/
│   ├── shared/                     (Python code shared between image+video)
│   │   ├── __init__.py
│   │   ├── cache_manager.py        (S3-backed model cache w/ LRU + pinning)
│   │   ├── spot_handler.py         (IMDS spot termination watcher)
│   │   ├── comfy_client.py         (HTTP client for local ComfyUI)
│   │   ├── ddb_client.py           (job state updates)
│   │   ├── s3_client.py            (output upload, model download)
│   │   └── worker.py               (main SQS poll loop)
│   ├── image/
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   └── extra_models.json       (image-specific pinned models)
│   └── video/
│       ├── Dockerfile
│       ├── entrypoint.sh
│       └── extra_models.json       (video-specific pinned models, incl. Wan)
│
├── services/                       (Lambda code)
│   ├── dispatcher/
│   │   ├── handler.py              (POST /prompt, GET /history, GET /object_info)
│   │   ├── workflow_router.py      (detect image vs video by node class_type)
│   │   ├── object_info_builder.py  (synthesize ComfyUI /object_info from DDB)
│   │   └── requirements.txt
│   ├── downloader/
│   │   ├── handler.py              (CivitAI URL → S3 multipart streaming)
│   │   ├── civitai_client.py
│   │   └── requirements.txt
│   ├── catalog/
│   │   ├── handler.py              (GET/POST/DELETE /models)
│   │   └── requirements.txt
│   └── status/
│       ├── handler.py              (GET /jobs/{id}, GET /view, /upload)
│       └── requirements.txt
│
├── frontend/
│   ├── comfyui/                    (vanilla ComfyUI frontend, vendored or submodule)
│   │   └── (Vite app, configured to point at our API)
│   ├── models/                     (standalone /models.html page)
│   │   ├── index.html
│   │   ├── app.js                  (catalog browse, CivitAI download form)
│   │   └── styles.css
│   ├── viewer/                     (image/video gallery)
│   │   ├── index.html
│   │   └── app.js
│   └── build.sh                    (assemble + sync to S3)
│
├── codebuild/
│   ├── buildspec-image-worker.yml
│   ├── buildspec-video-worker.yml
│   └── README.md
│
└── scripts/
    ├── bootstrap.sh                (cdk bootstrap, npm ci, etc.)
    ├── deploy-stacks.sh            (ordered CDK deploy)
    ├── check-costs.sh              (current month spend snapshot)
    ├── trigger-build.sh            (start CodeBuild project)
    └── seed-catalog.sh             (sample models inserted into DDB for testing)
```

---

## 4. AWS Resources by Stack

### 4.1 NetworkStack
- VPC: 10.0.0.0/16, single AZ (us-west-2a) for cost (drop second AZ saves $7/mo)
- Public subnets only (no NAT). Workers get public IPs (~$0.005/hr each).
- S3 gateway endpoint (free)
- Security groups:
  - `sg-worker`: outbound only (HTTPS to AWS APIs, S3 via gateway endpoint)
  - No inbound; workers receive jobs via SQS poll, never accept connections

### 4.2 StorageStack
- S3 buckets:
  - `comfy-models-{account}`: 1.4 TB model catalog. Intelligent-Tiering with Archive Instant Access opt-in.
  - `comfy-outputs-{account}`: generated images/videos. Lifecycle: standard → IA at 30 days.
  - `comfy-uploads-{account}`: user-uploaded inputs (e.g., I2V source images). Lifecycle: delete after 7 days.
  - `comfy-frontend-{account}`: static website hosting for ComfyUI frontend.
- DDB tables:
  - `comfy-models`: model catalog. PK=`name`, attrs={type, s3_key, size_gb, pinned, civitai_version_id, preview_url, added_at, last_used_at}. GSI on `type` for browse.
  - `comfy-jobs`: job lifecycle. PK=`job_id`, attrs={type, status, workflow_json, input_keys, output_keys, attempt_count, created_at, started_at, completed_at, last_heartbeat, error}. **No TTL** — job history is user data and backs the viewer's gallery; a 30-day `expire_at` used to reap it and was removed 2026-08-19.
  - `comfy-downloads`: download progress. PK=`download_id`, attrs={civitai_url, status, bytes_done, total_bytes, model_name, error}. TTL 24 hrs.

### 4.3 QueueStack
- SQS queues:
  - `image-jobs`: visibility timeout 15 min, message retention 7 days
  - `video-jobs`: visibility timeout 30 min, message retention 7 days
  - `image-jobs-dlq`, `video-jobs-dlq`: max receive count 3 then DLQ
- Worker scaling alarm metric: `ApproximateNumberOfMessagesVisible`

### 4.4 ComputeStack
- ECS cluster: `comfy-cluster` (no Fargate, EC2 only)
- Two ASGs with mixed-instance policy + capacity-optimized spot:

```
image-asg:
  Primary: g4dn.xlarge
  Fallbacks: (none — single type, max=1 anyway)
  AllocationStrategy: capacity-optimized
  CapacityRebalance: true
  Min: 0, Max: 1
  RootVolume: gp3, 150 GB (50 GB OS+container, 100 GB cache)

video-asg:
  Primary: g5.xlarge (weight 4)
  Fallback: g5.2xlarge (weight 3)
  Fallback: g6e.xlarge (weight 2)
  AllocationStrategy: capacity-optimized
  CapacityRebalance: true
  Min: 0, Max: 5
  RootVolume: gp3, 250 GB (50 GB OS+container, 200 GB cache)
```

- Capacity providers: `cp-image-spot`, `cp-video-spot`. Managed scaling enabled, target capacity 100, target backlog per task = 10.
- ECS services: `comfy-image`, `comfy-video`. One task per instance. Placement constraint: `attribute:fleet == image|video`.
- Task definitions:
  - networkMode: host (one task per host, simplest)
  - GPU resource requirement: 1
  - CloudWatch logs driver
  - Env vars: SQS queue URL, DDB table names, S3 bucket names, AWS region, fleet type
  - IAM task role: SQS receive/delete on its queue, DDB read/write jobs+models, S3 read models bucket, S3 write outputs bucket

### 4.5 ApiStack
- API Gateway REST API (regional, not edge)
- API key + usage plan (1000 req/day quota, 10 req/s burst — plenty for 1 user, caps blast radius)
- Routes:
  - `POST /prompt` → DispatcherFn
  - `GET /history/{id}` → DispatcherFn
  - `GET /object_info` → DispatcherFn
  - `POST /upload/image` → StatusFn (returns presigned PUT URL)
  - `GET /view` → StatusFn (returns presigned GET URL)
  - `POST /models/download` → DownloaderFn (kicks off async)
  - `GET /downloads/{id}` → DownloaderFn (status)
  - `GET /models` → CatalogFn
  - `POST /models` → CatalogFn (manual add)
  - `DELETE /models/{name}` → CatalogFn
- All routes require `x-api-key` header
- CORS configured for the frontend bucket's origin
- Lambda functions: Python 3.12, 512 MB default (downloader: 1024 MB, 15 min timeout), arm64 architecture (cheaper)

### 4.6 FrontendStack
- S3 bucket with static website hosting
- Public-read bucket policy on the frontend prefix only
- CORS allowing the API GW domain
- No CloudFront (per cost constraint)

### 4.7 CIStack
- ECR repos: `comfy-image-worker`, `comfy-video-worker`. Lifecycle policy: keep last 5 untagged.
- CodeBuild projects:
  - `build-image-worker`: triggers on git push to main with changes under workers/image/ or workers/shared/. BUILD_GENERAL1_LARGE compute (15 GB RAM, 8 vCPU). Builds Docker image, pushes to ECR with both `latest` and commit-sha tags.
  - `build-video-worker`: same but for video.
- CodePipeline (optional, v2): source from GitHub via webhook → CodeBuild → no deploy step (workers fetch latest tag on next launch via ECS service force-new-deployment)
- Alternatively: keep it simple, use GitHub Actions to trigger CodeBuild via AWS CLI. Decision: **CodeBuild + GitHub webhooks** is simpler than full CodePipeline.

### 4.8 MonitoringStack
- CloudWatch dashboard: queue depth, worker count, job throughput, error rate
- Alarms (SNS → email):
  - DLQ depth > 0 (job permanently failed)
  - Job latency p95 > 30 min (something's stuck)
  - Spot interruption rate spike (region capacity issue)
- **AWS Budget: $100/mo with email alert at 50%, 80%, 100%** ← critical safety net

---

## 5. Container Images

### 5.1 Common base layer (multi-stage build)

Stage 1 base (shared between image+video):
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base
# (NVIDIA's NGC PyTorch image — already has PyTorch + CUDA + cuDNN, big but reliable)

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI
WORKDIR /opt/comfy
RUN git clone --depth 1 --branch <PIN_TAG> https://github.com/comfyanonymous/ComfyUI.git . \
    && pip install --no-cache-dir -r requirements.txt

# Sage Attention 2.x
RUN pip install --no-cache-dir sageattention
# (verify install: python -c "import sageattention; print(sageattention.__version__)")

# Worker shared code
COPY workers/shared /opt/worker/shared
RUN pip install --no-cache-dir boto3 requests pyyaml

# AWS SDK + Python deps for worker
COPY workers/shared/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

WORKDIR /opt/worker
COPY workers/<image|video>/entrypoint.sh /entrypoint.sh
COPY workers/<image|video>/extra_models.json /opt/worker/extra_models.json
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

### 5.2 Image worker specifics
- No Wan custom nodes (saves ~2 GB)
- Pinned models: SDXL base, ~1-2 favorite checkpoints (defined in extra_models.json, downloaded at runtime from S3)
- Default port: 8188
- Image size estimate: ~10-12 GB compressed

### 5.3 Video worker specifics
- Add `kijai/ComfyUI-WanVideoWrapper` custom node:
  ```dockerfile
  WORKDIR /opt/comfy/custom_nodes
  RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
      && cd ComfyUI-WanVideoWrapper && pip install --no-cache-dir -r requirements.txt
  ```
- Add ComfyUI-VideoHelperSuite for video output handling
- Pinned models: Wan 2.2 14B I2V Q8 GGUF, Lightning LoRA, VAE
- Image size estimate: ~13-15 GB compressed
- **Default ComfyUI launch flag: `--use-sage-attention`** (sage attention enabled globally)

### 5.4 Sage Attention black-frames bug mitigation
Per research findings: SageAttention + FP8 weights = black frames in some Wan workflows. Default the video worker to **GGUF Q8 + Sage** (not FP8 + Sage). Document this in worker README and pre-pin the Q8 GGUF version.

---

## 6. Worker Code (Python)

### 6.1 worker.py — main loop

```python
def main():
    config = load_config_from_env()
    cache = CacheManager(...)
    spot = SpotHandler(on_terminate=lambda jid: requeue(jid))

    # Start ComfyUI subprocess
    comfy = subprocess.Popen([
        "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
        "--use-sage-attention",
    ], cwd="/opt/comfy")
    wait_for_comfy_ready(timeout=120)

    # Pre-warm pinned models in parallel
    cache.warm_pinned()

    # SQS poll loop
    while not spot.terminating:
        msgs = sqs.receive_message(
            QueueUrl=config.queue_url,
            WaitTimeSeconds=20,
            MaxNumberOfMessages=1,
            VisibilityTimeout=config.visibility_timeout,
        )
        if not msgs.get("Messages"):
            continue

        msg = msgs["Messages"][0]
        job = json.loads(msg["Body"])
        spot.set_in_flight(job["job_id"], msg["ReceiptHandle"])

        try:
            run_job(job, cache, comfy)
            sqs.delete_message(...)
            spot.clear_in_flight()
        except SpotInterrupted:
            # Already requeued by spot handler
            return
        except Exception as e:
            handle_failure(job, e)
            # Visibility timeout will expire and SQS redelivers
```

### 6.2 cache_manager.py — model cache

Key methods:
- `ensure(model_name) -> path`: download if missing, returns local path. Bumps access time. LRU evicts non-pinned to make room.
- `warm_pinned()`: parallel download of all pinned models on boot.
- `evict_lru(bytes_needed)`: removes oldest non-pinned non-in-use models until enough free space.
- Refcount tracking via `with cache.use(model_name):` context manager to prevent eviction during job.
- Lock file per model to prevent concurrent download (within instance — across instances, S3 keys are immutable so worst case both download).

### 6.3 spot_handler.py — interruption recovery

```python
class SpotHandler:
    def __init__(self, on_terminate):
        self.terminating = False
        self.in_flight = None  # (job_id, receipt_handle)
        self.on_terminate = on_terminate
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        url = "http://169.254.169.254/latest/meta-data/spot/instruction"
        # Use IMDSv2 token first
        token = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2,
        ).text
        while not self.terminating:
            try:
                r = requests.get(url, headers={"X-aws-ec2-metadata-token": token}, timeout=2)
                if r.status_code == 200 and r.text.strip() == "terminate":
                    self.terminating = True
                    if self.in_flight:
                        # Reset SQS visibility so message redelivers immediately
                        job_id, receipt = self.in_flight
                        sqs.change_message_visibility(
                            QueueUrl=os.environ["QUEUE_URL"],
                            ReceiptHandle=receipt,
                            VisibilityTimeout=0,
                        )
                        ddb.update_item(...)  # status=queued
                        self.on_terminate(job_id)
                    return
            except (requests.RequestException, Timeout):
                pass
            time.sleep(5)
```

### 6.4 Heartbeat for long jobs

For video jobs that exceed visibility timeout (rare with Lightning LoRA, but possible), the worker should call `ChangeMessageVisibility` periodically:

```python
def heartbeat_loop(receipt_handle, queue_url, interval=300):
    while True:
        time.sleep(interval)
        sqs.change_message_visibility(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=900,  # extend by 15 min
        )
        ddb.update_item(...)  # bump last_heartbeat
```

Backstop: scheduled Lambda sweeps DDB for `last_heartbeat > 10 min` and resets job to `queued` (handles cases where worker dies before spot handler fires).

---

## 7. Lambda Code

### 7.1 Dispatcher (services/dispatcher/handler.py)

Routes:
- `POST /prompt`: parse workflow JSON, call `workflow_router.classify(workflow)` → `image|video`. Write job to DDB. Send SQS message. Return `{prompt_id}`.
- `GET /history/{prompt_id}`: read DDB job record. Format response to match ComfyUI's `/history` shape.
- `GET /object_info`: read DDB models, build ComfyUI-compatible `/object_info` JSON. Cache result in Lambda memory for 60s to avoid hammering DDB.

### 7.2 workflow_router.py

```python
VIDEO_NODE_PATTERNS = [
    "WanVideo", "Wan2", "HunyuanVideo", "LTXVideo", "CogVideoX",
    "AnimateDiff", "VideoLinearCFGGuidance", "VHS_VideoCombine",
]

def classify(workflow_api_json) -> Literal["image", "video"]:
    for node_id, node in workflow_api_json.items():
        class_type = node.get("class_type", "")
        if any(p in class_type for p in VIDEO_NODE_PATTERNS):
            return "video"
    return "image"
```

### 7.3 Downloader (services/downloader/handler.py)

```python
def handler(event, context):
    body = json.loads(event["body"])
    download_id = str(uuid4())

    # Resolve CivitAI URL → real download URL
    version_id = parse_civitai_url(body["civitai_url"])
    secrets = boto3.client("secretsmanager").get_secret_value(SecretId="civitai/api-token")
    token = json.loads(secrets["SecretString"])["token"]

    meta = requests.get(
        f"https://civitai.com/api/v1/model-versions/{version_id}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
    ).json()
    file_meta = meta["files"][0]

    s3_key = f"{body['model_type']}/{file_meta['name']}"
    total = file_meta["sizeKB"] * 1024

    ddb.put_item(TableName="comfy-downloads", Item={
        "download_id": {"S": download_id},
        "status": {"S": "downloading"},
        "total_bytes": {"N": str(int(total))},
        "bytes_done": {"N": "0"},
        "model_name": {"S": file_meta["name"]},
    })

    # Stream from CivitAI to S3 multipart upload (no local disk)
    with requests.get(file_meta["downloadUrl"], stream=True) as r:
        s3_streaming_upload(
            stream=r.iter_content(chunk_size=8 * 1024 * 1024),
            bucket=os.environ["MODELS_BUCKET"],
            key=s3_key,
            on_progress=lambda done: ddb_update_progress(download_id, done),
        )

    # Add to model catalog
    ddb.put_item(TableName="comfy-models", Item={
        "name": {"S": file_meta["name"].rsplit(".", 1)[0]},
        "type": {"S": body["model_type"]},
        "s3_key": {"S": s3_key},
        "size_gb": {"N": str(round(total / 1e9, 2))},
        "pinned": {"BOOL": False},
        "civitai_version_id": {"N": str(version_id)},
        "preview_url": {"S": meta.get("images", [{}])[0].get("url", "")},
        "added_at": {"S": datetime.utcnow().isoformat()},
    })
    ddb.update_item(TableName="comfy-downloads", ..., status="complete")

    return {"statusCode": 200, "body": json.dumps({"download_id": download_id})}
```

Lambda config: 1024 MB memory, 15 min timeout (max), 10 GB ephemeral storage (unused if streaming works), arm64.

### 7.4 Catalog (services/catalog/handler.py)
- `GET /models?type=checkpoint` → query DDB by GSI on type
- `POST /models` → manual add (used by downloader internally; could expose for manual entries)
- `DELETE /models/{name}` → remove from DDB and S3

### 7.5 Status (services/status/handler.py)
- `GET /jobs/{id}` → read DDB job
- `GET /view?key=<s3-key>` → 302 redirect to presigned GET URL (5 min TTL)
- `POST /upload/image` → return presigned PUT URL for direct browser-to-S3 upload (avoid Lambda payload limits)

---

## 8. Frontend Strategy

### 8.1 Vanilla ComfyUI bundle
- Vendor `comfyanonymous/ComfyUI/web` directory contents to `frontend/comfyui/`
- Modify the `api.js` (or equivalent) to:
  - Read API base URL from a config file (`config.js`)
  - Inject `x-api-key` header on all fetch + WS requests
- Build script: `frontend/build.sh`:
  1. Sync vanilla ComfyUI `web/` to local `comfyui/`
  2. Apply our patches (config injection, x-api-key header)
  3. Sync to S3 bucket

### 8.2 Standalone /models.html page
- Plain HTML + vanilla JS (no framework, ~150 lines)
- Reads API base URL + key from `config.js`
- Sections:
  - **Add from CivitAI**: URL input, type dropdown, submit → `POST /models/download`, polls `GET /downloads/{id}`
  - **Catalog**: tree view by type, count + total size, filter
  - **Active downloads**: list with progress bars

### 8.3 Viewer
- Could be a separate page or integrated into models page
- Lists recent outputs (read DDB jobs table by status=complete, sorted by completed_at desc)
- Click thumbnail → presigned URL display

---

## 9. CodeBuild Pipeline

### 9.1 buildspec-image-worker.yml

```yaml
version: 0.2
phases:
  pre_build:
    commands:
      - aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
      - REPO=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/comfy-image-worker
      - SHA=$CODEBUILD_RESOLVED_SOURCE_VERSION
  build:
    commands:
      - docker build -f workers/image/Dockerfile -t $REPO:$SHA -t $REPO:latest .
  post_build:
    commands:
      - docker push $REPO:$SHA
      - docker push $REPO:latest
      - echo "Image pushed:$REPO:$SHA"
cache:
  paths:
    - '/root/.cache/pip/**/*'
```

Triggered by GitHub webhook on push to main with path filter `workers/image/**` or `workers/shared/**`.

Image is BUILD_GENERAL1_LARGE (15 GB RAM, 8 vCPU). Estimated build time: first build ~30 min, subsequent ~10 min with cache. Cost: ~$0.30 per build.

### 9.2 buildspec-video-worker.yml — same shape, different Dockerfile

### 9.3 Worker rollout
After CodeBuild push, force a new ECS service deployment to pull the new image:
```
aws ecs update-service --cluster comfy-cluster --service comfy-image --force-new-deployment
```
Could automate via CodePipeline, but for v1 keep it manual to control rollouts.

---

## 10. Deployment Phases (with cost gates)

### Phase 0: Prep (no cost)
- `npm install -g aws-cdk`
- Create empty git repo locally, init CDK app
- Configure CDK context for account/region

### Phase 1: Bootstrap & free stacks (~$0/mo)
- `cdk bootstrap` (creates S3 bucket + IAM roles for CDK assets, free at idle)
- Deploy NetworkStack (VPC, free)
- Deploy StorageStack (empty buckets, empty DDB tables — pay for use, $0 idle)
- Deploy QueueStack (empty SQS — $0 idle)
- Deploy MonitoringStack with $100/mo budget alert active

**🚧 Cost gate: confirm with user before continuing. AWS Budget alert ARMED.**

### Phase 2: API + Lambda (~$1/mo idle)
- Deploy ApiStack (API Gateway + 4 Lambda functions — $0 at zero traffic)
- Deploy FrontendStack (S3 static — $0)
- Test `POST /models` manually with sample data (cents)

### Phase 3: CodeBuild + first image build (one-off ~$0.50)
- Deploy CIStack (CodeBuild + ECR repos)
- Manually trigger first build of image worker → ECR
- Manually trigger first build of video worker → ECR
- Verify images pushed (~10 GB each in ECR, ~$1/mo storage)

### Phase 4: Compute (still $0 if min=0 and no jobs)
- Deploy ComputeStack (ECS cluster, ASGs with min=0, capacity providers)
- No instances actually launched until a job arrives
- Verify CloudWatch metrics flowing

**🚧 Cost gate: confirm with user before first end-to-end test. From here, every test launches real GPU instances.**

### Phase 5: Smoke test (one image, one video)
- Submit a simple SDXL image workflow → verify image worker spins up, runs, returns result
- Submit a simple Wan workflow → verify video worker spins up, runs, returns result
- Check CloudWatch dashboards: queue depth, latency, cost
- Tear down idle workers (verify scale-to-zero)

### Phase 6: Production handover
- Document operating procedures in README
- Provide kill switch script: `scripts/emergency-shutdown.sh` (sets ASG max=0, drains queues)

---

## 11. Open Risks & Followups

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | ComfyUI `/object_info` synthesis from DDB doesn't perfectly match what frontend expects | High | Test early. Fallback: proxy `/object_info` through to a running worker on demand. |
| 2 | Sage attention build fails on NGC base image due to CUDA version mismatch | Medium | Pin known-good CUDA version in Dockerfile. Test build locally before trusting CodeBuild. |
| 3 | First Docker build exceeds CodeBuild default timeout (60 min) | Low | Increase timeout to 90 min in CDK config. |
| 4 | Spot capacity for g5 family unavailable in us-west-2a | Medium | Configure mixed-instance policy with g5.xlarge → g5.2xlarge → g6e.xlarge. Capacity rebalance. |
| 5 | CivitAI auth-gated models require user's session cookies (not API token) | Low | Document workaround: user uploads model manually via S3 console. Future: support session cookie passthrough. |
| 6 | Public IPv4 charge ($0.005/hr) higher than expected | Low | <$1/mo at projected usage. Acceptable. |
| 7 | API key in browser localStorage = visible in dev tools | Medium | Acceptable for 1-user system. Document that user shouldn't share the URL. Rotate key periodically. |
| 8 | Worker container size (~12-15 GB) → ECR pull on cold start = 50-90s | Medium | Acceptable per <1 min cold start budget for warm-cache case. First-job-after-cold-start budget is 2-4 min including model download. |
| 9 | DDB model catalog out of sync with S3 (e.g., manual S3 upload bypasses DDB) | Medium | Provide `scripts/reconcile-catalog.sh` that scans S3 and reconciles DDB. |
| 10 | CodeBuild costs runaway from frequent rebuilds | Low | $0.30/build × 10 builds/mo = $3/mo. Acceptable. |
| 11 | Stuck job in SQS with high attempt_count → DLQ → silent failure | Low | DLQ alarm fires email. User intervenes. |
| 12 | Workflow contains nodes our `/object_info` doesn't know about | High | Catalog all default ComfyUI nodes + WanVideoWrapper nodes upfront. Test broad workflow imports. |
| 13 | Workers compete for SQS message → wasted GPU time on duplicate work | Low | SQS visibility timeout prevents this. With max=1 image worker, no contention there. |
| 14 | Frontend can't reach Lambda due to CORS misconfiguration | Medium | Test CORS preflight in CI. Document expected origin. |
| 15 | API Gateway 30s timeout < Lambda 15min timeout for downloader | High | Make downloader **async**: API GW returns immediately with `download_id`, Lambda invoked asynchronously via SNS or direct async invoke. |

---

## 12. Cost Projection (revised)

Compared to previous $74/mo estimate, this plan removes:
- Second AZ ($7) ✓
- AMI snapshots ($4) ✓
- Multi-AZ VPC endpoints (replaced with public subnet, $7-14 saved)

But adds:
- Public IPv4 charge: ~$0.65/mo
- CodeBuild builds: ~$3/mo
- Lambda + API GW: <$1/mo

**Revised total estimate: ~$60-65/mo at projected usage.**

| Component | Cost/mo |
|---|---|
| Image worker (g4dn.xlarge spot, max=1, ~65 hrs) | $10 |
| Video worker (g5.xlarge spot, ~65 hrs baseline) | $20 |
| Video burst capacity (sporadic 2nd-3rd worker) | $3 |
| S3 models 1.4 TB (Intelligent-Tiering, mostly cold) | $18 |
| S3 outputs + egress | $7 |
| EBS root cache (active hours) | $3 |
| Public IPv4 | $1 |
| CodeBuild (~10 builds/mo) | $3 |
| ECR (12-15 GB × 2 images) | $2 |
| CloudWatch (logs + alarms + dashboards) | $3 |
| API GW + Lambdas (dispatcher, downloader, catalog, status) | <$1 |
| DDB on-demand (jobs, models, downloads) | <$1 |
| SQS | $0 (within free tier) |
| Frontend S3 | <$0.50 |
| **Total** | **~$72/mo** |

(Slightly higher than $60-65 estimate above; the CodeBuild+ECR add ~$5 that I'd missed initially. **Plan target: ~$72/mo.**)

---

## 13. Open Questions for User Before Implementation

1. ✅ Repo name: `comfyui-aws-queue`
2. ✅ Auth: API key in custom header
3. ✅ Frontend base: vanilla ComfyUI
4. ✅ Local path: current dir
5. ❓ AWS Budget alert email: same as git config (`<user-email>`)?
6. ❓ Initial pinned models for video worker: Wan 2.2 14B I2V Q8 GGUF — is this what you want pre-baked into the catalog manifest? (Workers download these on every fresh boot.)
7. ❓ CivitAI API token: do you have one? We'll store it in Secrets Manager. Required for some gated models. v1 can work without one for public models.

---

## 14. Build Order (file-by-file, after sign-off)

1. `infra/` skeleton: package.json, cdk.json, bin/, lib/config.ts
2. `lib/stacks/network.ts` + `lib/stacks/storage.ts` (free, deployable first)
3. `lib/stacks/queue.ts` + `lib/stacks/monitoring.ts` (with budget alarm)
4. `services/catalog/` Lambda + `lib/stacks/api.ts` (catalog only first)
5. `frontend/models/` standalone page (test models flow without compute)
6. `services/downloader/` (test CivitAI download flow)
7. `services/dispatcher/` (test routing without workers running)
8. `workers/shared/` Python code (cache_manager, spot_handler, worker)
9. `workers/image/Dockerfile`, `workers/video/Dockerfile`
10. `codebuild/` buildspecs + `lib/stacks/ci.ts`
11. First worker container build (CodeBuild)
12. `lib/stacks/compute.ts` (ECS, ASGs)
13. Frontend ComfyUI bundle integration
14. End-to-end smoke test

Each step is its own commit. Code-reviewer agent runs after big chunks (after step 4, after step 8, after step 13).

---

## 15. Grilling Findings (Self-Review as Distinguished Engineer)

Verified versions (from research run 2026-05-10):
- ComfyUI **v0.20.1** (pin to tag, not `main`)
- SageAttention **2.2.0** (`pip install sageattention==2.2.0 --no-build-isolation`)
- PyTorch **2.7.0** stable (cu128 channel for video, cu126 OK for image-only Ampere/Ada)
- NGC base: `nvcr.io/nvidia/pytorch:26.04-py3` (CUDA 13.2.1, ~12 GB), or fallback to `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` if NGC ABI conflicts arise
- WanVideoWrapper: pin by commit SHA (no releases)
- AWS CDK **2.253.1**
- ECS GPU AMI: **AL2023 only** (AL2 GPU AMI EOL 2026-06-30, ~7 weeks away). Use SSM `/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended`
- CivitAI: Bearer token in Authorization header (never query-param), 429 backoff design

### CRITICAL findings

**C1. T4 GPU does NOT support SageAttention 2.x.**
The image worker on g4dn.xlarge has a Tesla T4 (Turing, sm_75). SageAttention 2.x requires Ampere or newer (sm_80+). User explicitly asked for sage attention enabled — but it physically can't run on T4. Three options:
- **(a)** Use SageAttention 1.x on image worker (older, less optimized but T4-compatible). Different package version per Dockerfile.
- **(b)** Disable sage entirely on image worker. Use xformers attention. SDXL on T4 with xformers is ~12-15s/image — well under our 36s/image budget.
- **(c)** Switch image worker to g5.xlarge (A10G, sm_86, supports sage 2.x). Cost: +$10/mo. Unifies fleet.
- **Recommended: (b).** xformers on image worker, sage 2.2 on video worker. SDXL doesn't benefit much from sage anyway. Lowest cost, simplest. Document the why in worker README.

**C2. `/object_info` synthesis from DDB is high-risk.**
ComfyUI's frontend uses `/object_info` to populate node menus. It's a complex JSON describing every node's INPUT_TYPES (with type info, defaults, validators). Custom nodes (like WanVideoWrapper) define their own. Synthesizing this from a DDB model catalog only covers model dropdowns, not node definitions. We can't synthesize — we need a real `/object_info` from a running ComfyUI.
- **Fix: dispatcher caches the LAST KNOWN /object_info from a worker** (workers POST it on startup). Frontend always gets a real /object_info, just possibly stale. When user uploads a model via downloader → also write the new model into the cached /object_info's `LoraLoader.required.lora_name[0]` (or appropriate field per type). Or: always proxy /object_info from a live worker, accepting that if no worker is up, we keep one warm OR briefly spin up an instance just to fetch it.
- **Recommended:** workers POST their /object_info to the dispatcher on startup. Dispatcher caches it in DDB. Catalog Lambda merges new models into the cached /object_info's relevant fields when models are added. Frontend reads from dispatcher's cached /object_info.

**C3. Workflow JSON inspection for routing is fragile.**
A `class_type` string-match is brittle. Custom nodes have arbitrary names. A workflow with a Wan node renamed by user could mis-route.
- **Fix:** Frontend submits workflow with explicit `type: "image" | "video"` parameter chosen by user. Dispatcher uses that as primary signal, with class_type sniffing as a verification/fallback.
- This requires modifying the frontend's submit button to pick a queue. Acceptable — we own the frontend bundle.

**C4. Downloader Lambda must be async — API Gateway has 30s integration timeout.**
v1 plan called downloader synchronously from API GW. Won't work — downloads take 1-15 min.
- **Fix:** API GW route `POST /models/download` invokes a thin "kick-off" Lambda (returns `download_id` in <1s, writes "queued" record to DDB), which async-invokes the actual downloader Lambda. Frontend polls `GET /downloads/{id}` for progress.

**C5. Spot interruption handler vs heartbeat race.**
If heartbeat thread extends visibility timeout 1 second before spot termination, `change_message_visibility(0)` from the spot handler should still work (it's idempotent). Verify the SDK contract: setting visibility lower than current is allowed. (It is — but document.)

### MAJOR findings

**M1. Cold-start time underestimated.**
Realistic cold start = EC2 boot (30s) + ECR pull of 12-15 GB image (~80-100s at typical regional bandwidth) + container start (30s) + ComfyUI process startup with sage attention init (~30-60s) + first model load from S3 (e.g., Wan 18 GB Q8 = ~120s) = **5-7 minutes** for a fully cold first job, not the 2-4 min I quoted earlier. Pinned-set parallel download helps (workers warm during ECR pull).
- **Fix:** Update cold-start budget in user-facing docs. Mitigation: pre-warm option via scheduled `min=1` for predictable session windows (already noted as optional knob).

**M2. ComfyUI subprocess crash needs supervision.**
If ComfyUI dies (OOM, segfault, model load failure), worker loop will block on `requests.post('/prompt')` and time out. No restart logic in v1.
- **Fix:** Wrap ComfyUI in a `subprocess.Popen` with a watchdog thread. On exit, restart up to N times then mark instance unhealthy (let ASG terminate it). Job currently in-flight is requeued via SQS visibility timeout.

**M3. Routing-by-frontend-button means frontend must know about both queues.**
Implies the standard ComfyUI "Queue Prompt" button needs to be replaced or augmented. Two buttons ("Queue Image" / "Queue Video") or a toggle. Modifying ComfyUI's frontend requires touching its source.
- **Fix:** Patch the queue-prompt button to be a dropdown: "Queue (Image)" / "Queue (Video)". Patch is small (~20 lines of Vue/TS). Alternative: rely on workflow inspection only and accept the brittleness — but C3 mitigation is preferred.

**M4. CodeBuild webhook on GitHub requires CodeStar Connection.**
Setting up a webhook from a GitHub repo to AWS CodeBuild requires either personal access token (legacy, deprecated) or AWS CodeStar Connection (newer, requires manual auth handshake in console).
- **Fix:** v1 ships without auto-trigger. Manual `aws codebuild start-build` via local script (`scripts/trigger-build.sh`). Add CodeStar Connection later as enhancement.

**M5. Frontend bucket public-read policy carries risk.**
Public S3 buckets are a common breach pattern. We need:
- Public-read on the frontend prefix only, never on outputs/models/uploads buckets
- S3 Block Public Access EXPLICITLY disabled only on the frontend bucket (other buckets keep BPA on)
- Bucket policy that grants `s3:GetObject` only to `principal: *` for `s3://bucket/index.html`, `*.js`, `*.css`, etc.
- **Fix:** Document and review carefully. Code-reviewer must verify before push.

**M6. Worker has IAM access to S3 models bucket — read-only enforced?**
If a malicious workflow attempts to write to models bucket via custom node, that's bad. v1 IAM policy in plan said "S3 read models bucket, S3 write outputs bucket" — needs to be enforced as separate IAM statements with explicit `s3:GetObject, s3:ListBucket` (no PutObject) on models, vs `s3:PutObject` on outputs.
- **Fix:** Tight IAM policy. Easy to get wrong. Code-reviewer must verify.

**M7. No DLQ replay strategy.**
If a job DLQs (3 failed receives), it's stuck forever unless user intervenes.
- **Fix:** Provide `scripts/replay-dlq.sh` that re-sends DLQ messages to main queue. Document.

**M8. Containers are big — first ECR push from CodeBuild.**
12-15 GB compressed images means CodeBuild's first push to ECR is 12-15 GB upload (~5 min) + initial build is 30+ min. CodeBuild default timeout is 60 min. With buildx/buildkit cache, subsequent builds are <10 min.
- **Fix:** Set CodeBuild timeout to 90 min explicitly. Use `docker buildx` with `--cache-to type=registry,ref=$REPO:cache,mode=max` to cache layers in ECR for next build.

### MINOR findings

**Mi1.** SSM parameter for AL2023 GPU AMI: `/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended` (not AL2). Confirmed.
**Mi2.** CDK `AsgCapacityProvider` is correct (not the buggy new `ManagedInstancesCapacityProvider`).
**Mi3.** CivitAI dedupe at model-version-id level — add unique constraint on DDB.
**Mi4.** CloudWatch logs retention: set to 14 days everywhere to bound cost.
**Mi5.** DDB Point-in-Time Recovery: enable on `comfy-models` and `comfy-jobs` (free for first 35 days of backups).
**Mi6.** S3 versioning on `comfy-outputs`: enable (cheap, protects against accidental overwrite).
**Mi7.** AWS Budget at $100/mo — make it an SNS topic that emails AND triggers a Lambda that can shut down ASGs at 100% as a kill switch (opt-in).
**Mi8.** Sample workflows shipped under `examples/` so user can smoke-test without building one.
**Mi9.** API GW caching on `/object_info` (TTL 5 min, $0.02/mo for 0.5 GB cache size — but cheaper to cache in Lambda memory).
**Mi10.** Use SQS message attributes (not just body) for `job_id` and `type` so worker can filter without parsing JSON.

---

## 16. Plan Iterations (Applied to v2)

Based on grilling findings, the following changes are now part of the plan:

### 16.1 Sage Attention strategy (resolves C1)
- **Image worker Dockerfile:** uses `xformers` (no sage). Documented in worker README.
- **Video worker Dockerfile:** uses `sageattention==2.2.0`, launched with ComfyUI flag `--use-sage-attention`. Pinned to GGUF Q8 models to avoid the FP8+Sage black-frames bug.
- Update DRAFT Dockerfile in section 5 — image variant drops sage install line, adds `pip install xformers`.

### 16.2 `/object_info` strategy (resolves C2)
- Workers POST `/object_info` to dispatcher's `/internal/object_info` endpoint on startup.
- Dispatcher writes to a new DDB table `comfy-object-info` (PK=`fleet`, attr=`object_info_json`, `updated_at`).
- Dispatcher's `GET /object_info` returns merged image+video object_info, with model lists in relevant fields swapped for the live DDB catalog (e.g., `CheckpointLoaderSimple.required.ckpt_name[0]` becomes the list of all checkpoint model names from the catalog).
- Catalog Lambda invalidates the dispatcher's in-memory cache via Lambda invocation when a model is added/deleted.

### 16.3 Routing strategy (resolves C3 + M3)
- Frontend submits workflows with explicit `type: "image" | "video"` field (chosen by user).
- Modify ComfyUI's queue button to "Queue (Image) ▾ / Queue (Video)" dropdown.
- Dispatcher uses `type` as primary signal. If `type` missing, falls back to class_type sniffing.
- Workflow router code includes a comprehensive list of video node patterns from ComfyUI core + WanVideoWrapper.

### 16.4 Async downloader (resolves C4)
- New Lambda function: `downloader-kickoff` (small, fast). Handles `POST /models/download` API GW request.
- Returns `{download_id}` in <1s, writes `status: queued` record to DDB.
- Async-invokes `downloader-worker` Lambda (the long-running one).
- Frontend polls `GET /downloads/{id}` for progress.

### 16.5 ComfyUI process supervision (resolves M2)
- Worker wraps ComfyUI in a `subprocess.Popen` with watchdog thread.
- Restart up to 3 times on crash before marking instance unhealthy.
- Unhealthy → exit container → ECS replaces → ASG provides new instance if available.

### 16.6 IAM tightening (resolves M6)
- Worker IAM policy explicit:
  - `s3:GetObject, s3:ListBucket` on `models-bucket/*` ONLY (no PutObject)
  - `s3:PutObject, s3:GetObject` on `outputs-bucket/*` (write own outputs, read for verification)
  - `sqs:ReceiveMessage, DeleteMessage, ChangeMessageVisibility` on its own queue ONLY
  - `dynamodb:UpdateItem, GetItem` on `comfy-jobs` (no PutItem — dispatcher creates jobs, worker only updates)
  - `dynamodb:GetItem` on `comfy-models` (read-only for cache lookups)

### 16.7 Frontend bucket safety (resolves M5)
- Two separate stacks: `FrontendBucketStack` (public-read, frontend only) vs other storage (BPA enabled).
- Bucket policy explicit: `s3:GetObject` to principal `*` on `arn:aws:s3:::frontend-bucket/*` only.
- Code-reviewer agent will verify this before push.

### 16.8 Cold-start budget update (resolves M1)
- Updated docs to say:
  - Warm cache hit: <10s overhead
  - Cold cache, warm worker: +50s (image) / +2-3 min (video) for model fetch
  - Fully cold (worker + cache): **5-7 min for video**, ~3 min for image
- Optional `min=1` scheduled warming knob documented but not deployed in v1.

### 16.9 CodeBuild config update (resolves M8)
- BuildSpec uses `docker buildx` with registry-cached layers.
- CodeBuild project timeout: 90 min.
- BUILD_GENERAL1_LARGE compute (15 GB RAM, 8 vCPU).
- Manual trigger via `scripts/trigger-build.sh` for v1; CodeStar Connection + GitHub webhook later.

### 16.10 DLQ replay (resolves M7)
- `scripts/replay-dlq.sh` script that drains DLQ to main queue with confirmation prompt.

### 16.11 Other minor fixes (Mi1-Mi10)
- All applied in respective stack code.

---

## 17. Final Cost Re-estimate (v2)

No material change. Image worker stays on g4dn (xformers, no sage 2.x), video on g5 (sage 2.x). All other tweaks are operational. **~$72/mo.**

---

## 18. (Open Questions section moved to section 21 after final iterations)

---

## 19. Independent Code-Reviewer Findings

After v2 was self-grilled, an independent code-reviewer agent reviewed the plan with focus on issues NOT already caught. It found 5 Critical issues, 8 Major, and 7 Minor.

### CRITICAL (block-merge):

**N1. CloudWatch Logs ingestion is the most likely path to a $500+/mo bill.**
ComfyUI is chatty (per-step diffusion progress, tqdm bars, sage init logs). Two video workers stuck running 24/7 (e.g., from a scaling bug) at 20-50 GB/day of logs × $0.50/GB ingestion = $300-750/mo from logs alone. Budget alarm fires *after* the spend, on a 24h delay.

**N2. ECS scaling overshoot will burn money on bursts.**
Target backlog of 10 + max=5 + 1-min metric latency means a 20-job submission can spawn 4-5 instances simultaneously while the first is still pulling its container. $2/hr just to drain a backlog one instance could handle in 90 min.

**N3. Secrets Manager is referenced but never deployed.**
The downloader Lambda reads from `secretsmanager:civitai/api-token` but no stack creates that secret. First CivitAI download throws `ResourceNotFoundException`.

**N4. Public GitHub repo will leak account ID, bucket names, and CDK outputs.**
Account IDs aren't strictly secret but enable role-confusion attacks. Synthesized CFN templates and `cdk.context.json` are accidentally-committable. API key, CivitAI token, API GW invoke URL must never land in git.

**N5. API key in browser config.js served from public S3 = world-readable API key.**
Anyone who finds the frontend URL gets the API key. Usage plan caps (1000/day) won't stop someone queueing 1000 videos and burning $200 of GPU before throttling kicks in. The "don't share the URL" mitigation is too weak.

### MAJOR:

**N6.** Service quota limits on personal account: G/VT spot vCPU defaults to **0**, must request increase via support ticket (24-72 hr).
**N7.** No cost allocation tags = can't break down a budget breach by component.
**N8.** ECR pull bandwidth limit on burst — 5 simultaneous 13 GB pulls may throttle.
**N9.** SQS visibility (30 min) vs heartbeat (5 min, +15 min) has off-by-one window — possible duplicate execution.
**N10.** AL2023 GPU AMI driver regression has no rollback. SSM `:latest` auto-updates → silent break.
**N11.** ComfyUI custom nodes (esp. WanVideoWrapper) write outputs to hardcoded paths; plan assumes a known location.
**N12.** Force-new-deployment on max=1 service hangs (default `minimumHealthyPercent=100` conflicts with no-headroom).
**N13.** Unbounded DDB write spend possible from runaway progress-update or heartbeat loops.

### MINOR:
**N14.** No DDB cross-table backup (PITR is gone if table deleted).
**N15.** DeleteObject IAM perms not specified for outputs (probably correct to omit, but document).
**N16.** ECR lifecycle won't fire because every image is tagged.
**N17.** Internal redundancy: section 11 risk #15 duplicates section 15 C4.
**N18.** README must warn against switching ComfyUI to `--listen 0.0.0.0` (workers have public IPs).
**N19.** Deploy script must assert correct AWS account ID before any deploy.
**N20.** Severity reassessment: original C5 (visibility race) is actually Minor; original M4 (CodeBuild webhook) is Minor.

### Reviewer's overall take:
> "The v2 plan is solid... what it misses is the long tail of 'small things that compound into a big bill or a security incident.' Address N1-N5 before writing code; the rest can be issues filed against the repo. The kill-switch Lambda from Mi7 should be **mandatory not opt-in**."

---

## 20. Plan Iterations v3 (Applied)

### 20.1 Log spend control (resolves N1)

**MANDATORY.** The single largest runaway-cost vector identified.

- **Worker logging defaults**: Python logging at WARNING for all libs; ComfyUI launched with `--quiet` flag (or stdout filtered through `tee | grep -v -E '^[[:space:]]*[0-9]+%'` to drop tqdm progress lines).
- **awslogs driver options on ECS task definition**:
  ```
  awslogs-multiline-pattern: '^[A-Z]+|^Traceback'
  awslogs-datetime-format: '%Y-%m-%d %H:%M:%S'
  ```
  These prevent each tqdm `\r` from becoming its own log event.
- **CloudWatch metric filter alarm** on each log group: alarm if `IncomingBytes > 5 GB / 1 hour`. SNS → email + auto-disable scaling (set ASG max=0).
- **Log group retention**: 7 days (was 14). Cuts storage cost in half.

### 20.2 ECS scaling tuning (resolves N2)

- **Target backlog per task**: 25 (was 10). At 1 user, this means up to 25 jobs queue before scale-out fires.
- **Scale-out cooldown**: 300s (was 60s). Lets first instance finish ECR pull + warm pinned models before peers spawn.
- **Scale-in cooldown**: 900s (15 min) — keeps workers warm through short pauses.
- **Max instances**: video fleet max=3 (was 5). 3 × g5.xlarge × 30/hr each = 90 vid/hr capacity, 4.5× the steady target. Plenty.

### 20.3 Add SecretsStack (resolves N3)

- New stack `SecretsStack`. Creates `civitai/api-token` secret with placeholder. IAM grants Downloader Lambda `secretsmanager:GetSecretValue` on that ARN only.
- Documented Phase-1 manual step: `aws secretsmanager put-secret-value --secret-id civitai/api-token --secret-string '{"token":"YOUR_TOKEN"}'`
- Cost: $0.40/mo per secret. Added to cost table.

### 20.4 Public-repo safety hardening (resolves N4)

- **Use CDK tokens**: bucket names use `Aws.ACCOUNT_ID.toString()` not literal substitution. Resolved at deploy time, never in source.
- **`.gitignore`** includes: `cdk.out/`, `cdk.context.json`, `*.template.json`, `.env*`, `config.local.js`, `frontend/dist/config.js` (deploy artifact, not source).
- **Pre-commit hook** runs `gitleaks` against staged files. Repo includes `.gitleaks.toml` with custom patterns for AWS account IDs.
- **CI check** (.github/workflows/ci.yml) runs `gitleaks detect --no-git` on every PR.
- **Sanitization checklist** in `scripts/pre-publish-check.sh` runs before any push to public repo.

### 20.5 API security uplift (resolves N5)

The user picked "API key in custom header" before this risk surfaced. Three pragmatic options:

**Option A (chosen for v3): API key + per-IP throttle + aggressive kill-switch**
- API GW usage plan: per-IP throttle 5 req/sec, 50 burst.
- CloudWatch alarm on `Count` metric > 100/5min on the API → SNS → kill-switch Lambda → IMMEDIATELY disables API key + sets ASG max=0 + sends email.
- $50/mo budget alert (was $100) for earlier warning.
- Document: API URL = treat as semi-secret, never share publicly.
- **Cost: $0 added.**

**Option B (alternative): Cognito user pool**
- Single user pool, JWT in Authorization header. Stronger but adds setup complexity.
- Not chosen for v3 to respect user's stated preference for simplicity. Documented as future enhancement.

**Mandatory Kill-Switch Lambda** (was Mi7 opt-in, now required):
- Triggered by SNS topic from any of: budget breach, log spend alarm, API rate alarm, DLQ depth alarm.
- Actions: set ASG max=0 on both fleets; disable API key in usage plan; send email.
- One-line manual recovery: `scripts/restore-after-killswitch.sh` re-enables.

### 20.6 Service-quota pre-flight (resolves N6)

- `scripts/preflight-quotas.sh` checks expected quotas, prints required increases.
- Documented in README **before** deploy instructions: "If you haven't already, request quota increases now (24-72 hr lead time)":
  - L-3819A6DF (Spot vCPU for G/VT): request ≥40
  - (Other ECS/EBS quotas as needed)
- `bootstrap.sh` refuses to proceed if quotas not met.

### 20.7 Cost allocation tags (resolves N7)

CDK app applies global tags via `Tags.of(app)`:
- `Project=comfyui-aws-queue`
- `Component=image-worker|video-worker|api|storage|ci|frontend|monitoring`
- `Environment=prod`
- `ManagedBy=cdk`

README documents the manual step to activate cost allocation tags in Billing console (~24h to populate).

### 20.8 ECR pull risk (resolves N8)

- v1: monitor only. CloudWatch metric filter on ECR pull errors. Alarm if frequency increases.
- If observed in practice: add ECR pull-through cache or revisit AMI baking trade-off.

### 20.9 Visibility timeout / heartbeat (resolves N9)

- Initial visibility timeout: **45 min** (was 30 min).
- Heartbeat interval: **60s** (was 5 min). Each heartbeat extends visibility by 15 min.
- Failed jobs (validation errors, missing models) deleted from queue immediately, not redelivered.
- Idempotency check: worker reads DDB job status before starting; if already `running` by another worker (from prior failed delivery), exits cleanly.

### 20.10 AMI rollback safety (resolves N10)

- SSM parameter pinned to a specific version: e.g., `/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended:42` (look up at deploy time, write to CDK context).
- Weekly canary Lambda (~$0.50/wk): launches one g4dn instance, runs `nvidia-smi`, `python -c "import sageattention"`, terminates. Pages on failure.
- Documented manual rollback procedure.

### 20.11 Worker output handling (resolves N11)

- Worker creates per-job scratch dir: `/tmp/comfy-job-{id}/`
- Symlinks `/opt/comfy/output` → scratch dir before starting job
- After job completes, globs scratch dir for any `.png|.jpg|.webp|.mp4|.gif` and uploads all
- Container build includes a "smoke test workflow" that runs SDXL "1 black pixel" at build time to verify nodes load

### 20.12 ECS deployment without downtime (resolves N12)

- Set `minimumHealthyPercent=0` and `maximumPercent=100` on services
- Acceptable: 1-user system, brief outage during deploy is fine
- Alternative documented: just terminate the running instance and let ASG recreate

### 20.13 DDB write throttling (resolves N13)

- Downloader progress updates: throttle to once per 256 MB OR once per 5%, whichever fires less often
- Heartbeat: 60s interval, no faster
- CloudWatch alarm on DDB consumed write capacity > 1000/min

### 20.14 Other minor fixes (N14-N20)
- DDB weekly export to S3 (~$1/mo for catalog)
- Document Delete on outputs is intentionally absent
- ECR lifecycle: keep last 10 by `imagePushedAt`, regardless of tag
- Removed redundant Risk #15 from section 11
- README warns against `--listen 0.0.0.0`
- `bootstrap.sh` and `deploy-stacks.sh` assert expected account ID (set via `EXPECTED_ACCOUNT_ID` env var, never committed) before any AWS call
- Severity adjustments: Original C5 → Minor. Original M4 → Minor.

---

## 21. Final Cost Re-estimate (v3)

Changes from v2:
- + Secrets Manager: $0.40/mo
- + DDB weekly export: ~$1/mo
- + Canary Lambda: ~$2/mo
- + CloudWatch metric filter alarms: <$1/mo

| Component | Cost/mo |
|---|---|
| Image worker (g4dn.xlarge spot, max=1, ~65 hrs) | $10 |
| Video worker (g5.xlarge spot, ~65 hrs baseline) | $20 |
| Video burst (now max=3, less likely to fully burn) | $2 |
| S3 models 1.4 TB (Intelligent-Tiering) | $18 |
| S3 outputs + egress | $7 |
| EBS root cache (active hours) | $3 |
| Public IPv4 | $1 |
| CodeBuild (~10 builds/mo) | $3 |
| ECR (12-15 GB × 2 images) | $2 |
| CloudWatch (logs at 7d retention + filters) | $4 |
| Secrets Manager (1 secret) | $0.40 |
| DDB on-demand + weekly export | $2 |
| Canary Lambda + kill-switch Lambda | $2 |
| API GW + Lambdas | <$1 |
| Frontend S3 | <$0.50 |
| **Total** | **~$76/mo** |

Slight uptick from $72/mo (v2) to **~$76/mo (v3)**, driven by added safety controls. Still well under the $100/mo budget alert.

---

## 22. Final Open Questions Before Implementation

(Section number incremented to 22 since 18 became 19/20.)

1. ✅ Repo name: `comfyui-aws-queue`
2. ✅ Auth: API key in custom header + per-IP throttle + kill-switch (v3 hardening)
3. ✅ Frontend base: vanilla ComfyUI v0.20.1
4. ✅ Local path: current dir
5. ❓ **Sage attention strategy:** Image worker on g4dn uses **xformers** (no sage 2.x — T4 incompatibility). Video worker on g5 uses sage 2.2.0. OK?
6. ❓ **Auth uplift:** API key + per-IP throttle + mandatory kill-switch Lambda (Option A) vs Cognito user pool (Option B)? v3 picks A for simplicity. Confirm or switch to B?
7. ❓ **Budget alert email:** (configured via deploy parameter, not committed)
8. ❓ **Budget alert threshold:** $50/mo (v3 picks this, gives early warning) or $100/mo?
9. ❓ **Initial pinned model set for video worker:** Wan 2.2 14B I2V Q8 GGUF + Lightning LoRA + WanVideo VAE? Or empty catalog at launch?
10. ❓ **CivitAI API token:** do you have one? Goes into Secrets Manager. Optional for public models.
11. ❓ **Service quota request:** have you already requested G/VT spot vCPU increase ≥40 in us-west-2? If not, do this NOW (24-72h lead time) — Phase 4 deploy will fail without it.

---

## 22b. v3 Final Decisions (post user sign-off)

- ✅ Auth: **Cognito user pool** (Option B). Replaces API key + custom header.
- ✅ Budget: $100/mo email alert + kill-switch at 100%.
- ✅ CivitAI token: Secrets Manager will be created with placeholder; user populates after deploy.
- ✅ Service quota: user chose "try and see" — Phase 4 will fail until quota raised, but we proceed anyway.
- ✅ All other Section 22 questions resolved with v3 defaults.

### v3.1 cost re-estimate (with verified us-west-2 spot prices, May 2026)

Spot prices today (us-west-2, snapshot 2026-05-10):
- g4dn.xlarge: ~$0.21/hr
- g5.xlarge: ~$0.63/hr
- g6e.xlarge: ~$1.18/hr

Important: spot prices fluctuate. These may be elevated due to current GenAI demand.

| Component | Cost/mo (revised) |
|---|---|
| Image worker (g4dn.xlarge spot, ~65 hrs) | $14 |
| Video worker (g5.xlarge spot, ~65 hrs steady) | $41 |
| Video burst (sporadic 2nd-3rd worker, 10 hrs) | $7 |
| S3 models 1.4 TB (Intelligent-Tiering, mostly cold) | $18 |
| S3 outputs + egress | $7 |
| EBS root cache | $3 |
| Public IPv4 | $1 |
| CodeBuild | $3 |
| ECR | $2 |
| CloudWatch | $4 |
| Secrets Manager | $0.40 |
| DDB + weekly export | $2 |
| Canary + kill-switch Lambdas | $2 |
| Cognito (free under 50K MAU) | $0 |
| API GW + Lambdas | <$1 |
| Frontend S3 | <$0.50 |
| **Total** | **~$105/mo** |

**Within $100/mo budget envelope but close to it.** Risk: if spot prices spike further or video usage exceeds 65 hrs/mo, easy to breach. Mitigations:
- Aggressive scale-in cooldown (already in v3)
- Kill-switch at $100 budget (already in v3)
- Consider us-east-1 in future if cost-prohibitive (lower spot historically, but user picked us-west-2)

### Cognito design note (replacing API key)

- Single user pool: `comfy-users`
- Single user, created manually post-deploy (no self-signup)
- App client: public, no client secret, ALLOW_USER_PASSWORD_AUTH flow
- Frontend: `amazon-cognito-identity-js` (vanilla JS, no Amplify dependency) for sign-in flow
- API GW: Cognito User Pool authorizer on all routes
- ID token (not access token) used as Authorization Bearer header
- Token TTL: 1 hour (default), refresh token 30 days

### v3.1 cost-related plan adjustments

The kill-switch Lambda and budget alarm are now even more important given the slimmer headroom. Wire-up of triggers:
- Budget at 50% → email only
- Budget at 80% → email + ASG max=0 (preventive)
- Budget at 100% → email + ASG max=0 + Cognito user disabled (kill-switch)

---

## 23. Implementation Readiness Assessment

| Aspect | Status |
|---|---|
| Architectural soundness | ✅ Vetted (self + reviewer) |
| Cost controls | ✅ Multiple layers (budget, log alarms, kill-switch, throttles) |
| Security posture | ✅ Acceptable for personal-project (with v3 hardening) |
| Public-repo safety | ✅ Sanitization checks in place |
| Spot interruption recovery | ✅ Tested-by-design |
| Operational runbook | ⚠️ Thin — README needs to cover replay-DLQ, restore-after-killswitch, manual AMI rollback |
| Open external dependencies | ⚠️ Service quota request must precede deploy (lead time) |

**Recommendation:** plan is ready for implementation pending answers to section 22 open questions and confirmation that service quota requests are filed.

---
