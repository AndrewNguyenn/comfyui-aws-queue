/**
 * Centralized configuration for the comfyui-aws-queue CDK app.
 *
 * All values that vary by environment (region, instance types, sizing) live here.
 * Account ID is resolved at deploy time via cdk.Aws.ACCOUNT_ID — never hardcoded.
 */
import { Duration } from 'aws-cdk-lib';

/**
 * SSM parameter that holds the current golden-AMI id for the image fleet.
 * Written by the EC2 Image Builder pipeline (imagebuilder.ts) on each bake;
 * read by the image-fleet launch template (compute.ts) when the golden AMI is
 * activated via `-c useGoldenAmi=true`.
 */
export const GOLDEN_AMI_PARAM = '/comfy/image/golden-ami-id';

/**
 * Name of the EC2 Image Builder pipeline that bakes the golden AMI. Fixed so
 * other stacks can grant on / trigger its ARN without a cross-stack reference.
 * Defined here (not imagebuilder.ts) to avoid a circular import with ci.ts.
 */
export const GOLDEN_PIPELINE_NAME = 'comfy-image-golden-pipeline';

export interface AppConfig {
  readonly projectName: string;
  readonly region: string;
  readonly availabilityZone: string;
  // Extra AZs to add as public subnets so the ASGs have more spot pools to
  // pull from. The primary `availabilityZone` is always included; these are
  // appended. Free in our setup (no NAT GW, no interface endpoints).
  readonly additionalAvailabilityZones: string[];
  readonly tags: Record<string, string>;

  readonly fleets: {
    readonly image: FleetConfig;
    readonly video: FleetConfig;
    readonly minimax: FleetConfig;
  };

  readonly storage: {
    readonly modelsBucketPrefix: string;
    readonly outputsBucketPrefix: string;
    readonly uploadsBucketPrefix: string;
    readonly frontendBucketPrefix: string;
    readonly failedWorkflowsBucketPrefix: string;
    readonly outputsLifecycleDays: number;
    readonly uploadsLifecycleDays: number;
  };

  readonly queues: {
    readonly imageVisibilityTimeout: Duration;
    readonly videoVisibilityTimeout: Duration;
    readonly minimaxVisibilityTimeout: Duration;
    readonly messageRetention: Duration;
    readonly maxReceiveCount: number;
  };

  readonly scaling: {
    readonly imageMin: number;
    readonly imageMax: number;
    readonly videoMin: number;
    readonly videoMax: number;
    readonly minimaxMin: number;
    readonly minimaxMax: number;
    readonly targetBacklogPerTask: number;
  };

  readonly cost: {
    readonly budgetAmountUsd: number;
    readonly budgetEmailParameter: string; // SSM parameter name, not literal email
  };

  readonly logs: {
    readonly retentionDays: number;
    readonly ingestionAlarmGbPerHour: number;
  };

  readonly api: {
    readonly throttleRatePerSec: number;
    readonly throttleBurst: number;
    readonly callRateAlarmThreshold: number;
  };
}

export type FleetName = 'image' | 'video' | 'minimax';

export interface FleetConfig {
  readonly fleetName: FleetName;
  readonly primaryInstanceType: string;
  readonly fallbackInstanceTypes: readonly string[];
  readonly rootVolumeGb: number;
  // gp3 root-volume performance. The container image (~26 GB uncompressed for
  // image, similar for video) is pulled + extracted to /var/lib/docker on this
  // root volume on every cold boot, so its write throughput gates cold-start
  // time. Per-fleet so we can tune the image fleet without touching video.
  readonly rootVolumeIops: number;
  readonly rootVolumeThroughputMbps: number;
  // Golden-AMI only (image fleet). When the fleet boots from the golden AMI
  // (root restored from the worker-image snapshot), provision this EBS init
  // rate (MiB/s, 100–300) so the snapshot hydrates eagerly instead of
  // lazy-loading at ~26 MB/s (measured: lazy-load never reached ComfyUI-ready
  // in the 300s grace; 300 MiB/s → ~32s). Ignored unless `-c useGoldenAmi=true`.
  // See compute.ts and imagebuilder.ts.
  readonly volumeInitializationRateMiBs?: number;
  // (cacheGb removed — model cache lives on the included NVMe instance store,
  // mounted by workers/image/entrypoint.sh. Was never wired into a separate
  // EBS volume in compute.ts anyway.)
}

/**
 * The single configuration used by the app. If multiple environments are needed
 * later, factor this into a function returning per-env config.
 */
export const APP_CONFIG: AppConfig = {
  projectName: 'comfyui-aws-queue',
  region: 'us-east-1',
  availabilityZone: 'us-east-1a',
  // Span all five GPU-capable us-east-1 AZs for the widest spot-pool spread.
  // a/b/c/d/f all offer the g4dn and g5 sizes the image fleet uses, so pool
  // depth directly drives how long a job waits during a spot drought;
  // g6e.xlarge (video fallback) is offered in a/b/c/d — capacity-optimized
  // simply skips f for that type. 1e is excluded (no GPU capacity). Extra
  // public subnets are free here — no NAT GW, no interface endpoints.
  additionalAvailabilityZones: ['us-east-1b', 'us-east-1c', 'us-east-1d', 'us-east-1f'],
  tags: {
    Project: 'comfyui-aws-queue',
    Environment: 'prod',
    ManagedBy: 'cdk',
  },

  fleets: {
    image: {
      fleetName: 'image',
      // HISTORY (superseded — fleet was g5.xlarge-only until 2026-07-12; see the
      // dated blocks below for the current mix):
      // We trialled a g4dn (T4) lowest-price config for cost and it BACKFIRED for
      // this SDXL + ESRGAN-upscale workload (measured 2026-06-08):
      //   - SPEED: T4 ran ~192 s/image vs A10G ~61 s — ~3.2x slower. At g4dn.xlarge
      //     spot (~$0.24/hr) vs g5.xlarge (~$0.48/hr) that's ~$12.95 vs $8.11 per
      //     1000 images — the T4 is ~60% MORE expensive per image despite the
      //     lower hourly rate (half price can't beat 3x slower).
      //   - STABILITY: the T4's 16 GB VRAM forced ComfyUI to offload ~14 GB of
      //     weights to pinned CPU RAM; on the 16 GB-RAM g4dn.xlarge that OOM-killed
      //     ComfyUI (jobs failed "[Errno 111] Connection refused" to :8188). The
      //     32 GB g4dn.2xlarge survived, but is still slow + pricier per image.
      //   - xformers efficient attention doesn't even load on the image (build
      //     mismatch vs the NGC torch), worsening VRAM pressure.
      // So: A10G primary. The T4 rejection above does NOT apply to g6.xlarge
      // (L4): L4 has 24 GB VRAM (same as A10G — no OOM/offload), 4 vCPU + 16 GB
      // RAM (identical sizing), Ada/sm_89 with native bf16, and is offered in all
      // 5 AZs we span. So g6.xlarge is added as a spot FALLBACK to survive g5
      // droughts (live UnfulfillableCapacity walls observed 2026-06-08).
      // capacity-optimized-PRIORITIZED keeps g5.xlarge first; g6 fires only on a
      // g5 shortfall, so the common case is unchanged. De-risked: the video fleet
      // already runs Ada (g6e/L40S, sm_89) on this stack; the golden AMI's AL2023
      // GPU driver + NGC fat binaries cover both sm_86 and sm_89. L4 is ~15-40%
      // slower than A10G on this workload but completes — strictly better than no
      // worker during a drought. Still NO g4dn (T4: 3.2x slower, 16 GB OOM).
      // The container has no hard memory cap (see makeTaskDefinition).
      //
      // g6.2xlarge added as a deeper fallback (2026-06-09): same L4 GPU as
      // g6.xlarge (sm_89, 24 GB VRAM — golden AMI + NGC fat binaries already
      // cover it), but 8 vCPU + 32 GB RAM. Two reasons: (1) more spot pools to
      // survive a g5+g6.xlarge drought; (2) the 32 GB system RAM fits the
      // big-checkpoint / FLOW-class jobs that OOM on a 16 GB .xlarge — the image
      // task has no hard memoryLimit and only an 11 GiB soft reservation, so it
      // places on .xlarge AND .2xlarge and uses the extra RAM when present.
      // capacity-optimized-PRIORITIZED keeps g5.xlarge first, so this only fires
      // on a deep drought. CAVEAT: g6.2xlarge is 8 vCPU, so at imageMax (6) a
      // full fall-back to it = 48 vCPU = the entire G/VT spot quota (shared with
      // video) — only reachable in a deep drought at max scale; raise the quota
      // or lower imageMax if that becomes real.
      // 2026-07-12: moved the PRIMARY to 32 GB-RAM instances. Z-Image (a FLOW-arch
      // model: ~6 GB UNET + ~5.6 GB qwen text encoder + VAE ≈ 12 GB of weights +
      // ComfyUI working set) OOM-kills ComfyUI (rc=-9) on a 16 GB .xlarge even after
      // stripping the FaceDetailer/upscale — exactly the FLOW-class OOM the g6.2xlarge
      // note above predicted. g5.2xlarge = same A10G (24 GB VRAM, sm_86) but 8 vCPU /
      // 32 GB RAM; g6.2xlarge (L4) + g6e.xlarge (L40S, 4 vCPU/32 GB) are 32 GB
      // fallbacks. All ≥32 GB RAM so no pool can OOM a FLOW job. 16 GB g5/g6.xlarge
      // pools are intentionally dropped. imageMax 3 × 8 vCPU = 24, under the 48 G/VT
      // spot quota (shared with video).
      //
      // 2026-08-10: swapped PRIMARY to g6.2xlarge on fresh job-ledger data. 94%
      // of the last 30 days' jobs are Z-Image fp8 (divingZImageTurbo_v60Fp8),
      // which needs one of these 32 GB-RAM pools (see the 2026-07-12 note
      // above). Measured median job duration by instance type for that model
      // (2,429 jobs, 5s<dur<1200s filter): g5.2xlarge (A10G) 76s, g6.2xlarge
      // (L4) 58s, g6e.xlarge (L40S) 47s — L4/L40S beat the A10G because Ada has
      // native fp8 support; Ampere upcasts fp8, costing time. At measured spot
      // prices (g5.2xlarge ~$1.11/hr, g6.2xlarge ~$0.92/hr, g6e.xlarge
      // ~$1.74/hr) that's ~$0.0234/image (g5.2xlarge) vs ~$0.0148/image
      // (g6.2xlarge) vs ~$0.0227/image (g6e.xlarge) — g6.2xlarge is BOTH the
      // fastest and the cheapest of the three, ~35% cheaper per image than
      // either alternative. Under the OLD config (primary g5.2xlarge,
      // CAPACITY_OPTIMIZED_PRIORITIZED) only 5.8% of jobs actually landed on
      // the primary; 66.6% fell through to g6.2xlarge and 26.9% to the pricier
      // g6e.xlarge anyway — the allocator was already routing most traffic off
      // the "primary". Making g6.2xlarge the explicit primary (and switching
      // the spot allocation strategy to PRICE_CAPACITY_OPTIMIZED — see
      // compute.ts) fixes both: the common case now gets the fastest+cheapest
      // pool directly, and g6e.xlarge only fires in a real g5/g6.2xlarge
      // drought instead of absorbing over a quarter of all traffic.
      primaryInstanceType: 'g6.2xlarge',
      fallbackInstanceTypes: ['g5.2xlarge', 'g6e.xlarge'],
      rootVolumeGb: 150,
      // gp3 250 MB/s / 6000 IOPS (above the 125/3000 floor). The cold-start
      // bottleneck is the ECR image pull+extract to this root volume. Measured
      // on an on-demand g5.xlarge (2026-06-08): the same `docker pull` of the
      // 13 GB (→25.7 GB extracted) image took 394 s at 125/3000 vs 273 s at
      // 250/6000 — a measured 121 s (−31%) cold-start saving for ~$2–8/mo
      // (scale-from-zero, per-second billed). Going higher hits a single-thread
      // gzip-decompress floor (~273 s) AND the g5.xlarge ~437 MB/s EBS burst
      // ceiling, so 250 is the sweet spot; image-slimming can't help (69% of
      // bytes are inherited NGC base layers). Only a golden AMI removes the
      // pull entirely — deferred (re-bake-per-release maintenance burden).
      rootVolumeIops: 6000,
      rootVolumeThroughputMbps: 250,
      // Golden-AMI hydration rate (only applied when `-c useGoldenAmi=true`).
      // Measured: 300 MiB/s → ComfyUI cold-ready ~32s; default lazy-load (~26
      // MB/s) never ready in the 300s grace. See imagebuilder.ts / compute.ts.
      volumeInitializationRateMiBs: 300,
    },
    video: {
      fleetName: 'video',
      // 2026-08-19: primary moved g5.xlarge -> g5.2xlarge. Same A10G (22 GB
      // VRAM, sm_86) but 32 GB system RAM instead of 16 GB, and RAM is what
      // binds here. SCAIL-2 loads a 15.3 GiB transformer plus a 10.6 GiB umt5
      // encoder, and WanVideoBlockSwap deliberately parks swapped blocks in
      // SYSTEM memory — on a 16 GB host that is an OOM waiting to happen, and
      // the failure lands after a ~10 min cold start. ~1.9x hourly on spot
      // ($0.93 vs $0.50) buys a fleet that can actually hold its own models.
      // Every remaining type is >= 32 GB RAM, so no pool can OOM a video job.
      primaryInstanceType: 'g5.2xlarge',
      fallbackInstanceTypes: ['g6e.xlarge', 'g6e.2xlarge'],
      rootVolumeGb: 250,
      // Left at the gp3 floor for now (the image-fleet bump above is the
      // measured win; flip these to 6000/250 the same way if video cold-start
      // proves to be a problem).
      rootVolumeIops: 3000,
      rootVolumeThroughputMbps: 125,
    },

    // ----- MINIMAX fleet -----
    // Its own fleet rather than a mode on video, because its floor is far
    // higher and video jobs should not pay for it. MiniMax H3 needs ~45 GB
    // VRAM (19.5 GiB transformer + 4.9 GiB video VAE + activations at
    // 768x1344x124) and >= 32 GB RAM to stage a 25.3 GiB text encoder that
    // ComfyUI offloads to CPU (text_encoder_initial_device wants
    // size * 1.2 < free VRAM, which 27 GB never satisfies). Only g6e (L40S,
    // 45 GB) clears that, at ~3.2x g5.xlarge on spot — which is exactly why
    // SCAIL and Wan keep the cheaper A10G pool above.
    //
    // Shares the comfy-video-worker image: MiniMax H3 is native to ComfyUI
    // from v0.30.0, and that image is now v0.33.0.
    minimax: {
      fleetName: 'minimax',
      // g6e.2xlarge (64 GB RAM) is primary, not g6e.xlarge (32 GB), for the
      // same reason the video fleet left g5.xlarge: the binding constraint is
      // SYSTEM memory, not VRAM. ComfyUI parks the 25.3 GiB int8_convrot text
      // encoder on the CPU (text_encoder_initial_device keeps an encoder on the
      // GPU only when size * 1.2 < free VRAM, which 27 GB never satisfies), and
      // 25.3 GiB inside 32 GB leaves ~6 GB for the OS, ComfyUI and the mount-s3
      // cache. Both sizes carry the same L40S, so the extra ~$0.35/hr on spot
      // buys headroom rather than GPU. xlarge stays as a fallback for a
      // 2xlarge capacity drought — it may work, it is just not worth betting a
      // 15-minute cold start on.
      primaryInstanceType: 'g6e.2xlarge',
      fallbackInstanceTypes: ['g6e.xlarge'],
      // 45 GiB of weights land on the NVMe instance store, not this volume,
      // but the ~26 GB container image is extracted here on every cold boot.
      rootVolumeGb: 250,
      rootVolumeIops: 3000,
      rootVolumeThroughputMbps: 125,
    },
  },

  storage: {
    modelsBucketPrefix: 'comfy-models',
    outputsBucketPrefix: 'comfy-outputs',
    uploadsBucketPrefix: 'comfy-uploads',
    frontendBucketPrefix: 'comfy-frontend',
    failedWorkflowsBucketPrefix: 'comfy-failed-workflows',
    outputsLifecycleDays: 30, // Standard → IA at this age
    uploadsLifecycleDays: 7, // Auto-delete user-uploaded inputs
  },

  queues: {
    imageVisibilityTimeout: Duration.minutes(15),
    videoVisibilityTimeout: Duration.minutes(45), // v3: raised from 30 (resolves N9)
    // MiniMax H3 is slower than Wan per clip AND has a much heavier cold start
    // (19.5 GiB transformer + 25.3 GiB text encoder to stream from S3), so a
    // redelivery mid-job would duplicate work that has already cost 10+ min.
    minimaxVisibilityTimeout: Duration.minutes(60),
    messageRetention: Duration.days(7),
    maxReceiveCount: 3,
  },

  scaling: {
    imageMin: 0,
    // 3 concurrent image workers (lowered from 6 to cap the peak fleet burn rate /
    // cost ceiling). g5.xlarge 4 vCPU → 12 vCPU; on a drought the 2xlarge fallbacks
    // (8 vCPU) → 24 vCPU — both well within the 48-vCPU G/VT spot quota, leaving
    // room for the video fleet (shared pool).
    imageMax: 3,
    videoMin: 0,
    videoMax: 3, // v3: lowered from 5 (resolves N2)
    minimaxMin: 0,
    // Deliberately 1. g6e.xlarge is ~3.2x g5.xlarge on spot and each worker
    // holds ~45 GiB of weights; a second concurrent worker doubles the burn
    // for a model this slow. Raise once there is a real queue for it.
    minimaxMax: 1,
    // NOTE: targetBacklogPerTask is a leftover from a BacklogPerTask design
    // that was never wired up (both fleets used a flat targetValue=1 message
    // instead — see attachSqsTargetTracking's history) and is unread anywhere
    // in the codebase even before scaleOutCooldown/scaleInCooldown below were
    // removed. Left in place — out of scope for this change.
    targetBacklogPerTask: 25, // v3: raised from 10 (resolves N2)
    // scaleOutCooldown/scaleInCooldown REMOVED 2026-08-10: they only fed
    // attachSqsTargetTracking's target-tracking policy, which no longer
    // exists (video's target-tracking scaling was replaced by the graduated
    // Lambda scaler — see compute.ts makeFleetScaler / services/fleet_scaler).
  },

  cost: {
    // Budget thresholds are PERCENTAGE-based (monitoring.ts): 50% warn, 80%
    // preventive scale-down, 100% kill-switch (full compute shutdown). At $250
    // that's warn $125 / scale-down $200 / kill-switch $250.
    budgetAmountUsd: 250,
    budgetEmailParameter: '/comfy/alerts/budget-email',
  },

  logs: {
    retentionDays: 7, // v3: lowered from 14 (resolves N1)
    ingestionAlarmGbPerHour: 5, // alarm if > 5 GB/hour
  },

  api: {
    throttleRatePerSec: 5,
    throttleBurst: 50,
    callRateAlarmThreshold: 100, // alarm if > 100 calls in 5 min (kill-switch trigger)
  },
};
