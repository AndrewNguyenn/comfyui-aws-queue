/**
 * Centralized configuration for the comfyui-aws-queue CDK app.
 *
 * All values that vary by environment (region, instance types, sizing) live here.
 * Account ID is resolved at deploy time via cdk.Aws.ACCOUNT_ID — never hardcoded.
 */
import { Duration } from 'aws-cdk-lib';

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
    readonly messageRetention: Duration;
    readonly maxReceiveCount: number;
  };

  readonly scaling: {
    readonly imageMin: number;
    readonly imageMax: number;
    readonly videoMin: number;
    readonly videoMax: number;
    readonly targetBacklogPerTask: number;
    readonly scaleOutCooldown: Duration;
    readonly scaleInCooldown: Duration;
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

export interface FleetConfig {
  readonly fleetName: 'image' | 'video';
  readonly primaryInstanceType: string;
  readonly fallbackInstanceTypes: readonly string[];
  readonly rootVolumeGb: number;
  // gp3 root-volume performance. The container image (~26 GB uncompressed for
  // image, similar for video) is pulled + extracted to /var/lib/docker on this
  // root volume on every cold boot, so its write throughput gates cold-start
  // time. Per-fleet so we can tune the image fleet without touching video.
  readonly rootVolumeIops: number;
  readonly rootVolumeThroughputMbps: number;
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
      // g5.xlarge ONLY (A10G: 24 GB VRAM, native bf16, sm_86; 4 vCPU, 16 GB RAM).
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
      // So: A10G only. NO fallback types — a g5.xlarge spot drought means zero
      // image workers (accepted; revisit if droughts recur). NO g5.2xlarge / g4dn.
      // The container has no hard memory cap (see makeTaskDefinition).
      primaryInstanceType: 'g5.xlarge',
      fallbackInstanceTypes: [],
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
    },
    video: {
      fleetName: 'video',
      primaryInstanceType: 'g5.xlarge',
      fallbackInstanceTypes: ['g5.2xlarge', 'g6e.xlarge'],
      rootVolumeGb: 250,
      // Left at the gp3 floor for now (the image-fleet bump above is the
      // measured win; flip these to 6000/250 the same way if video cold-start
      // proves to be a problem).
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
    messageRetention: Duration.days(7),
    maxReceiveCount: 3,
  },

  scaling: {
    imageMin: 0,
    // 4 concurrent image workers. g5.xlarge is 4 vCPU → 16 vCPU total, well
    // within the 48-vCPU G/VT spot quota (approved 2026-06-08), leaving ample
    // headroom for the video fleet.
    imageMax: 4,
    videoMin: 0,
    videoMax: 3, // v3: lowered from 5 (resolves N2)
    targetBacklogPerTask: 25, // v3: raised from 10 (resolves N2)
    scaleOutCooldown: Duration.minutes(5), // v3: raised from 1 min (resolves N2)
    scaleInCooldown: Duration.minutes(15),
  },

  cost: {
    budgetAmountUsd: 100,
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
