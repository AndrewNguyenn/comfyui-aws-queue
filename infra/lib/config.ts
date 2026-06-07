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
      // COST-OPTIMIZED for SDXL. g4dn (T4: 16 GB VRAM, sm_75) is the cheapest
      // GPU spot, and the image worker image is BUILT for it — xformers, not
      // SageAttention (which needs Ampere sm_80+); see workers/image/Dockerfile.
      // SDXL/Illustrious throughput on T4 comfortably clears the 100 imgs/hr
      // target, so g4dn.2xlarge (8 vCPU, 32 GB sys RAM) is the primary.
      primaryInstanceType: 'g4dn.2xlarge',
      // ORDER IS PRIORITY: the ASG uses capacity-optimized-PRIORITIZED, so
      // g4dn.2xlarge is used whenever its spot pool has capacity; the fleet
      // walks down to g4dn.xlarge, then g5.xlarge (A10G), only during a g4
      // drought. g5.xlarge stays as a bf16/24 GB-VRAM safety net and an extra
      // spot pool for capacity. (g6 and g5.2xlarge removed for cost — g5.2xlarge
      // was the priciest size in the set; see the trade-off below.)
      //
      // ACCEPTED TRADE-OFF — FLOW/Flux-class jobs are UNRELIABLE on this fleet.
      // A T4 has no hardware bf16 (ComfyUI falls back to fp32) and can't fit a
      // 20 GB FLOW checkpoint in 16 GB VRAM, so it offloads every step → 25-40
      // min/image, which blows the 15-min SQS visibility timeout (the job
      // retries, then DLQs). There is ONE image queue and any worker grabs any
      // job, so a FLOW job may land on a g4 worker and fail. This fleet is
      // intentionally tuned for SDXL cost; run FLOW-class models elsewhere.
      //
      // The .xlarge sizes (g4dn.xlarge, g5.xlarge) have 16 GB sys RAM — fine for
      // SDXL (~7 GB checkpoints) but they CANNOT mmap 20 GB+ checkpoints. The
      // container has no hard memory cap (see makeTaskDefinition) so it
      // schedules on both 16 GB and 32 GB sizes.
      fallbackInstanceTypes: ['g4dn.xlarge', 'g5.xlarge'],
      rootVolumeGb: 150,
    },
    video: {
      fleetName: 'video',
      primaryInstanceType: 'g5.xlarge',
      fallbackInstanceTypes: ['g5.2xlarge', 'g6e.xlarge'],
      rootVolumeGb: 250,
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
    // 3 concurrent image workers. Each 2xlarge (g5/g6) is 8 vCPU; 3 = 24 vCPU,
    // which exactly equals the us-east-1 G/VT spot quota (raised to 24,
    // approved 2026-05-10). Note the 24-vCPU pool is SHARED with the video
    // fleet — running 3 image workers consumes the entire quota, so a
    // concurrent video job is starved until image scales back in. Raising
    // imageMax past 3 (or wanting image+video headroom) needs a quota bump.
    imageMax: 3,
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
