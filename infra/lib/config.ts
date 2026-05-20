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
  // a/b/c/d/f all offer g5.xlarge/g5.2xlarge (image fleet is g5.2xlarge-only,
  // so pool depth directly drives how long a job waits during a spot
  // drought); g6e.xlarge is offered in a/b/c/d — capacity-optimized simply
  // skips f for that type. 1e is excluded (no GPU capacity). Extra public
  // subnets are free here — no NAT GW, no interface endpoints.
  additionalAvailabilityZones: ['us-east-1b', 'us-east-1c', 'us-east-1d', 'us-east-1f'],
  tags: {
    Project: 'comfyui-aws-queue',
    Environment: 'prod',
    ManagedBy: 'cdk',
  },

  fleets: {
    image: {
      fleetName: 'image',
      // g5.2xlarge: A10G (24 GB VRAM, native bf16, sm_86) + 32 GB sys RAM.
      // The catalog now includes Flux-class FLOW checkpoints (e.g. the 20 GB
      // redcraft model). A T4 (g4dn) has no hardware bf16 — ComfyUI falls
      // back to fp32 — and a 20 GB model can't fit the T4's 16 GB VRAM, so
      // it offloads every step: a single generation takes 25-40 min and
      // blows the worker timeout. On an A10G the same model runs in bf16,
      // mostly in-VRAM: ~2-3 min/image.
      primaryInstanceType: 'g5.2xlarge',
      // No fallback. g5.xlarge / g4dn.xlarge (16 GB sys RAM) can't mmap the
      // 20 GB+ checkpoints — the kernel overcommit heuristic refuses a
      // mapping larger than physical RAM. g4dn.2xlarge has the RAM but its
      // T4 is too slow for Flux (see above). A fallback that's guaranteed to
      // fail (or take 40 min) is worse than waiting — it burns a spot launch
      // before erroring. g5.2xlarge only; if A10G spot is unavailable the
      // job waits in SQS. 8 vCPU = the full us-east-1 G/VT spot quota.
      fallbackInstanceTypes: [],
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
    imageMax: 1,
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
