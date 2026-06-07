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
      // Fallbacks across the g5 (A10G) and g6 (L4) families — added after a
      // us-east-1 g5 spot drought left the fleet unable to launch anything
      // (UnfulfillableCapacity in every g5 pool). All four are bf16-capable.
      // ORDER IS PRIORITY: the ASG uses capacity-optimized-PRIORITIZED, so
      // g5.2xlarge (the intended main: A10G, 8 vCPU, 32 GB) is used whenever
      // it has spot capacity; the fleet only falls to g5.xlarge, then the g6
      // (L4) sizes, during a drought. g6 is actually cheaper than g5, so
      // falling back never raises cost.
      // Trade-off: the .xlarge sizes have 16 GB sys RAM — fine for SDXL
      // (~7 GB checkpoints) but they CANNOT mmap 20 GB+ FLOW checkpoints
      // (kernel overcommit refuses a mapping > physical RAM), so a heavy
      // FLOW job that lands on a .xlarge will OOM mid-load. Accepted: a
      // worker running SDXL beats no worker at all during a drought.
      // The container has no hard memory cap (see makeTaskDefinition) so it
      // schedules on both .xlarge (16 GB) and .2xlarge (32 GB).
      fallbackInstanceTypes: ['g5.xlarge', 'g6.2xlarge', 'g6.xlarge'],
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
