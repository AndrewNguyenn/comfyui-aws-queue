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
  readonly cacheGb: number;
}

/**
 * The single configuration used by the app. If multiple environments are needed
 * later, factor this into a function returning per-env config.
 */
export const APP_CONFIG: AppConfig = {
  projectName: 'comfyui-aws-queue',
  region: 'us-west-2',
  availabilityZone: 'us-west-2a',
  // 2d excluded: g4dn / g5 / g5.2 instance types are not offered there.
  // Keeping it would trigger InvalidFleetConfiguration errors on every
  // launch attempt that lands in 2d. 2a/2b/2c all support our full type set.
  additionalAvailabilityZones: ['us-west-2b', 'us-west-2c'],
  tags: {
    Project: 'comfyui-aws-queue',
    Environment: 'prod',
    ManagedBy: 'cdk',
  },

  fleets: {
    image: {
      fleetName: 'image',
      // g4dn.2xlarge primary (32GB sys RAM, 16GB T4 GPU). Big checkpoints like
      // the 20GB redcraft model OOM-killed ComfyUI on .xlarge (16GB sys RAM)
      // mid-sampling. .2xlarge fits comfortably; ~2x spot price (~$0.40/hr)
      // but still well under the personal-project budget.
      primaryInstanceType: 'g4dn.2xlarge',
      // .xlarge fallback when .2xlarge spot is unavailable — accepts smaller
      // model failures over total unavailability. g4 only (no g5/g6).
      // 8 vCPU quota fits 1×g4dn.2xlarge (8) or 1×g4dn.xlarge (4).
      fallbackInstanceTypes: ['g4dn.xlarge'],
      rootVolumeGb: 150,
      cacheGb: 100,
    },
    video: {
      fleetName: 'video',
      primaryInstanceType: 'g5.xlarge',
      fallbackInstanceTypes: ['g5.2xlarge', 'g6e.xlarge'],
      rootVolumeGb: 250,
      cacheGb: 200,
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
