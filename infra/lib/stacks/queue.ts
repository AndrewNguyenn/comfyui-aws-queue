import { Stack, StackProps, RemovalPolicy, Duration } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { AppConfig } from '../config';

export interface QueueStackProps extends StackProps {
  readonly config: AppConfig;
}

/**
 * SQS queues for image and video job submission.
 *
 * Design choices:
 * - Two queues, not one. Lets each fleet poll its own queue and isolates traffic.
 * - DLQ per queue with maxReceiveCount=3. After 3 failed receives, message moves
 *   to DLQ for human review (alarmed in MonitoringStack).
 * - Long visibility timeouts (15 min image / 45 min video) absorb long jobs without
 *   needing aggressive heartbeat. Heartbeat extends to 12hr max if needed.
 * - SQS-managed encryption (free, AWS-owned KMS key).
 */
export class QueueStack extends Stack {
  public readonly imageJobsQueue: sqs.Queue;
  public readonly videoJobsQueue: sqs.Queue;
  public readonly imageJobsDlq: sqs.Queue;
  public readonly videoJobsDlq: sqs.Queue;
  // Self-ticking wake queue for the reaper Lambda. Not on a fixed cron — the
  // dispatcher seeds a tick when a job is submitted, and each reaper run
  // re-arms a delayed tick only while jobs are queued/running; at idle the
  // chain stops. 1 h retention so a stale tick can't resurrect a dead chain.
  public readonly reaperTicksQueue: sqs.Queue;

  constructor(scope: Construct, id: string, props: QueueStackProps) {
    super(scope, id, props);
    const { config } = props;

    this.imageJobsDlq = new sqs.Queue(this, 'ImageJobsDlq', {
      queueName: 'comfy-image-jobs-dlq',
      retentionPeriod: config.queues.messageRetention,
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    this.imageJobsQueue = new sqs.Queue(this, 'ImageJobsQueue', {
      queueName: 'comfy-image-jobs',
      visibilityTimeout: config.queues.imageVisibilityTimeout,
      retentionPeriod: config.queues.messageRetention,
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      deadLetterQueue: {
        queue: this.imageJobsDlq,
        maxReceiveCount: config.queues.maxReceiveCount,
      },
      removalPolicy: RemovalPolicy.DESTROY,
    });

    this.videoJobsDlq = new sqs.Queue(this, 'VideoJobsDlq', {
      queueName: 'comfy-video-jobs-dlq',
      retentionPeriod: config.queues.messageRetention,
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    this.videoJobsQueue = new sqs.Queue(this, 'VideoJobsQueue', {
      queueName: 'comfy-video-jobs',
      visibilityTimeout: config.queues.videoVisibilityTimeout,
      retentionPeriod: config.queues.messageRetention,
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      deadLetterQueue: {
        queue: this.videoJobsDlq,
        maxReceiveCount: config.queues.maxReceiveCount,
      },
      removalPolicy: RemovalPolicy.DESTROY,
    });

    this.reaperTicksQueue = new sqs.Queue(this, 'ReaperTicksQueue', {
      queueName: 'comfy-reaper-ticks',
      // The reaper sweep finishes in seconds; 2 min is ample for a redelivery
      // if it errors. No DLQ — a tick is cheap and idempotent, and the next
      // job submit re-arms the chain anyway.
      visibilityTimeout: Duration.minutes(2),
      retentionPeriod: Duration.hours(1),
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      removalPolicy: RemovalPolicy.DESTROY,
    });
  }
}
