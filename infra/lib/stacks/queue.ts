import { Stack, StackProps, RemovalPolicy } from 'aws-cdk-lib';
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
  }
}
