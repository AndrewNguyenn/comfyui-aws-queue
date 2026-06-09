import { Stack, StackProps, RemovalPolicy, Duration, Aws, SecretValue } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as iam from 'aws-cdk-lib/aws-iam';
import { AppConfig } from '../config';

export interface StorageStackProps extends StackProps {
  readonly config: AppConfig;
}

/**
 * Persistent storage: S3 buckets, DynamoDB tables, Secrets Manager.
 *
 * Design choices:
 * - Bucket names use Aws.ACCOUNT_ID (resolved at deploy time, not in source).
 * - Models bucket has Intelligent-Tiering with Archive Instant Access opt-in for
 *   the long-tail catalog (resolves user request to keep 1.4 TB browseable cheaply).
 * - Output bucket has lifecycle to IA after 30 days; uploads auto-delete after 7.
 * - Frontend bucket is the only one with public-read; everything else has
 *   BlockPublicAccess enforced. Code reviewer must verify before push.
 * - DDB tables on-demand (no provisioned capacity to reason about). PITR enabled
 *   on models and jobs (catalog is irreplaceable, jobs are user history).
 * - Secrets Manager creates the CivitAI token holder; user populates manually post-deploy.
 */
export class StorageStack extends Stack {
  public readonly modelsBucket: s3.Bucket;
  public readonly outputsBucket: s3.Bucket;
  public readonly uploadsBucket: s3.Bucket;
  public readonly frontendBucket: s3.Bucket;
  public readonly failedWorkflowsBucket: s3.Bucket;
  public readonly modelsTable: dynamodb.Table;
  public readonly jobsTable: dynamodb.Table;
  public readonly downloadsTable: dynamodb.Table;
  public readonly objectInfoTable: dynamodb.Table;
  public readonly civitaiTokenSecret: secretsmanager.Secret;
  public readonly wsTicketSecret: secretsmanager.Secret;
  public readonly metadataAuthSecret: secretsmanager.Secret;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);
    const { config } = props;
    const acct = Aws.ACCOUNT_ID; // CDK token, resolved at deploy time
    // S3 bucket names are GLOBALLY unique. Suffix with the region so the same
    // account can run the stack in more than one region (or migrate between
    // regions) without name collisions — e.g. comfy-models-<acct>-us-east-1.
    const sfx = `${acct}-${this.region}`;

    // ---------- S3 BUCKETS ----------
    this.modelsBucket = new s3.Bucket(this, 'ModelsBucket', {
      bucketName: `${config.storage.modelsBucketPrefix}-${sfx}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: false, // Models are content-addressed; versioning adds cost without value
      removalPolicy: RemovalPolicy.RETAIN, // 1.4 TB of models is expensive to re-download
      intelligentTieringConfigurations: [
        {
          name: 'archive-cold-models',
          archiveAccessTierTime: Duration.days(90),
          // Note: Deep Archive is async retrieval — not enabled, would block jobs
        },
      ],
    });

    this.outputsBucket = new s3.Bucket(this, 'OutputsBucket', {
      bucketName: `${config.storage.outputsBucketPrefix}-${sfx}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true, // Cheap protection against accidental overwrite of user's work
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: 'standard-to-ia',
          enabled: true,
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: Duration.days(config.storage.outputsLifecycleDays),
            },
          ],
          noncurrentVersionExpiration: Duration.days(30),
        },
      ],
      cors: [
        {
          allowedMethods: [s3.HttpMethods.GET, s3.HttpMethods.HEAD],
          allowedOrigins: ['*'], // Restricted at API level; presigned URLs are scoped
          allowedHeaders: ['*'],
        },
      ],
    });

    this.uploadsBucket = new s3.Bucket(this, 'UploadsBucket', {
      bucketName: `${config.storage.uploadsBucketPrefix}-${sfx}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      lifecycleRules: [
        {
          id: 'auto-delete-uploads',
          enabled: true,
          expiration: Duration.days(config.storage.uploadsLifecycleDays),
        },
      ],
      cors: [
        {
          allowedMethods: [s3.HttpMethods.PUT, s3.HttpMethods.POST],
          allowedOrigins: ['*'], // Presigned URLs scope this
          allowedHeaders: ['*'],
        },
      ],
    });

    this.frontendBucket = new s3.Bucket(this, 'FrontendBucket', {
      bucketName: `${config.storage.frontendBucketPrefix}-${sfx}`,
      // Public-read intentional. Frontend is static HTML/JS/CSS only.
      // Operator: NEVER store sensitive config in this bucket.
      // Block-public-access is intentionally relaxed for this bucket only;
      // the BucketPolicy below is the actual authority on what's readable.
      publicReadAccess: false, // Set false; bucket policy below controls access
      blockPublicAccess: new s3.BlockPublicAccess({
        blockPublicAcls: true,
        blockPublicPolicy: false, // Bucket policy needs to grant public-read
        ignorePublicAcls: true,
        restrictPublicBuckets: false,
      }),
      websiteIndexDocument: 'index.html',
      websiteErrorDocument: 'index.html', // SPA-style fallback
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      cors: [
        {
          allowedMethods: [s3.HttpMethods.GET, s3.HttpMethods.HEAD],
          allowedOrigins: ['*'],
          allowedHeaders: ['*'],
        },
      ],
    });

    // Public-read bucket policy — explicit and reviewable.
    // Grants s3:GetObject on the entire bucket to anonymous principals.
    // SECURITY: only HTML/JS/CSS go here. Never config.js with secrets
    // beyond Cognito IDs and API URL (both of which are public anyway).
    this.frontendBucket.addToResourcePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        principals: [new iam.AnyPrincipal()],
        actions: ['s3:GetObject'],
        resources: [`${this.frontendBucket.bucketArn}/*`],
      })
    );

    // Workflows the reaper gave up on (failed every retry) — parked here for
    // later analysis. RETAIN on stack delete; a 90-day lifecycle keeps the
    // bucket from growing without bound.
    this.failedWorkflowsBucket = new s3.Bucket(this, 'FailedWorkflowsBucket', {
      bucketName: `${config.storage.failedWorkflowsBucketPrefix}-${sfx}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules: [
        { id: 'expire-old-analysis', enabled: true, expiration: Duration.days(90) },
      ],
    });

    // ---------- DYNAMODB TABLES ----------
    this.modelsTable = new dynamodb.Table(this, 'ModelsTable', {
      tableName: 'comfy-models',
      partitionKey: { name: 'name', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: RemovalPolicy.RETAIN, // Catalog is hard to rebuild
    });
    this.modelsTable.addGlobalSecondaryIndex({
      indexName: 'type-index',
      partitionKey: { name: 'type', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'name', type: dynamodb.AttributeType.STRING },
    });

    this.jobsTable = new dynamodb.Table(this, 'JobsTable', {
      tableName: 'comfy-jobs',
      partitionKey: { name: 'job_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      timeToLiveAttribute: 'expire_at',
      removalPolicy: RemovalPolicy.RETAIN,
    });
    // jobs-by-status: the status GSI backing the /jobs list, cancel-group, and
    // reaper sweeps. An INCLUDE projection of only the small attrs readers need
    // (the /jobs list attrs + model/character/subject denormalized at dispatch,
    // PLUS last_heartbeat + attempt_count which the reaper reads for zombie
    // detection + retry counting) — deliberately NOT the ~47 KB workflow_json.
    // DynamoDB bills GSI read capacity on the STORED item size in the index, so
    // each list read is ~1 KB instead of ~47 KB (~12x fewer read units/row, and
    // far less GSI write amplification). status/created_at are the GSI keys and
    // job_id is the base PK — all auto-projected; workflow_json/workflow_ui are
    // excluded (readers fall back to a base-table GetItem for the graph). This
    // replaced an ALL-projection 'status-index' via a zero-downtime cutover
    // (add new index -> switch readers -> drop old), 2026-06-09.
    this.jobsTable.addGlobalSecondaryIndex({
      indexName: 'jobs-by-status',
      partitionKey: { name: 'status', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'created_at', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: [
        'type', 'started_at', 'completed_at', 'output_keys', 'error',
        'progress', 'instance_type', 'cancel_requested',
        'model', 'character', 'subject',
        'last_heartbeat', 'attempt_count',
      ],
    });

    this.downloadsTable = new dynamodb.Table(this, 'DownloadsTable', {
      tableName: 'comfy-downloads',
      partitionKey: { name: 'download_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'expire_at', // 24 hour TTL set by app
      removalPolicy: RemovalPolicy.DESTROY, // Ephemeral progress tracking
    });

    this.objectInfoTable = new dynamodb.Table(this, 'ObjectInfoTable', {
      tableName: 'comfy-object-info',
      partitionKey: { name: 'fleet', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // ---------- SECRETS MANAGER ----------
    // Placeholder secret. Operator populates after Phase 1 deploy with:
    //   aws secretsmanager put-secret-value --secret-id civitai/api-token \
    //       --secret-string '{"token":"YOUR_TOKEN"}'
    // (Use put-secret-value, not create-secret, since this stack already created it.)
    this.civitaiTokenSecret = new secretsmanager.Secret(this, 'CivitaiTokenSecret', {
      secretName: 'civitai/api-token',
      description:
        'CivitAI API token for downloading gated models. Populate manually after deploy.',
      secretObjectValue: {
        token: SecretValue.unsafePlainText('PLACEHOLDER_NOT_SET'),
      },
      removalPolicy: RemovalPolicy.RETAIN,
    });

    // Auto-generated HMAC secret for short-lived WebSocket tickets. Used by:
    //   - dispatcher /ws-ticket endpoint to sign tickets
    //   - WebSocket $connect Lambda to verify them
    // Auto-rotated never (client tickets are 60 sec); CDK generates a 32-byte
    // hex string at first deploy and reuses it forever unless secret is deleted.
    this.wsTicketSecret = new secretsmanager.Secret(this, 'WsTicketSecret', {
      secretName: 'comfy/ws-ticket-hmac',
      description:
        'HMAC secret for signing short-lived WebSocket connection tickets',
      generateSecretString: {
        excludePunctuation: true,
        passwordLength: 64,
      },
      removalPolicy: RemovalPolicy.RETAIN,
    });

    // Shared secret authenticating the dispatcher Lambda to the metadata
    // instance's ComfyUI. The metadata instance sits on a public IP with
    // 8188 open 0.0.0.0/0 (the non-VPC dispatcher can only reach it over the
    // internet). An in-container nginx requires this value in the
    // X-Comfy-Auth header and 403s everything else — so an internet scan of
    // <eip>:8188 can't reach ComfyUI or ComfyUI-Manager (an RCE surface).
    // Fetched by both metadata/entrypoint.sh and the dispatcher by the fixed
    // name below. RETAIN so the value survives stack churn.
    this.metadataAuthSecret = new secretsmanager.Secret(this, 'MetadataAuthSecret', {
      secretName: 'comfy/metadata-auth',
      description:
        'Shared secret: dispatcher Lambda -> metadata instance ComfyUI (X-Comfy-Auth header)',
      generateSecretString: {
        excludePunctuation: true,
        passwordLength: 48,
      },
      removalPolicy: RemovalPolicy.RETAIN,
    });
  }
}
