import { Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as appscaling from 'aws-cdk-lib/aws-applicationautoscaling';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { AppConfig, FleetConfig } from '../config';
import { NetworkStack } from './network';
import { StorageStack } from './storage';
import { QueueStack } from './queue';
import { CiStack } from './ci';

export interface ComputeStackProps extends StackProps {
  readonly config: AppConfig;
  readonly network: NetworkStack;
  readonly storage: StorageStack;
  readonly queue: QueueStack;
  readonly ci: CiStack;
}

/**
 * ECS cluster, two ASGs (image fleet + video fleet), capacity providers, services.
 *
 * Each fleet has its own ASG with mixed-instance spot policy and capacity-rebalance.
 * Capacity provider managed scaling enabled (target 100). Service-level scaling
 * uses backlog-per-task target tracking.
 *
 * IMPORTANT: This stack will fail to launch instances until the AWS account has
 * spot vCPU quota for G/VT > 0. See README "Prerequisites" — request quota first.
 */
export class ComputeStack extends Stack {
  public readonly cluster: ecs.Cluster;
  public readonly imageAsg: autoscaling.AutoScalingGroup;
  public readonly videoAsg: autoscaling.AutoScalingGroup;

  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id, props);
    const { config, network, storage, queue, ci } = props;

    this.cluster = new ecs.Cluster(this, 'Cluster', {
      clusterName: `${config.projectName}-cluster`,
      vpc: network.vpc,
      containerInsights: false, // Costs $1/container/mo; not needed at this scale
    });

    // ----- IMAGE fleet -----
    this.imageAsg = this.makeFleetAsg('image', config.fleets.image, {
      network,
      config,
      ecrRepoArn: ci.imageWorkerRepo.repositoryArn,
    });
    const imageCapacityProvider = this.makeCapacityProvider('image', this.imageAsg);
    this.cluster.addAsgCapacityProvider(imageCapacityProvider);

    const imageTaskDef = this.makeTaskDefinition('image', {
      ecrRepository: ci.imageWorkerRepo,
      queueUrl: queue.imageJobsQueue.queueUrl,
      storage,
      config,
    });
    this.grantWorkerPermissions(imageTaskDef.taskRole, queue.imageJobsQueue, storage);

    const imageService = new ecs.Ec2Service(this, 'ImageService', {
      serviceName: 'comfy-image',
      cluster: this.cluster,
      taskDefinition: imageTaskDef,
      desiredCount: 0, // Scaling controlled by App Auto Scaling on SQS depth
      capacityProviderStrategies: [
        {
          capacityProvider: imageCapacityProvider.capacityProviderName,
          weight: 1,
        },
      ],
      minHealthyPercent: 0, // v3 N12: allow brief downtime for max=1 deployments
      maxHealthyPercent: 100,
      placementConstraints: [
        ecs.PlacementConstraint.memberOf(`attribute:fleet == image`),
      ],
    });
    this.attachSqsTargetTracking(
      'image',
      imageService,
      queue.imageJobsQueue,
      config.scaling.imageMin,
      config.scaling.imageMax,
      config,
    );

    // ----- VIDEO fleet -----
    this.videoAsg = this.makeFleetAsg('video', config.fleets.video, {
      network,
      config,
      ecrRepoArn: ci.videoWorkerRepo.repositoryArn,
    });
    const videoCapacityProvider = this.makeCapacityProvider('video', this.videoAsg);
    this.cluster.addAsgCapacityProvider(videoCapacityProvider);

    const videoTaskDef = this.makeTaskDefinition('video', {
      ecrRepository: ci.videoWorkerRepo,
      queueUrl: queue.videoJobsQueue.queueUrl,
      storage,
      config,
    });
    this.grantWorkerPermissions(videoTaskDef.taskRole, queue.videoJobsQueue, storage);

    const videoService = new ecs.Ec2Service(this, 'VideoService', {
      serviceName: 'comfy-video',
      cluster: this.cluster,
      taskDefinition: videoTaskDef,
      desiredCount: 0,
      capacityProviderStrategies: [
        {
          capacityProvider: videoCapacityProvider.capacityProviderName,
          weight: 1,
        },
      ],
      minHealthyPercent: 0,
      maxHealthyPercent: 100,
      placementConstraints: [
        ecs.PlacementConstraint.memberOf(`attribute:fleet == video`),
      ],
    });
    this.attachSqsTargetTracking(
      'video',
      videoService,
      queue.videoJobsQueue,
      config.scaling.videoMin,
      config.scaling.videoMax,
      config,
    );
  }

  /**
   * Application Auto Scaling target tracking driven by SQS visible-message
   * count. Scales the ECS service desired count up when messages queue,
   * down when the queue empties. Capacity provider follows the service
   * (already wired) so the ASG provisions instances as needed.
   *
   * Why target=1 (not BacklogPerTask): for our 1-user scale (max=1 image,
   * max=3 video), a target of 1 visible message per task means any single
   * job triggers scale-up. Target tracking caps at maxCapacity so a 50-msg
   * burst doesn't blow past max. Simpler than metric-math BacklogPerTask
   * and behaves the same at this scale.
   *
   * Cooldowns: aggressive scale-out (60s — boot is the slow part anyway,
   * no point dawdling), patient scale-in (15 min — keeps worker warm
   * across short pauses in interactive use).
   */
  private attachSqsTargetTracking(
    fleetName: 'image' | 'video',
    service: ecs.Ec2Service,
    queue: import('aws-cdk-lib/aws-sqs').IQueue,
    minCapacity: number,
    maxCapacity: number,
    _config: AppConfig,
  ): void {
    const target = service.autoScaleTaskCount({
      minCapacity,
      maxCapacity,
    });

    target.scaleToTrackCustomMetric(`${fleetName}TrackBacklog`, {
      customMetric: queue.metricApproximateNumberOfMessagesVisible({
        period: Duration.minutes(1),
        statistic: 'Maximum',
      }),
      targetValue: 1,
      scaleOutCooldown: Duration.seconds(60),
      scaleInCooldown: Duration.minutes(15),
    });
  }

  /**
   * Build an ASG with spot mixed-instance policy + capacity rebalance.
   * Uses AL2023 GPU AMI (AL2 EOL is 2026-06-30).
   */
  private makeFleetAsg(
    fleetName: 'image' | 'video',
    fleet: FleetConfig,
    deps: { network: NetworkStack; config: AppConfig; ecrRepoArn: string }
  ): autoscaling.AutoScalingGroup {
    const { network, config } = deps;

    // ECS-optimized AL2023 GPU AMI (resolved via SSM at deploy time).
    // v3 N10: pin SSM parameter version in production to avoid silent driver regressions.
    const machineImage = ecs.EcsOptimizedImage.amazonLinux2023(ecs.AmiHardwareType.GPU);

    const userData = ec2.UserData.forLinux();
    // ECS agent picks up these attributes for placement constraints.
    userData.addCommands(
      `echo ECS_CLUSTER=${config.projectName}-cluster >> /etc/ecs/ecs.config`,
      `echo ECS_INSTANCE_ATTRIBUTES='{"fleet":"${fleetName}"}' >> /etc/ecs/ecs.config`,
      `echo ECS_ENABLE_SPOT_INSTANCE_DRAINING=true >> /etc/ecs/ecs.config`,
      `echo ECS_LOG_LEVEL=info >> /etc/ecs/ecs.config`,
      `echo ECS_DISABLE_IMAGE_CLEANUP=false >> /etc/ecs/ecs.config`,
      `echo ECS_IMAGE_CLEANUP_INTERVAL=10m >> /etc/ecs/ecs.config`,
      `echo ECS_IMAGE_MINIMUM_CLEANUP_AGE=30m >> /etc/ecs/ecs.config`
    );

    const launchTemplate = new ec2.LaunchTemplate(this, `${fleetName}LaunchTemplate`, {
      launchTemplateName: `comfy-${fleetName}-lt`,
      machineImage,
      instanceType: new ec2.InstanceType(fleet.primaryInstanceType),
      userData,
      securityGroup: network.workerSecurityGroup,
      role: this.makeInstanceRole(fleetName),
      requireImdsv2: true,
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: autoscaling.BlockDeviceVolume.ebs(fleet.rootVolumeGb, {
            volumeType: autoscaling.EbsDeviceVolumeType.GP3,
            iops: 3000,
            throughput: 125,
            deleteOnTermination: true,
          }),
        },
      ],
    });

    const minCapacity =
      fleetName === 'image' ? config.scaling.imageMin : config.scaling.videoMin;
    const maxCapacity =
      fleetName === 'image' ? config.scaling.imageMax : config.scaling.videoMax;

    const asg = new autoscaling.AutoScalingGroup(this, `${fleetName}Asg`, {
      autoScalingGroupName: `comfy-${fleetName}-asg`,
      vpc: network.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      mixedInstancesPolicy: {
        launchTemplate,
        launchTemplateOverrides: [
          { instanceType: new ec2.InstanceType(fleet.primaryInstanceType) },
          ...fleet.fallbackInstanceTypes.map((t) => ({
            instanceType: new ec2.InstanceType(t),
          })),
        ],
        instancesDistribution: {
          // Pure spot — never on-demand. User accepts spot capacity risk.
          onDemandBaseCapacity: 0,
          onDemandPercentageAboveBaseCapacity: 0,
          spotAllocationStrategy: autoscaling.SpotAllocationStrategy.CAPACITY_OPTIMIZED,
        },
      },
      minCapacity,
      maxCapacity,
      capacityRebalance: true,
      newInstancesProtectedFromScaleIn: false,
    });

    return asg;
  }

  /**
   * Capacity provider with managed scaling so ASG follows ECS task count.
   */
  private makeCapacityProvider(
    fleetName: 'image' | 'video',
    asg: autoscaling.AutoScalingGroup
  ): ecs.AsgCapacityProvider {
    return new ecs.AsgCapacityProvider(this, `${fleetName}CapacityProvider`, {
      capacityProviderName: `cp-${fleetName}-spot`,
      autoScalingGroup: asg,
      enableManagedScaling: true,
      enableManagedTerminationProtection: false, // Spot interrupts can't be protected
      targetCapacityPercent: 100,
      minimumScalingStepSize: 1,
      maximumScalingStepSize: 1,
      spotInstanceDraining: true, // Critical: graceful drain on spot termination notice
    });
  }

  /**
   * Task definition. Worker is a single container, network mode `host`.
   *
   * DISPATCHER_API_URL and worker API key are looked up from SSM at deploy
   * time (resolves circular Compute ↔ Api dependency from review C3).
   */
  private makeTaskDefinition(
    fleetName: 'image' | 'video',
    deps: { ecrRepository: import('aws-cdk-lib/aws-ecr').IRepository; queueUrl: string; storage: StorageStack; config: AppConfig }
  ): ecs.Ec2TaskDefinition {
    const { ecrRepository, queueUrl, storage, config } = deps;

    const taskDef = new ecs.Ec2TaskDefinition(this, `${fleetName}TaskDef`, {
      family: `comfy-${fleetName}`,
      networkMode: ecs.NetworkMode.HOST,
    });

    const logGroup = new logs.LogGroup(this, `${fleetName}WorkerLogs`, {
      logGroupName: `/comfy/workers/${fleetName}`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // SSM parameter lookups happen at deploy time, not runtime — they resolve
    // to literal strings in the task def. Compute stack only needs to be
    // deployed AFTER api stack (already declared in bin via addDependency).
    const dispatcherApiUrl = require('aws-cdk-lib/aws-ssm').StringParameter
      .valueForStringParameter(this, '/comfy/api/url');
    const workerApiKeyId = require('aws-cdk-lib/aws-ssm').StringParameter
      .valueForStringParameter(this, '/comfy/api/worker-key-id');

    taskDef.addContainer(`${fleetName}Container`, {
      containerName: `comfy-${fleetName}-worker`,
      image: ecs.ContainerImage.fromEcrRepository(ecrRepository, 'latest'),
      gpuCount: 1,
      memoryReservationMiB: fleetName === 'image' ? 11264 : 24576,
      memoryLimitMiB: fleetName === 'image' ? 13312 : 28672,
      essential: true,
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: fleetName,
        logGroup,
        // v3 N1: multiline pattern collapses tqdm progress bars into single events
        multilinePattern: '^[A-Z]{3,}|^Traceback|^\\d{4}-\\d{2}-\\d{2}',
        datetimeFormat: '%Y-%m-%d %H:%M:%S',
      }),
      environment: {
        FLEET: fleetName,
        QUEUE_URL: queueUrl,
        MODELS_BUCKET: storage.modelsBucket.bucketName,
        OUTPUTS_BUCKET: storage.outputsBucket.bucketName,
        UPLOADS_BUCKET: storage.uploadsBucket.bucketName,
        FRONTEND_BUCKET: storage.frontendBucket.bucketName,
        MODELS_TABLE: storage.modelsTable.tableName,
        JOBS_TABLE: storage.jobsTable.tableName,
        OBJECT_INFO_TABLE: storage.objectInfoTable.tableName,
        AWS_REGION: this.region,
        VISIBILITY_TIMEOUT_SECONDS: fleetName === 'image' ? '900' : '2700',
        // For /internal/object_info publish (resolves review C2 + C3):
        DISPATCHER_API_URL: dispatcherApiUrl,
        WORKER_API_KEY_ID: workerApiKeyId,
      },
    });

    return taskDef;
  }

  /**
   * IAM role for the EC2 instance itself (ECS agent + S3 + ECR pull).
   */
  private makeInstanceRole(fleetName: string): iam.Role {
    const role = new iam.Role(this, `${fleetName}InstanceRole`, {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });
    return role;
  }

  /**
   * v3 N6/M6: tightly-scoped IAM for the worker task. NEVER PutObject on models bucket.
   */
  private grantWorkerPermissions(
    role: iam.IRole,
    queue: import('aws-cdk-lib/aws-sqs').IQueue,
    storage: StorageStack
  ): void {
    queue.grantConsumeMessages(role);
    storage.modelsBucket.grantRead(role); // GetObject + ListBucket only — NO put/delete
    storage.outputsBucket.grantWrite(role); // PutObject + GetObject (verify); no Delete
    storage.uploadsBucket.grantRead(role);
    storage.jobsTable.grantReadWriteData(role);
    storage.modelsTable.grantReadData(role);
    storage.objectInfoTable.grantReadWriteData(role);
    // Frontend bucket: write under extensions/ prefix only (custom-node JS files).
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['s3:PutObject', 's3:PutObjectAcl'],
      resources: [`${storage.frontendBucket.bucketArn}/extensions/*`],
    }));
    // Allow worker to fetch its API key value for /internal/object_info push.
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['apigateway:GET'],
      resources: [`arn:aws:apigateway:${this.region}::/apikeys/*`],
    }));
  }
}
