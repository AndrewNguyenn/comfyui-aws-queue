import { Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as appscaling from 'aws-cdk-lib/aws-applicationautoscaling';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as path from 'path';
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
      desiredCount: 0, // Owned by the comfy-image-scaler Lambda (see below)
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
    // The image fleet uses a custom graduated/sticky scaler Lambda (below)
    // instead of App Auto Scaling target-tracking — the latter is symmetric and
    // jumped to max on any message. Strip DesiredCount from the CFN resource so
    // CloudFormation never sets it: otherwise any deploy that updates the
    // service for another reason (e.g. a task-def revision from a worker
    // rebuild) re-sends DesiredCount: 0 and drains the scaler-owned fleet
    // mid-batch. The Lambda owns the count; the ASG MaxSize (imageMax) bounds it.
    (imageService.node.defaultChild as ecs.CfnService).addPropertyDeletionOverride(
      'DesiredCount',
    );
    this.makeImageQueueScaler(imageService, queue.imageJobsQueue, config);

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

    // targetValue=0.5: the AlarmHigh threshold becomes 0.5 (metric > 0.5),
    // so a single visible message (= 1.0) triggers scale-out. AlarmLow
    // threshold is 0.45 (metric < 0.45), so empty queue (= 0) scales in.
    // Using targetValue=1 was a bug: alarm = metric > 1, so 1 message sat
    // at the boundary forever and nothing fired.
    target.scaleToTrackCustomMetric(`${fleetName}TrackBacklog`, {
      metric: queue.metricApproximateNumberOfMessagesVisible({
        period: Duration.minutes(1),
        statistic: 'Maximum',
      }),
      targetValue: 0.5,
      scaleOutCooldown: Duration.seconds(60),
      scaleInCooldown: Duration.minutes(15),
    });
  }

  /**
   * Graduated + sticky image-fleet autoscaler. A Lambda runs every minute,
   * reads the live image-queue depth, and sets the comfy-image ECS service's
   * desired count: ramp UP lazily by depth (50→1, 150→2, 300→3, ≥300→max),
   * hold (never shed) on the way DOWN, and release to 0 only when the queue is
   * fully cleared (nothing visible AND nothing in flight), one worker per tick.
   * See services/image_scaler/handler.py. It owns desiredCount outright (CFN no
   * longer manages it — see the DeletionOverride above); the ASG MaxSize bounds it.
   */
  private makeImageQueueScaler(
    service: ecs.Ec2Service,
    queue: import('aws-cdk-lib/aws-sqs').IQueue,
    config: AppConfig,
  ): void {
    const fn = new lambda.Function(this, 'ImageScalerFn', {
      functionName: 'comfy-image-scaler',
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      handler: 'handler.lambda_handler',
      code: lambda.Code.fromAsset(
        path.join(__dirname, '../../../services/image_scaler'),
        { exclude: ['__pycache__', '*.pyc'] }
      ),
      timeout: Duration.seconds(30),
      memorySize: 128,
      logRetention: logs.RetentionDays.ONE_WEEK,
      environment: {
        IMAGE_QUEUE_URL: queue.queueUrl,
        CLUSTER: this.cluster.clusterName,
        SERVICE: service.serviceName,
        MAX_WORKERS: String(config.scaling.imageMax),
      },
    });
    queue.grant(fn, 'sqs:GetQueueAttributes');
    fn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['ecs:DescribeServices', 'ecs:UpdateService'],
        resources: [service.serviceArn],
      })
    );
    new events.Rule(this, 'ImageScalerSchedule', {
      ruleName: 'comfy-image-scaler-tick',
      schedule: events.Schedule.rate(Duration.minutes(1)),
      targets: [new targets.LambdaFunction(fn)],
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

    // ---- NVMe instance store → /opt/cache (host) ----
    // Both fleets use mount-s3 with NVMe-backed disk cache. g4dn/g5/g6e all
    // ship local NVMe free with the instance. Format on the host so the
    // container can bind-mount /opt/cache without needing block-device access.
    userData.addCommands(
        'set -ex',
        // xfsprogs is already baked into the ECS-optimized AL2023 GPU AMI, but
        // an unconditional `dnf install` still forces a full repo-metadata
        // refresh (~149 MB) on every cold boot — measured ~15–21s, all of it on
        // the critical path before the ECS agent registers (ecs.service is
        // After=cloud-final.service). Guard it so it's a true no-op on the
        // current AMI while self-healing if a future AMI ever drops xfsprogs.
        'command -v mkfs.xfs >/dev/null || dnf install -y xfsprogs',
        'mkdir -p /opt/cache',
        'ROOT_SRC=$(findmnt -no SOURCE /)',
        'NVME_DEV=""',
        'for dev in /dev/nvme*n1; do',
        '  [ -b "$dev" ] || continue',
        '  if grep -q "^$dev " /proc/mounts; then continue; fi',
        '  if [ "$dev" = "$ROOT_SRC" ]; then continue; fi',
        '  if lsblk -no MOUNTPOINTS "$dev" 2>/dev/null | grep -q "^/$"; then continue; fi',
        '  NVME_DEV="$dev"; break',
        'done',
        'if [ -n "$NVME_DEV" ]; then',
        // -K skips the format-time TRIM/discard: instance-store SSDs are fully
        // trimmed before allocation, so the discard is redundant work on the
        // boot path (AWS explicitly recommends skipping it). Saves ~1–5s.
        '  if ! blkid "$NVME_DEV" >/dev/null 2>&1; then mkfs.xfs -f -K "$NVME_DEV"; fi',
        '  mount "$NVME_DEV" /opt/cache',
        '  chmod 1777 /opt/cache',
        '  echo "instance store $NVME_DEV mounted at /opt/cache"',
        'else',
        '  echo "no instance store; /opt/cache stays on root volume"',
        'fi'
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
            // Per-fleet (see config.ts). The image fleet runs 250/6000 to speed
            // the cold-boot ECR image pull+extract onto this volume (measured
            // −121 s); video stays at the 125/3000 floor.
            iops: fleet.rootVolumeIops,
            throughput: fleet.rootVolumeThroughputMbps,
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
          // Pure spot — user explicitly never wants on-demand (and the
          // account's on-demand G/VT quota is 0 anyway).
          onDemandBaseCapacity: 0,
          onDemandPercentageAboveBaseCapacity: 0,
          // capacity-optimized-PRIORITIZED for both fleets — honors the
          // launchTemplate overrides order (primaryInstanceType first) and
          // optimizes for spot capacity (fewest interruptions) as the tiebreak.
          // (We briefly ran image on LOWEST_PRICE to force g4/T4 for cost, but
          // the T4 was ~3x slower, pricier per image, AND OOM-unstable on
          // g4dn.xlarge — reverted to g5.xlarge-only; see config.ts.)
          spotAllocationStrategy:
            autoscaling.SpotAllocationStrategy.CAPACITY_OPTIMIZED_PRIORITIZED,
        },
      },
      minCapacity,
      maxCapacity,
      // capacityRebalance:false — with it on, AWS rebalance recommendations
      // made the ASG terminate at-risk spot instances, but during a capacity
      // drought the replacement launch fails (UnfulfillableCapacity) and the
      // fleet churns to zero. Off: an instance runs until an actual spot
      // interruption (with its 2-minute warning) — strictly better than
      // being pre-emptively killed with no replacement available.
      capacityRebalance: false,
      // Grace period so a freshly-launched spot instance has time to boot
      // and register before EC2 health checks can act on it.
      healthCheck: autoscaling.HealthCheck.ec2({ grace: Duration.seconds(300) }),
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

    // Both fleets use mount-s3 + NVMe cache. host-cache bind-mount + FUSE
    // capability + /dev/fuse device are uniform across image and video.
    const taskDef = new ecs.Ec2TaskDefinition(this, `${fleetName}TaskDef`, {
      family: `comfy-${fleetName}`,
      networkMode: ecs.NetworkMode.HOST,
      volumes: [{ name: 'host-cache', host: { sourcePath: '/opt/cache' } }],
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

    const container = taskDef.addContainer(`${fleetName}Container`, {
      containerName: `comfy-${fleetName}-worker`,
      image: ecs.ContainerImage.fromEcrRepository(ecrRepository, 'latest'),
      gpuCount: 1,
      // FUSE access for mount-s3 — uniform across image + video.
      // TODO: investigate unprivileged FUSE fd handoff (mount-s3 CONFIGURATION.md)
      // if we ever need to drop CAP_SYS_ADMIN — currently requires a host-side
      // helper we don't have.
      linuxParameters: new ecs.LinuxParameters(this, `${fleetName}LinuxParams`, {}),
      // Memory:
      //   - memoryReservationMiB (soft / used for ECS placement): 11264 fits
      //     both .xlarge (16 GB) and .2xlarge (32 GB) with ECS-agent + OS
      //     headroom.
      //   - image: NO hard memoryLimitMiB. A hard cap sized for .2xlarge
      //     (28672) exceeds what a .xlarge fallback even has, making the task
      //     unschedulable there — which silently stranded the fleet. Without
      //     a hard cap the container uses whatever the instance provides; a
      //     20 GB+ FLOW checkpoint landing on a .xlarge will still OOM the
      //     box (accepted — see config.ts fallbackInstanceTypes).
      //   - video: keep the 28672 hard cap — always runs on instances with
      //     enough RAM; bounds it against mount-s3's host-side RSS.
      memoryReservationMiB: fleetName === 'image' ? 11264 : 24576,
      memoryLimitMiB: fleetName === 'image' ? undefined : 28672,
      essential: true,
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: fleetName,
        logGroup,
        // Multiline pattern alone — docker rejects both multilinePattern AND
        // datetimeFormat in the same log driver config:
        //   "you cannot configure log opt 'awslogs-datetime-format' and
        //    'awslogs-multiline-pattern' at the same time"
        // multilinePattern is the more general option (matches any line that
        // starts a new event), so we keep that and drop datetimeFormat.
        multilinePattern: '^[A-Z]{3,}|^Traceback|^\\d{4}-\\d{2}-\\d{2}',
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

    // FUSE capability + device + host-cache bind-mount for mount-s3.
    container.linuxParameters!.addCapabilities(ecs.Capability.SYS_ADMIN);
    container.linuxParameters!.addDevices({
      hostPath: '/dev/fuse',
      containerPath: '/dev/fuse',
      permissions: [
        ecs.DevicePermission.READ,
        ecs.DevicePermission.WRITE,
        ecs.DevicePermission.MKNOD,
      ],
    });
    container.addMountPoints({
      sourceVolume: 'host-cache',
      containerPath: '/opt/cache',
      readOnly: false,
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
    // manifest_installer reads s3://<outputs>/manifests/custom-nodes.json on
    // worker boot to know which custom nodes to git-clone. grantWrite doesn't
    // include GetObject so we add it explicitly here, scoped to manifests/*.
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject'],
      resources: [`${storage.outputsBucket.bucketArn}/manifests/*`],
    }));
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
