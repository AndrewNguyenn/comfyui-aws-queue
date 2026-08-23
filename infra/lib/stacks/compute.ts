import { Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as path from 'path';
import { AppConfig, FleetConfig, GOLDEN_AMI_PARAM, FleetName } from '../config';
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
 * for BOTH fleets is owned by one graduated/sticky Lambda (see makeFleetScaler /
 * services/fleet_scaler) — not ECS Application Auto Scaling target-tracking,
 * which video used until 2026-08-10 and which was too slow to scale in.
 *
 * IMPORTANT: This stack will fail to launch instances until the AWS account has
 * spot vCPU quota for G/VT > 0. See README "Prerequisites" — request quota first.
 */
export class ComputeStack extends Stack {
  public readonly cluster: ecs.Cluster;
  public readonly imageAsg: autoscaling.AutoScalingGroup;
  public readonly videoAsg: autoscaling.AutoScalingGroup;
  public readonly minimaxAsg: autoscaling.AutoScalingGroup;

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
      desiredCount: 0, // Owned by the comfy-fleet-scaler Lambda (see below)
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
      desiredCount: 0, // Owned by the comfy-fleet-scaler Lambda (see below)
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
    // Same reasoning as the image service above: the fleet scaler owns
    // DesiredCount outright, so a task-def-revision deploy must not let CFN
    // reset it back to 0 mid-batch.
    (videoService.node.defaultChild as ecs.CfnService).addPropertyDeletionOverride(
      'DesiredCount',
    );

    // ----- MINIMAX fleet -----
    // Same worker image as video (MiniMax H3 is native to the ComfyUI in it),
    // but its own ASG so the g6e/L40S premium is confined to jobs that need
    // it, and its own queue so the scaler sizes it independently.
    this.minimaxAsg = this.makeFleetAsg('minimax', config.fleets.minimax, {
      network,
      config,
      ecrRepoArn: ci.videoWorkerRepo.repositoryArn,
    });
    const minimaxCapacityProvider = this.makeCapacityProvider('minimax', this.minimaxAsg);
    this.cluster.addAsgCapacityProvider(minimaxCapacityProvider);

    const minimaxTaskDef = this.makeTaskDefinition('minimax', {
      ecrRepository: ci.videoWorkerRepo,
      queueUrl: queue.minimaxJobsQueue.queueUrl,
      storage,
      config,
    });
    this.grantWorkerPermissions(minimaxTaskDef.taskRole, queue.minimaxJobsQueue, storage);

    const minimaxService = new ecs.Ec2Service(this, 'MinimaxService', {
      serviceName: 'comfy-minimax',
      cluster: this.cluster,
      taskDefinition: minimaxTaskDef,
      desiredCount: 0, // Owned by the comfy-fleet-scaler Lambda (see below)
      capacityProviderStrategies: [
        {
          capacityProvider: minimaxCapacityProvider.capacityProviderName,
          weight: 1,
        },
      ],
      minHealthyPercent: 0,
      maxHealthyPercent: 100,
      placementConstraints: [
        ecs.PlacementConstraint.memberOf(`attribute:fleet == minimax`),
      ],
    });
    (minimaxService.node.defaultChild as ecs.CfnService).addPropertyDeletionOverride(
      'DesiredCount',
    );

    // ONE Lambda drives all fleets' desired counts — see makeFleetScaler.
    // Replaces video's former ECS Application Auto Scaling target-tracking
    // policy (removed 2026-08-10): target-tracking's scale-in alarm needs
    // ~15 consecutive minutes below target to fire, so every video burst
    // paid ~15-30 min of idle GPU at burst end. The graduated/sticky Lambda
    // releases one worker per tick (~1-2 min) once a fleet's queue drains,
    // same as image already did.
    this.makeFleetScaler(
      [
        { name: 'image', service: imageService, queue: queue.imageJobsQueue, max: config.scaling.imageMax },
        { name: 'video', service: videoService, queue: queue.videoJobsQueue, max: config.scaling.videoMax },
        { name: 'minimax', service: minimaxService, queue: queue.minimaxJobsQueue, max: config.scaling.minimaxMax },
      ],
    );
  }

  /**
   * Graduated + sticky autoscaler for BOTH fleets. ONE Lambda runs every
   * minute, reads each fleet's live queue depth, and sets that fleet's ECS
   * service desired count independently: ramp UP lazily by depth (see
   * services/fleet_scaler/handler.py for the per-fleet band tables), hold
   * (never shed) on the way DOWN, and release to 0 only when a fleet's queue
   * is fully cleared (nothing visible AND nothing in flight), one worker per
   * tick. It owns both services' desiredCount outright (CFN no longer
   * manages either — see the DeletionOverride calls above); each ASG's
   * MaxSize bounds its own fleet.
   */
  private makeFleetScaler(
    fleets: {
      name: FleetName;
      service: ecs.Ec2Service;
      queue: import('aws-cdk-lib/aws-sqs').IQueue;
      max: number;
    }[],
  ): void {
    const fn = new lambda.Function(this, 'FleetScalerFn', {
      functionName: 'comfy-fleet-scaler',
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      handler: 'handler.lambda_handler',
      code: lambda.Code.fromAsset(
        path.join(__dirname, '../../../services/fleet_scaler'),
        { exclude: ['__pycache__', '*.pyc'] }
      ),
      timeout: Duration.seconds(30),
      memorySize: 128,
      logRetention: logs.RetentionDays.ONE_WEEK,
      environment: {
        CLUSTER: this.cluster.clusterName,
        // <FLEET>_QUEUE_URL / _SERVICE / _MAX_WORKERS per fleet; the handler
        // discovers fleets from these rather than a hardcoded list, so adding
        // one here is the only change needed on the Lambda side.
        ...Object.fromEntries(
          fleets.flatMap((f) => {
            const k = f.name.toUpperCase();
            return [
              [`${k}_QUEUE_URL`, f.queue.queueUrl],
              [`${k}_SERVICE`, f.service.serviceName],
              [`${k}_MAX_WORKERS`, String(f.max)],
            ];
          })
        ),
      },
    });
    for (const f of fleets) {
      f.queue.grant(fn, 'sqs:GetQueueAttributes');
    }
    fn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['ecs:DescribeServices', 'ecs:UpdateService'],
        resources: fleets.map((f) => f.service.serviceArn),
      })
    );
    // Container-instance reads let the scaler see capacity that is ALREADY
    // running, so it never leaves a warm GPU idle while a job waits (see
    // warm_capacity in the handler). Read-only, but the two APIs authorize on
    // DIFFERENT resource types and must be granted separately: List takes the
    // cluster, Describe takes each container-instance ARN. Granting both
    // against the cluster ARN — which is what this did at first — fails only
    // Describe, at runtime, with an AccessDenied the handler swallows by
    // design. It degrades silently back to band-only scaling, so the tell is
    // in the scaler's log rather than in a failed deploy.
    fn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['ecs:ListContainerInstances'],
        resources: [this.cluster.clusterArn],
      })
    );
    fn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['ecs:DescribeContainerInstances'],
        resources: [
          Stack.of(this).formatArn({
            service: 'ecs',
            resource: 'container-instance',
            resourceName: `${this.cluster.clusterName}/*`,
          }),
        ],
      })
    );
    new events.Rule(this, 'FleetScalerSchedule', {
      ruleName: 'comfy-fleet-scaler-tick',
      schedule: events.Schedule.rate(Duration.minutes(1)),
      targets: [new targets.LambdaFunction(fn)],
    });
  }

  /**
   * Build an ASG with spot mixed-instance policy + capacity rebalance.
   * Uses AL2023 GPU AMI (AL2 EOL is 2026-06-30).
   */
  private makeFleetAsg(
    fleetName: FleetName,
    fleet: FleetConfig,
    deps: { network: NetworkStack; config: AppConfig; ecrRepoArn: string }
  ): autoscaling.AutoScalingGroup {
    const { network, config } = deps;

    // machineImage is built AFTER userData below (the golden-AMI path needs the
    // userData object). See the `const machineImage = ...` block before the LT.

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

    // Machine image. The image fleet boots from the golden AMI by DEFAULT — its
    // root snapshot has the worker container pre-baked, so a cold instance skips
    // the ~273s ECR pull. The AMI id is published to GOLDEN_AMI_PARAM by
    // ComfyImageBuilderStack and read via fromSsmParameter, which resolves at
    // DEPLOY time (not per-launch) — so a new bake only lands on the next
    // `cdk deploy`, never surprise-rotating a running worker.
    //   ROLLBACK: `cdk deploy ComfyComputeStack -c useGoldenAmi=false` reverts
    //   the image fleet to the stock ECS GPU AMI (+ ECR pull) for one deploy.
    //   DEPENDENCY: default-on means GOLDEN_AMI_PARAM must exist (it is created
    //   by the first ComfyImageBuilderStack bake); a compute deploy fails fast on
    //   an unresolved SSM parameter otherwise. Video fleet never uses it.
    const ctxGoldenAmi = this.node.tryGetContext('useGoldenAmi');
    const useGoldenAmi =
      fleetName === 'image' && ctxGoldenAmi !== false && ctxGoldenAmi !== 'false';
    const machineImage = useGoldenAmi
      ? ec2.MachineImage.fromSsmParameter(GOLDEN_AMI_PARAM, {
          os: ec2.OperatingSystemType.LINUX,
        })
      : ecs.EcsOptimizedImage.amazonLinux2023(ecs.AmiHardwareType.GPU);

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

    // Golden-AMI cold boot reads the pre-baked image off the restored root
    // snapshot. Provision 300 MiB/s init so ~26 GiB hydrates in ~32s instead of
    // the default lazy-load (~26 MB/s → ~16 min → ComfyUI never ready in the
    // 300s grace; measured 2026-06-08). No typed CDK prop exists (autoscaling
    // EbsDeviceOptions lacks it) → L1 escape hatch. Only when the golden AMI is
    // active (snapshot-only feature; pointless on the stock AMI). Index .0 = the
    // single /dev/xvda device — sanity-check `cdk synth` if a 2nd device is added.
    if (useGoldenAmi && fleet.volumeInitializationRateMiBs) {
      const cfnLt = launchTemplate.node.defaultChild as ec2.CfnLaunchTemplate;
      cfnLt.addPropertyOverride(
        'LaunchTemplateData.BlockDeviceMappings.0.Ebs.VolumeInitializationRate',
        fleet.volumeInitializationRateMiBs,
      );
    }

    // Per-fleet, not a two-way image/other split. This was the latter, so the
    // minimax ASG silently sized itself from videoMin/videoMax and ignored
    // minimaxMin/minimaxMax entirely — the same shape of bug as the container
    // memory cap that made minimax inherit video's 28 GiB and get OOM-killed.
    // It happened to be harmless only because videoMax and minimaxMax were both
    // reachable; raising minimaxMax past videoMax would have capped invisibly at
    // the ASG while the scaler kept asking for more.
    const minCapacity = {
      image: config.scaling.imageMin,
      video: config.scaling.videoMin,
      minimax: config.scaling.minimaxMin,
    }[fleetName];
    const maxCapacity = {
      image: config.scaling.imageMax,
      video: config.scaling.videoMax,
      minimax: config.scaling.minimaxMax,
    }[fleetName];

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
          // price-capacity-optimized for both fleets (switched from
          // capacity-optimized-PRIORITIZED 2026-08-10). Prioritized honored
          // the launchTemplate overrides order (primaryInstanceType first)
          // ONLY as a capacity tiebreak, not a price signal — measured on the
          // image fleet's real job ledger, it sent just 5.8% of jobs to the
          // (cheaper, faster) g6.2xlarge primary and 26.9% to g6e.xlarge, an
          // ~2x-priced pool that should only fire in a real drought (see
          // config.ts fleets.image for the per-instance-type cost/latency
          // data). price-capacity-optimized picks from the lowest-priced
          // pools among those with good capacity, so it won't pay g6e's
          // premium just because prioritized capacity-optimized picked it.
          // (We also once briefly ran image on plain LOWEST_PRICE to force
          // g4/T4 for cost, but the T4 was ~3x slower, pricier per image, AND
          // OOM-unstable on g4dn.xlarge — reverted; see config.ts.)
          spotAllocationStrategy:
            autoscaling.SpotAllocationStrategy.PRICE_CAPACITY_OPTIMIZED,
          // Per-fleet bid ceiling; unset means bid on-demand (the AWS default).
          // A ceiling under the market price yields NO capacity rather than a
          // pricier instance — see fleet.spotMaxPrice in config.ts.
          ...(fleet.spotMaxPrice ? { spotMaxPrice: fleet.spotMaxPrice } : {}),
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
      // Required by the capacity provider's managed termination protection
      // (see makeCapacityProvider): the CP removes scale-in protection only
      // from drained/idle instances before the ASG terminates them.
      newInstancesProtectedFromScaleIn: true,
    });

    return asg;
  }

  /**
   * Capacity provider with managed scaling so ASG follows ECS task count.
   */
  private makeCapacityProvider(
    fleetName: FleetName,
    asg: autoscaling.AutoScalingGroup
  ): ecs.AsgCapacityProvider {
    return new ecs.AsgCapacityProvider(this, `${fleetName}CapacityProvider`, {
      capacityProviderName: `cp-${fleetName}-spot`,
      autoScalingGroup: asg,
      enableManagedScaling: true,
      // ENABLED so the capacity provider owns scale-in: it unprotects only
      // drained/idle instances, so the ASG never issues a raw terminate that
      // wedges in the drain lifecycle hook. A DISABLED CP once left a spot
      // instance stuck in Terminating:Wait for ~17h, jamming the ASG's
      // scale-in and holding ~2 GPUs up across idle windows (2026-07-12).
      // NOTE: scale-in protection does NOT block spot reclamation — AWS still
      // reclaims spot regardless; this only governs *voluntary* ASG scale-in.
      enableManagedTerminationProtection: true,
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
    fleetName: FleetName,
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
      //   - minimax: NO hard cap, like image. This branch used to be a
      //     two-way image/other split, so minimax silently inherited video's
      //     28 GiB cap — and MiniMax H3 loads ~50 GiB of weights (25.3 text
      //     encoder + 19.5 transformer + 4.9 video VAE). The cgroup OOM-killer
      //     therefore SIGKILLed ComfyUI (rc=-9) at 28 GiB no matter how large
      //     the host was; a 64 GB box and a 256 GB box failed identically,
      //     which is the tell that the limit was the container's, not the
      //     machine's. Every instance type in this fleet is >= 128 GB, so the
      //     cap has nothing useful to protect.
      memoryReservationMiB:
        fleetName === 'image' ? 11264 : fleetName === 'minimax' ? 49152 : 24576,
      memoryLimitMiB: fleetName === 'video' ? 28672 : undefined,
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
