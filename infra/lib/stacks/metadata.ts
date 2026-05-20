import { Stack, StackProps, CfnOutput, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { AppConfig } from '../config';
import { NetworkStack } from './network';
import { StorageStack } from './storage';
import { CiStack } from './ci';

export interface MetadataStackProps extends StackProps {
  readonly config: AppConfig;
  readonly network: NetworkStack;
  readonly storage: StorageStack;
  readonly ci: CiStack;
}

/**
 * Always-on metadata instance.
 *
 * t3.small EC2 running ComfyUI in CPU mode. Purpose: keep ComfyUI metadata
 * (node definitions, custom-node extensions, Manager UI) available to the
 * editor at all times, decoupled from GPU spot capacity. The metadata
 * instance does NOT process jobs — those still go to GPU spot workers via
 * SQS. It only publishes /object_info + /extensions to DDB/S3 on boot.
 *
 * Why on-demand t3.small ($15/mo) instead of free / serverless:
 *   - ComfyUI is a Python process that needs to be alive to enumerate nodes
 *   - Lambda can't run it (no long-lived runtime, no python+torch image)
 *   - Spot would defeat the "always available" purpose
 *
 * Tight on memory at t3.small (2 GB) — Python + CPU torch + ComfyUI +
 * Manager is ~1-1.5 GB resident. If we add many custom nodes via Manager
 * later, may need to bump to t3.medium ($30/mo).
 */
export class MetadataStack extends Stack {
  public readonly instance: ec2.Instance;

  constructor(scope: Construct, id: string, props: MetadataStackProps) {
    super(scope, id, props);
    const { network, storage, ci } = props;

    const ecrRepoName = 'comfy-metadata';
    const ecrUri = `${this.account}.dkr.ecr.${this.region}.amazonaws.com/${ecrRepoName}`;

    // ----- IAM role -----
    const role = new iam.Role(this, 'MetadataInstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });
    // ECR pull
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],
    }));
    // S3 frontend bucket (extensions/* uploads — same as worker grant)
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['s3:PutObject', 's3:PutObjectAcl'],
      resources: [`${storage.frontendBucket.bucketArn}/extensions/*`],
    }));
    // S3 outputs bucket: read the custom-node manifest on boot. manifest_installer
    // (run by entrypoint.sh) syncs /opt/comfy/custom_nodes against this file so
    // nodes installed via Manager persist across container/instance recreation.
    // Without this grant the read fails AccessDenied and is silently swallowed
    // as "manifest empty" — i.e. custom nodes would NOT survive a restart.
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject'],
      resources: [`${storage.outputsBucket.bucketArn}/manifests/*`],
    }));
    // DDB object_info table (write our published metadata)
    storage.objectInfoTable.grantReadWriteData(role);
    // API key value lookup for /internal/object_info + /internal/extensions auth
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['apigateway:GET'],
      resources: [`arn:aws:apigateway:${this.region}::/apikeys/*`],
    }));
    // CloudWatch logs for the docker daemon + container output
    role.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['logs:CreateLogStream', 'logs:CreateLogGroup', 'logs:PutLogEvents', 'logs:DescribeLogStreams'],
      resources: [`arn:aws:logs:${this.region}:${this.account}:log-group:/comfy/*`],
    }));

    // ----- Look up shared SSM values -----
    const dispatcherApiUrl = ssm.StringParameter.valueForStringParameter(this, '/comfy/api/url');
    const workerApiKeyId = ssm.StringParameter.valueForStringParameter(this, '/comfy/api/worker-key-id');

    // ----- User data: install docker, pull image, run container -----
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'set -ex',
      'dnf update -y',
      'dnf install -y docker',
      'systemctl enable --now docker',
      // Wait for docker socket
      'for i in $(seq 1 30); do [ -S /var/run/docker.sock ] && break; sleep 2; done',
      // Set up CloudWatch logging
      'mkdir -p /var/log/comfy-metadata',
      // Login to ECR
      `aws ecr get-login-password --region ${this.region} | docker login --username AWS --password-stdin ${this.account}.dkr.ecr.${this.region}.amazonaws.com`,
      // Pull with retry. On a fresh `cdk deploy` the comfy-metadata image is
      // built by CodeBuild *after* this instance boots — the image won't
      // exist for the first ~10-20 min. Retrying (rather than failing the
      // boot outright) means the metadata instance self-heals once the
      // build lands, with no manual intervention. ~20 min window.
      `for i in $(seq 1 40); do docker pull ${ecrUri}:latest && break; ` +
        `echo "comfy-metadata image not ready (attempt $i/40), retrying in 30s"; sleep 30; done`,
      // Idempotent: drop any stale container so a re-run of this script
      // (e.g. cloud-init re-exec) doesn't fail on the --name collision.
      `docker rm -f comfy-metadata 2>/dev/null || true`,
      `docker run -d --restart unless-stopped --name comfy-metadata \\
          -p 8188:8188 \\
          -e FLEET=image \\
          -e DISPATCHER_API_URL='${dispatcherApiUrl}' \\
          -e WORKER_API_KEY_ID='${workerApiKeyId}' \\
          -e FRONTEND_BUCKET='${storage.frontendBucket.bucketName}' \\
          -e OBJECT_INFO_TABLE='${storage.objectInfoTable.tableName}' \\
          -e OUTPUTS_BUCKET='${storage.outputsBucket.bucketName}' \\
          -e AWS_REGION='${this.region}' \\
          -e LOG_LEVEL=INFO \\
          --log-driver=awslogs \\
          --log-opt awslogs-region=${this.region} \\
          --log-opt awslogs-group=/comfy/metadata \\
          --log-opt awslogs-create-group=true \\
          ${ecrUri}:latest`,
    );

    // Dedicated SG for metadata instance: allow inbound 8188 from anywhere.
    // Why public: API Gateway HTTP_PROXY integrations can't reach private IPs
    // without VPC Link (NLB + LCU costs ~$20/mo); 0.0.0.0/0 keeps it free.
    // Cognito auth at API GW is the primary gate. Direct access to the
    // metadata instance bypass Cognito, but the only sensitive endpoint is
    // /customnode/install (could pip-install arbitrary packages on a $15/mo
    // metadata box — acceptable risk for 1-user personal project).
    const metadataSg = new ec2.SecurityGroup(this, 'MetadataSg', {
      vpc: network.vpc,
      securityGroupName: `${props.config.projectName}-metadata-sg`,
      description: 'Metadata instance: inbound 8188 from anywhere (for API GW HTTP_PROXY)',
      allowAllOutbound: true,
    });
    metadataSg.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8188),
      'ComfyUI HTTP for dispatcher Lambda Manager proxy',
    );

    // ----- Instance -----
    this.instance = new ec2.Instance(this, 'MetadataInstance', {
      vpc: network.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.SMALL),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      securityGroup: metadataSg,
      userData,
      role,
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(20, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            deleteOnTermination: true,
          }),
        },
      ],
      requireImdsv2: true,
    });

    // Elastic IP — stable URL for API GW HTTP_PROXY (instance public IP
    // would change on stop/start; EIP attached to running instance is free).
    const eip = new ec2.CfnEIP(this, 'MetadataEip', {
      domain: 'vpc',
      instanceId: this.instance.instanceId,
    });

    // Publish the EIP via SSM so ApiStack can read it at deploy time.
    new ssm.StringParameter(this, 'MetadataEipParam', {
      parameterName: '/comfy/metadata/eip',
      stringValue: eip.ref,
      description: 'Metadata instance EIP — used by API GW HTTP_PROXY to proxy Manager endpoints',
    });
    new ssm.StringParameter(this, 'MetadataUrlParam', {
      parameterName: '/comfy/metadata/url',
      stringValue: `http://${eip.ref}:8188`,
      description: 'Metadata instance ComfyUI base URL',
    });

    new CfnOutput(this, 'MetadataInstanceId', {
      value: this.instance.instanceId,
      description: 'Metadata EC2 instance ID',
    });
    new CfnOutput(this, 'MetadataEipOutput', {
      value: eip.ref,
      description: 'Metadata instance Elastic IP',
    });
  }
}
