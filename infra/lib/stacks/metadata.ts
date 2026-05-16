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
      // Pull and run
      `docker pull ${ecrUri}:latest`,
      `docker run -d --restart unless-stopped --name comfy-metadata \\
          -p 8188:8188 \\
          -e FLEET=image \\
          -e DISPATCHER_API_URL='${dispatcherApiUrl}' \\
          -e WORKER_API_KEY_ID='${workerApiKeyId}' \\
          -e FRONTEND_BUCKET='${storage.frontendBucket.bucketName}' \\
          -e OBJECT_INFO_TABLE='${storage.objectInfoTable.tableName}' \\
          -e AWS_REGION='${this.region}' \\
          -e LOG_LEVEL=INFO \\
          --log-driver=awslogs \\
          --log-opt awslogs-region=${this.region} \\
          --log-opt awslogs-group=/comfy/metadata \\
          --log-opt awslogs-create-group=true \\
          ${ecrUri}:latest`,
    );

    // ----- Instance -----
    this.instance = new ec2.Instance(this, 'MetadataInstance', {
      vpc: network.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.SMALL),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      securityGroup: network.workerSecurityGroup,
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

    new CfnOutput(this, 'MetadataInstanceId', {
      value: this.instance.instanceId,
      description: 'Metadata EC2 instance ID',
    });
    new CfnOutput(this, 'MetadataPublicIp', {
      value: this.instance.instancePublicIp,
      description: 'Metadata instance public IP (for debugging / future proxy)',
    });
  }
}
