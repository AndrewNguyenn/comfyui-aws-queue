import { Stack, StackProps, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { AppConfig } from '../config';

export interface NetworkStackProps extends StackProps {
  readonly config: AppConfig;
}

/**
 * Network primitives for the project.
 *
 * Design choices (per IMPLEMENTATION_PLAN.md sections 4.1, 16, 20):
 * - Single AZ. AZ outage = full outage. Acceptable for personal-project scope and
 *   saves ~$7/mo from VPC interface endpoints not needed across multiple AZs.
 * - Public subnets only. No NAT gateway ($32/mo saved). Workers get public IPs
 *   (~$0.005/hr per IPv4 in use); they only initiate outbound, never accept inbound.
 * - S3 gateway endpoint (free). Eliminates internet round-trip for model downloads.
 * - Worker security group: outbound only. No inbound rules. Operators must NEVER
 *   open this — workers expose ComfyUI on 127.0.0.1 only.
 */
export class NetworkStack extends Stack {
  public readonly vpc: ec2.Vpc;
  public readonly workerSecurityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: NetworkStackProps) {
    super(scope, id, props);

    const { config } = props;

    this.vpc = new ec2.Vpc(this, 'Vpc', {
      vpcName: `${config.projectName}-vpc`,
      ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
      availabilityZones: [config.availabilityZone],
      natGateways: 0, // Critical: no NAT (saves $32/mo). Workers in public subnets.
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
          mapPublicIpOnLaunch: true,
        },
      ],
      gatewayEndpoints: {
        S3: {
          service: ec2.GatewayVpcEndpointAwsService.S3,
        },
        // DynamoDB gateway endpoint is also free — add it.
        DynamoDB: {
          service: ec2.GatewayVpcEndpointAwsService.DYNAMODB,
        },
      },
    });

    this.workerSecurityGroup = new ec2.SecurityGroup(this, 'WorkerSecurityGroup', {
      vpc: this.vpc,
      securityGroupName: `${config.projectName}-worker-sg`,
      description:
        'Workers outbound only; no inbound rules. Workers only initiate connections.',
      allowAllOutbound: true,
    });

    // Explicitly remove the default inbound rule on the SG. Defense-in-depth in case
    // of accidental "allow all" rule by an operator. SG starts with no inbound, but
    // make this explicit in the resource description.

    // Use destroy on stack delete — VPC is cheap to recreate, no data loss.
    this.vpc.applyRemovalPolicy(RemovalPolicy.DESTROY);
    this.workerSecurityGroup.applyRemovalPolicy(RemovalPolicy.DESTROY);
  }
}
