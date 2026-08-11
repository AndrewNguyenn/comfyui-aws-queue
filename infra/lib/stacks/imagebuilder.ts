import { Stack, StackProps, Duration } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as imagebuilder from 'aws-cdk-lib/aws-imagebuilder';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as fs from 'fs';
import * as path from 'path';
import { AppConfig, GOLDEN_AMI_PARAM, GOLDEN_PIPELINE_NAME } from '../config';
import { NetworkStack } from './network';
import { CiStack } from './ci';

/**
 * EC2 Image Builder pipeline that bakes the comfy-image-worker container image
 * into a golden AMI for the image fleet.
 *
 * Why: the cold-start bottleneck is the ~273s ECR pull + extract of the 13 GB
 * (→25.7 GB) worker image onto the root volume on every scale-from-zero boot
 * (measured 2026-06-08). A golden AMI ships that image pre-extracted in the
 * root snapshot, so a cold instance skips the pull. Paired with the launch
 * template's VolumeInitializationRate=300 (compute.ts) — without which the
 * snapshot lazy-loads at ~26 MB/s and is WORSE than the pull — ComfyUI reaches
 * ready in ~32s (measured) instead of ~5+ min.
 *
 * A small publisher Lambda (triggered by the Image Builder "AVAILABLE" event)
 * writes the new AMI id to the SSM parameter `GOLDEN_AMI_PARAM`; the image-fleet
 * launch template reads it via `MachineImage.fromSsmParameter` (deploy-time
 * resolution → a new bake only lands on the next `cdk deploy`, never
 * surprise-rotating a running worker).
 *
 * Why a Lambda and not Image Builder's native ssmParameterConfigurations: the
 * Image Builder service-linked role (AWSServiceRoleForImageBuilder) has NO
 * ssm:PutParameter permission and can't be granted one (AWS-managed), so the
 * native distribution write fails ("not authorized to perform: ssm:PutParameter")
 * — verified on a real bake 2026-06-08. The Lambda writes the param under its
 * own role, which we control.
 *
 * Re-bake triggers: the monthly schedule below (base-AMI driver/security
 * patches) + the image-worker CodeBuild post_build step (so a worker rebuild
 * re-bakes — see codebuild/buildspec-image-worker.yml + ci.ts grant).
 */

// IMMUTABLE in Image Builder: re-creating an existing name+version FAILS the
// deploy. BUMP these whenever the component YAML or the recipe changes.
const COMPONENT_VERSION = '1.0.0';
const RECIPE_VERSION = '1.0.0';

export interface ImageBuilderStackProps extends StackProps {
  readonly config: AppConfig;
  readonly network: NetworkStack;
  readonly ci: CiStack;
}

export class ImageBuilderStack extends Stack {
  constructor(scope: Construct, id: string, props: ImageBuilderStackProps) {
    super(scope, id, props);
    const { config, network, ci } = props;
    const image = config.fleets.image;

    // ---- Build-instance role + profile ----
    // The bake host pulls from ECR and runs the AWSTOE agent (SSM).
    const buildRole = new iam.Role(this, 'GoldenAmiBuildRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('EC2InstanceProfileForImageBuilder'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('EC2InstanceProfileForImageBuilderECRContainerBuilds'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });
    ci.imageWorkerRepo.grantPull(buildRole); // explicit least-privilege pull on the worker repo
    const buildProfile = new iam.InstanceProfile(this, 'GoldenAmiBuildProfile', { role: buildRole });

    // ---- Component: the bake script ----
    const component = new imagebuilder.CfnComponent(this, 'BakeWorkerComponent', {
      name: 'comfy-image-bake-worker',
      platform: 'Linux',
      version: COMPONENT_VERSION,
      supportedOsVersions: ['Amazon Linux 2023'],
      data: fs.readFileSync(path.join(__dirname, '../imagebuilder/bake-worker.yaml'), 'utf8'),
    });

    // ---- Recipe: stock ECS GPU AMI (via SSM, so monthly builds pick up the
    // latest base + NVIDIA driver/security patches) + our bake component.
    const recipe = new imagebuilder.CfnImageRecipe(this, 'GoldenAmiRecipe', {
      name: 'comfy-image-golden',
      version: RECIPE_VERSION,
      parentImage: 'ssm:/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id',
      components: [{ componentArn: component.attrArn }],
      blockDeviceMappings: [
        {
          deviceName: '/dev/xvda',
          ebs: {
            // Must equal the fleet root size so the snapshot restores cleanly
            // onto the fleet's volume (a larger snapshot would fail the launch).
            volumeSize: image.rootVolumeGb,
            volumeType: 'gp3',
            iops: image.rootVolumeIops,
            throughput: image.rootVolumeThroughputMbps,
            deleteOnTermination: true,
            encrypted: true, // AWS-managed aws/ebs key
          },
        },
      ],
    });

    // ---- Infrastructure: a CHEAP NON-GPU build host. docker pull/extract is
    // CPU+IO, not GPU — and the parent GPU AMI's NVIDIA bits are already on disk,
    // so the resulting AMI stays GPU-capable regardless of build-host type.
    // Public subnet: ECR API needs egress and there is no NAT (layer blobs use
    // the S3 gateway endpoint; the ECR API calls go out via the IGW).
    const infra = new imagebuilder.CfnInfrastructureConfiguration(this, 'GoldenAmiInfra', {
      name: 'comfy-image-golden-infra',
      instanceProfileName: buildProfile.instanceProfileName, // PROFILE name, not role
      instanceTypes: ['c6i.2xlarge'],
      subnetId: network.vpc.publicSubnets[0].subnetId,
      securityGroupIds: [network.workerSecurityGroup.securityGroupId],
      terminateInstanceOnFailure: true,
      instanceMetadataOptions: { httpTokens: 'required', httpPutResponseHopLimit: 2 },
    });

    // ---- Distribution: just names the output AMI. The SSM-param write is done
    // by the publisher Lambda below (the Image Builder SLR lacks ssm:PutParameter).
    const dist = new imagebuilder.CfnDistributionConfiguration(this, 'GoldenAmiDist', {
      name: 'comfy-image-golden-dist',
      distributions: [
        {
          region: this.region,
          // PascalCase Name: amiDistributionConfiguration is typed `any`, so CDK
          // passes keys verbatim (no camel→Pascal transform) — lowercase `name`
          // is silently dropped by CFN and the AMI gets an auto-generated name.
          amiDistributionConfiguration: { Name: 'comfy-image-golden-{{ imagebuilder:buildDate }}' },
        },
      ],
    });

    // ---- Pipeline: monthly (EXPRESSION_MATCH_ONLY → builds every tick so
    // base-AMI patches land even with no worker rebuild). Test instance off.
    new imagebuilder.CfnImagePipeline(this, 'GoldenAmiPipeline', {
      name: GOLDEN_PIPELINE_NAME,
      imageRecipeArn: recipe.attrArn,
      infrastructureConfigurationArn: infra.attrArn,
      distributionConfigurationArn: dist.attrArn,
      imageTestsConfiguration: { imageTestsEnabled: false },
      schedule: {
        scheduleExpression: 'cron(0 7 1 * ? *)', // 07:00 UTC on the 1st (6-field, '?' required)
        pipelineExecutionStartCondition: 'EXPRESSION_MATCH_ONLY',
      },
      status: 'ENABLED',
    });

    // ---- Lifecycle policy: keep only the newest 3 golden AMIs ----
    // Every monthly bake (+ every worker-image rebuild, via the CodeBuild
    // post_build re-bake trigger noted above) leaves its 150 GB EBS snapshot
    // behind — there was no cleanup at all before this, ~$6.5/mo and growing
    // (4 AMIs / 4 snapshots measured 2026-08-10, one bake older than the
    // current cadence would fully warrant). Image Builder's native lifecycle
    // policy does deregister + snapshot delete in one DELETE action.
    //
    // Scoping so this can NEVER touch a non-golden AMI: resourceSelection
    // targets ONLY images built from the `comfy-image-golden` recipe (any
    // version — recipeSelection.semanticVersion supports the `x` wildcard,
    // so a future RECIPE_VERSION bump doesn't silently stop being covered).
    // That's on top of the AWS-managed execution-role policy below, which is
    // itself independently scoped to resources tagged `CreatedBy: EC2 Image
    // Builder` (verified via `aws iam get-policy-version` on
    // EC2ImageBuilderLifecycleExecutionPolicy) — two independent boundaries,
    // neither of which can reach an unrelated AMI/snapshot in this account.
    const lifecycleRole = new iam.Role(this, 'GoldenAmiLifecycleRole', {
      assumedBy: new iam.ServicePrincipal('imagebuilder.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/EC2ImageBuilderLifecycleExecutionPolicy',
        ),
      ],
    });
    new imagebuilder.CfnLifecyclePolicy(this, 'GoldenAmiLifecyclePolicy', {
      name: 'comfy-image-golden-lifecycle',
      description: 'Keep the newest 3 comfy-image-golden AMIs; deregister + delete snapshots for the rest.',
      executionRole: lifecycleRole.roleArn,
      resourceType: 'AMI_IMAGE',
      status: 'ENABLED',
      resourceSelection: {
        recipes: [{ name: recipe.name!, semanticVersion: 'x.x.x' }],
      },
      policyDetails: [
        {
          action: {
            type: 'DELETE',
            includeResources: { amis: true, snapshots: true },
          },
          // COUNT filter's `value` is the number of NEWEST resources to
          // KEEP, not a deletion threshold (verified against the API doc —
          // the field name is misleading). retainAtLeast doesn't apply to
          // COUNT filters (age-based only), so it's omitted here.
          filter: { type: 'COUNT', value: 3 },
        },
      ],
    });

    // ---- Publisher Lambda: on Image Builder "AVAILABLE", write the new AMI id
    // to the SSM param (create-or-update). Replaces the native distribution SSM
    // write, which the AWS-managed Image Builder SLR can't perform.
    const publisher = new lambda.Function(this, 'GoldenAmiPublisherFn', {
      functionName: 'comfy-golden-ami-publisher',
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      handler: 'handler.lambda_handler',
      code: lambda.Code.fromAsset(
        path.join(__dirname, '../../../services/golden_ami_publisher'),
        { exclude: ['__pycache__', '*.pyc'] }
      ),
      timeout: Duration.seconds(30),
      memorySize: 128,
      logRetention: logs.RetentionDays.ONE_WEEK,
      environment: {
        GOLDEN_AMI_PARAM,
        RECIPE_NAME: recipe.name!, // only publish for our recipe's images
      },
    });
    publisher.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['ssm:PutParameter'],
        resources: [`arn:aws:ssm:${this.region}:${this.account}:parameter${GOLDEN_AMI_PARAM}`],
      })
    );
    // GetImage has no resource-level scoping for the build-version ARN shape we
    // receive, so grant on the account's image resources (read-only metadata).
    publisher.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['imagebuilder:GetImage'],
        resources: [`arn:aws:imagebuilder:${this.region}:${this.account}:image/*`],
      })
    );

    // Fire only when a build reaches AVAILABLE. The handler re-checks the recipe
    // name and no-ops on anything else, so a shared rule is safe.
    new events.Rule(this, 'GoldenAmiAvailableRule', {
      ruleName: 'comfy-golden-ami-available',
      eventPattern: {
        source: ['aws.imagebuilder'],
        detailType: ['EC2 Image Builder Image State Change'],
        detail: { state: { status: ['AVAILABLE'] } },
      },
      targets: [new targets.LambdaFunction(publisher)],
    });
  }
}
