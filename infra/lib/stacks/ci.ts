import { Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { AppConfig, GOLDEN_PIPELINE_NAME } from '../config';

export interface CiStackProps extends StackProps {
  readonly config: AppConfig;
}

/**
 * ECR repos + CodeBuild projects for worker images.
 *
 * v1: manually triggered builds via `scripts/trigger-build.sh image|video`.
 * No GitHub webhook. CodeStar Connection setup added later if desired.
 *
 * Build size: BUILD_GENERAL1_LARGE (15 GB RAM, 8 vCPU). Required because
 * Sage Attention compiles CUDA kernels and the full Wan stack pulls heavy
 * Python wheels.
 *
 * Build time estimates:
 *   - first cold build: 30-45 min
 *   - subsequent with buildx layer cache in ECR: <10 min
 *
 * Cost: ~$0.30 per cold build, <$0.10 per cached build.
 */
export class CiStack extends Stack {
  public readonly imageWorkerRepo: ecr.Repository;
  public readonly videoWorkerRepo: ecr.Repository;
  /** Blackwell (sm_120) build of the same Dockerfile; its own repo so the
   *  keep-last-10 lifecycle rule never evicts one variant for the other. */
  public readonly videoWorkerBlackwellRepo: ecr.Repository;
  public readonly metadataRepo: ecr.Repository;
  public readonly imageWorkerProject: codebuild.Project;
  public readonly videoWorkerProject: codebuild.Project;
  public readonly videoWorkerBlackwellProject: codebuild.Project;
  public readonly metadataProject: codebuild.Project;

  constructor(scope: Construct, id: string, props: CiStackProps) {
    super(scope, id, props);
    const { config } = props;

    // ----- ECR Repos -----
    // Lifecycle. Task definitions pin images by commit SHA (config.ts
    // imageTag), so a pinned image must outlive many rebuilds: the old
    // "keep last 10, any tag" rule counted the buildx `:cache` manifest each
    // build leaves untagged, which made it ~5 builds of retention — and an
    // evicted pin only surfaces as CannotPullContainerError at the next
    // instance launch. Expire untagged manifests on age, keep 30 of the rest
    // (ECR requires the ANY rule to carry the highest priority number).
    const lifecycleRules: ecr.LifecycleRule[] = [
      {
        rulePriority: 1,
        description: 'Expire untagged (superseded buildx cache manifests)',
        tagStatus: ecr.TagStatus.UNTAGGED,
        maxImageAge: Duration.days(14),
      },
      {
        rulePriority: 2,
        description: 'Keep last 30 images',
        maxImageCount: 30,
        tagStatus: ecr.TagStatus.ANY,
      },
    ];

    this.imageWorkerRepo = new ecr.Repository(this, 'ImageWorkerRepo', {
      repositoryName: 'comfy-image-worker',
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules,
      emptyOnDelete: false,
    });

    this.videoWorkerRepo = new ecr.Repository(this, 'VideoWorkerRepo', {
      repositoryName: 'comfy-video-worker',
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules,
      emptyOnDelete: false,
    });

    this.videoWorkerBlackwellRepo = new ecr.Repository(this, 'VideoWorkerBlackwellRepo', {
      repositoryName: 'comfy-video-worker-blackwell',
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules,
      emptyOnDelete: false,
    });

    // Metadata instance image (CPU-only, ~1 GB, served from always-on t3.small).
    this.metadataRepo = new ecr.Repository(this, 'MetadataRepo', {
      repositoryName: 'comfy-metadata',
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules,
      emptyOnDelete: false,
    });

    // ----- CodeBuild role -----
    const buildRole = new iam.Role(this, 'CodeBuildRole', {
      assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
      description: 'CodeBuild role for comfy worker image builds',
    });
    this.imageWorkerRepo.grantPullPush(buildRole);
    this.videoWorkerRepo.grantPullPush(buildRole);
    this.videoWorkerBlackwellRepo.grantPullPush(buildRole);
    this.metadataRepo.grantPullPush(buildRole);
    buildRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['ecr:GetAuthorizationToken'],
        resources: ['*'],
      })
    );
    buildRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['logs:CreateLogStream', 'logs:PutLogEvents', 'logs:CreateLogGroup'],
        resources: [`arn:aws:logs:${this.region}:${this.account}:log-group:/aws/codebuild/*`],
      })
    );
    // The image-worker buildspec re-bakes the golden AMI after pushing a new
    // :latest (post_build) so the AMI never goes stale vs the container image.
    // Scoped to the one golden pipeline (constructed ARN — no cross-stack ref;
    // the pipeline lives in ComfyImageBuilderStack). Shared buildRole, but only
    // buildspec-image-worker.yml calls start-image-pipeline-execution.
    buildRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['imagebuilder:StartImagePipelineExecution'],
        resources: [
          `arn:aws:imagebuilder:${this.region}:${this.account}:image-pipeline/${GOLDEN_PIPELINE_NAME}`,
        ],
      })
    );

    const buildLogs = new logs.LogGroup(this, 'CodeBuildLogs', {
      logGroupName: '/comfy/codebuild',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // ----- CodeBuild Projects -----
    // The buildspecs live in `codebuild/` at repo root. They reference docker
    // buildx with registry layer caching to ECR for fast subsequent builds.
    //
    // GitHub owner/repo come from CDK context so this file stays portable for
    // any user. Set via:
    //   cdk deploy ComfyCiStack -c githubOwner=YOUR_USER -c githubRepo=THIS_REPO
    // or via cdk.context.json (which is gitignored).
    const githubOwner = this.node.tryGetContext('githubOwner') ?? 'YOUR_GITHUB_USERNAME';
    const githubRepo = this.node.tryGetContext('githubRepo') ?? config.projectName;
    const sourceFromGitHub = codebuild.Source.gitHub({
      owner: githubOwner,
      repo: githubRepo,
      // No webhooks — manual trigger via scripts/trigger-build.sh
      webhook: false,
    });

    const buildEnvironment: codebuild.BuildEnvironment = {
      buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
      computeType: codebuild.ComputeType.LARGE, // 15 GB RAM, 8 vCPU
      privileged: true, // Required for docker build
      environmentVariables: {
        AWS_REGION: { value: this.region },
        ACCOUNT: { value: this.account },
      },
    };

    this.imageWorkerProject = new codebuild.Project(this, 'ImageWorkerBuildProject', {
      projectName: 'comfy-build-image-worker',
      role: buildRole,
      source: sourceFromGitHub,
      environment: buildEnvironment,
      buildSpec: codebuild.BuildSpec.fromSourceFilename('codebuild/buildspec-image-worker.yml'),
      timeout: Duration.minutes(90),
      logging: { cloudWatch: { logGroup: buildLogs, prefix: 'image-worker' } },
    });

    this.metadataProject = new codebuild.Project(this, 'MetadataBuildProject', {
      projectName: 'comfy-build-metadata',
      role: buildRole,
      source: sourceFromGitHub,
      environment: buildEnvironment,
      buildSpec: codebuild.BuildSpec.fromSourceFilename('codebuild/buildspec-metadata.yml'),
      timeout: Duration.minutes(30),
      logging: { cloudWatch: { logGroup: buildLogs, prefix: 'metadata' } },
    });

    this.videoWorkerProject = new codebuild.Project(this, 'VideoWorkerBuildProject', {
      projectName: 'comfy-build-video-worker',
      role: buildRole,
      source: sourceFromGitHub,
      environment: buildEnvironment,
      buildSpec: codebuild.BuildSpec.fromSourceFilename('codebuild/buildspec-video-worker.yml'),
      timeout: Duration.minutes(90),
      logging: { cloudWatch: { logGroup: buildLogs, prefix: 'video-worker' } },
    });

    // Same Dockerfile, blackwell profile (NGC 26.07, +sm_120). Three CUDA
    // archs and a bigger base, so a longer timeout. See the buildspec header.
    this.videoWorkerBlackwellProject = new codebuild.Project(this, 'VideoWorkerBlackwellBuildProject', {
      projectName: 'comfy-build-video-worker-blackwell',
      role: buildRole,
      source: sourceFromGitHub,
      environment: buildEnvironment,
      buildSpec: codebuild.BuildSpec.fromSourceFilename('codebuild/buildspec-video-worker-blackwell.yml'),
      timeout: Duration.minutes(120),
      logging: { cloudWatch: { logGroup: buildLogs, prefix: 'video-worker-blackwell' } },
    });
  }
}
