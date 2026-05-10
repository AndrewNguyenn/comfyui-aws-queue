import { Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { AppConfig } from '../config';

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
  public readonly imageWorkerProject: codebuild.Project;
  public readonly videoWorkerProject: codebuild.Project;

  constructor(scope: Construct, id: string, props: CiStackProps) {
    super(scope, id, props);
    const { config } = props;

    // ----- ECR Repos -----
    // Lifecycle: keep last 10 by imagePushedAt, regardless of tag (since we
    // tag every image with both `latest` and the commit SHA).
    const lifecycleRules: ecr.LifecycleRule[] = [
      {
        rulePriority: 1,
        description: 'Keep last 10 images',
        maxImageCount: 10,
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

    // ----- CodeBuild role -----
    const buildRole = new iam.Role(this, 'CodeBuildRole', {
      assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
      description: 'CodeBuild role for comfy worker image builds',
    });
    this.imageWorkerRepo.grantPullPush(buildRole);
    this.videoWorkerRepo.grantPullPush(buildRole);
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

    const buildLogs = new logs.LogGroup(this, 'CodeBuildLogs', {
      logGroupName: '/comfy/codebuild',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // ----- CodeBuild Projects -----
    // The buildspecs live in `codebuild/` at repo root. They reference docker
    // buildx with registry layer caching to ECR for fast subsequent builds.
    const sourceFromGitHub = codebuild.Source.gitHub({
      owner: 'AndrewNguyenn',
      repo: config.projectName,
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

    this.videoWorkerProject = new codebuild.Project(this, 'VideoWorkerBuildProject', {
      projectName: 'comfy-build-video-worker',
      role: buildRole,
      source: sourceFromGitHub,
      environment: buildEnvironment,
      buildSpec: codebuild.BuildSpec.fromSourceFilename('codebuild/buildspec-video-worker.yml'),
      timeout: Duration.minutes(90),
      logging: { cloudWatch: { logGroup: buildLogs, prefix: 'video-worker' } },
    });
  }
}
