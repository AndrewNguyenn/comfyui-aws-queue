import { ArnFormat, Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as budgets from 'aws-cdk-lib/aws-budgets';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as snsSubs from 'aws-cdk-lib/aws-sns-subscriptions';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as cw from 'aws-cdk-lib/aws-cloudwatch';
import * as cwActions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as path from 'path';
import { AppConfig } from '../config';

export interface MonitoringStackProps extends StackProps {
  readonly config: AppConfig;
}

/**
 * Cost guardrails and observability.
 *
 * The most important resource here is the AWS Budget — it's the safety net that
 * prevents a runaway bill. Three thresholds:
 *   50% — email warning
 *   80% — email + (in v3.1+) SNS topic to ASG-shutdown Lambda (preventive)
 *   100% — kill-switch (full shutdown of compute)
 *
 * The kill-switch Lambda (services/killswitch) is created HERE and subscribed to
 * the killSwitchTopic — on a budget breach it zeros both fleet ASGs and ECS
 * services. It targets the ASGs/services by their deterministic names (no
 * cross-stack ref needed), so it lives next to the topic + budget it serves.
 *
 * Email subscriber address is pulled from SSM at deploy time, NOT hardcoded.
 * Operator must run before first deploy:
 *   aws ssm put-parameter --name /comfy/alerts/budget-email \
 *       --value "you@example.com" --type String
 */
export class MonitoringStack extends Stack {
  public readonly killSwitchTopic: sns.Topic;
  public readonly preventiveScaleDownTopic: sns.Topic;

  constructor(scope: Construct, id: string, props: MonitoringStackProps) {
    super(scope, id, props);
    const { config } = props;

    // Read alert email from SSM Parameter Store (manually populated by operator).
    // Using ssm.StringParameter.valueForStringParameter resolves at deploy time.
    const alertEmail = ssm.StringParameter.valueForStringParameter(
      this,
      config.cost.budgetEmailParameter
    );

    // ---------- SNS topics ----------
    this.killSwitchTopic = new sns.Topic(this, 'KillSwitchTopic', {
      topicName: 'comfy-killswitch',
      displayName: 'comfy-aws-queue kill-switch trigger',
    });

    this.preventiveScaleDownTopic = new sns.Topic(this, 'PreventiveScaleDownTopic', {
      topicName: 'comfy-preventive-scale-down',
      displayName: 'comfy-aws-queue preventive scale-down at 80% budget',
    });

    // Email subscriber for both — operator gets paged on either event.
    this.killSwitchTopic.addSubscription(new snsSubs.EmailSubscription(alertEmail));
    this.preventiveScaleDownTopic.addSubscription(new snsSubs.EmailSubscription(alertEmail));

    // AWS Budgets can only deliver to an SNS topic that grants
    // budgets.amazonaws.com sns:Publish via a RESOURCE policy. CfnBudget takes
    // the topic as a bare ARN string and does NOT add this — without it the
    // budget→topic delivery silently fails ("Invalid SNS topic") and neither the
    // kill-switch nor the preventive scale-down ever fires. (This was almost
    // certainly why the old setup never worked.)
    for (const topic of [this.killSwitchTopic, this.preventiveScaleDownTopic]) {
      topic.addToResourcePolicy(new iam.PolicyStatement({
        sid: 'AllowBudgetsPublish',
        effect: iam.Effect.ALLOW,
        principals: [new iam.ServicePrincipal('budgets.amazonaws.com')],
        actions: ['sns:Publish'],
        resources: [topic.topicArn],
        conditions: {
          StringEquals: { 'aws:SourceAccount': this.account },
          ArnLike: { 'aws:SourceArn': `arn:aws:budgets::${this.account}:*` },
        },
      }));
    }

    // ---------- Kill-switch Lambda ----------
    // Names are deterministic (see compute.ts: cluster `${projectName}-cluster`,
    // ASGs `comfy-${fleet}-asg`, services 'comfy-image'/'comfy-video'), so we
    // target them by name and avoid a cross-stack ref to ComputeStack.
    //
    // ALL three ASGs get MaxSize=0 — that is the only durable stop (the
    // handler docstring explains why). minimax matters most here: its ASG is
    // driven directly by the fleet scaler, so MaxSize=0 is its entire safety
    // net. Its services are DAEMON (comfy-minimax-ada/-blackwell) and have no
    // desiredCount to zero, so they are deliberately NOT in the service list;
    // with MaxSize=0 there is nothing for a daemon to place on anyway.
    const asgFleets = ['image', 'video', 'minimax'];
    const serviceFleets = ['image', 'video'];
    const asgNames = asgFleets.map((f) => `comfy-${f}-asg`);
    const serviceNames = serviceFleets.map((f) => `comfy-${f}`);
    const clusterName = `${config.projectName}-cluster`;

    const killSwitchFn = new lambda.Function(this, 'KillSwitchFn', {
      functionName: 'comfy-killswitch-handler',
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handler.lambda_handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/killswitch')),
      timeout: Duration.seconds(60),
      memorySize: 128,
      environment: {
        ASG_NAMES: asgNames.join(','),
        ECS_CLUSTER: clusterName,
        ECS_SERVICES: serviceNames.join(','),
      },
      description: 'Cost kill-switch: zeros every GPU ASG (image, video, minimax) and the replica ECS services on a budget breach.',
    });

    // Describe has no resource-level support; scope Update to the two ASGs and
    // the two services only.
    killSwitchFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['autoscaling:DescribeAutoScalingGroups'],
      resources: ['*'],
    }));
    killSwitchFn.addToRolePolicy(new iam.PolicyStatement({
      // SetInstanceProtection: minimax workers hold scale-in protection while
      // rendering, and an ASG cannot terminate a protected instance, so the
      // handler clears it before zeroing (services/killswitch/handler.py).
      actions: ['autoscaling:UpdateAutoScalingGroup', 'autoscaling:SetInstanceProtection'],
      resources: asgNames.map((n) => Stack.of(this).formatArn({
        service: 'autoscaling',
        resource: 'autoScalingGroup',
        resourceName: `*:autoScalingGroupName/${n}`,
        arnFormat: ArnFormat.COLON_RESOURCE_NAME,
      })),
    }));
    killSwitchFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['ecs:UpdateService'],
      resources: serviceNames.map((s) =>
        `arn:aws:ecs:${this.region}:${this.account}:service/${clusterName}/${s}`),
    }));

    // The budget's 100% ACTUAL notification publishes here -> fire the kill-switch.
    this.killSwitchTopic.addSubscription(new snsSubs.LambdaSubscription(killSwitchFn));

    // ---------- AWS Budget ----------
    new budgets.CfnBudget(this, 'MonthlyBudget', {
      budget: {
        // Budgets are an account-global resource — the name must be unique
        // across all regions, so suffix with the region (same reasoning as
        // the S3 bucket names in storage.ts).
        budgetName: `${config.projectName}-monthly-${this.region}`,
        budgetType: 'COST',
        timeUnit: 'MONTHLY',
        budgetLimit: {
          amount: config.cost.budgetAmountUsd,
          unit: 'USD',
        },
        // NO costFilters: track TOTAL account spend, not just project-tagged
        // resources. The old TagKeyValue filter required a cost-allocation tag
        // that was never populated, so the budget read $0 and the kill-switch
        // never fired. A safety net must see everything.
        // Track UNBLENDED cost excluding credits/refunds (i.e. raw usage), so
        // the kill-switch fires on $250 of USAGE even while promotional credits
        // are covering the actual invoice — consistent with the gross figure
        // the console budget already shows. Flip includeCredit/includeRefund to
        // true to track net out-of-pocket instead.
        costTypes: {
          includeCredit: false,
          includeRefund: false,
          includeDiscount: true,
          includeOtherSubscription: true,
          includeRecurring: true,
          includeSubscription: true,
          includeSupport: true,
          includeTax: true,
          includeUpfront: true,
          useAmortized: false,
          useBlended: false,
        },
      },
      notificationsWithSubscribers: [
        // 50% — email warning only
        {
          notification: {
            notificationType: 'ACTUAL',
            comparisonOperator: 'GREATER_THAN',
            threshold: 50,
            thresholdType: 'PERCENTAGE',
          },
          subscribers: [{ subscriptionType: 'EMAIL', address: alertEmail }],
        },
        // 80% — email + SNS to preventive-scale-down topic
        {
          notification: {
            notificationType: 'ACTUAL',
            comparisonOperator: 'GREATER_THAN',
            threshold: 80,
            thresholdType: 'PERCENTAGE',
          },
          subscribers: [
            { subscriptionType: 'EMAIL', address: alertEmail },
            {
              subscriptionType: 'SNS',
              address: this.preventiveScaleDownTopic.topicArn,
            },
          ],
        },
        // 100% — email + kill-switch
        {
          notification: {
            notificationType: 'ACTUAL',
            comparisonOperator: 'GREATER_THAN',
            threshold: 100,
            thresholdType: 'PERCENTAGE',
          },
          subscribers: [
            { subscriptionType: 'EMAIL', address: alertEmail },
            { subscriptionType: 'SNS', address: this.killSwitchTopic.topicArn },
          ],
        },
        // Forecasted overrun (uses AWS forecast)
        {
          notification: {
            notificationType: 'FORECASTED',
            comparisonOperator: 'GREATER_THAN',
            threshold: 100,
            thresholdType: 'PERCENTAGE',
          },
          subscribers: [{ subscriptionType: 'EMAIL', address: alertEmail }],
        },
      ],
    });

    // ---------- Log group for shared/operational use ----------
    new logs.LogGroup(this, 'OperationsLogGroup', {
      logGroupName: '/comfy/operations',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });
  }
}
