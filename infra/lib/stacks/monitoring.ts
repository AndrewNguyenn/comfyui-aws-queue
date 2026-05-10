import { Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as budgets from 'aws-cdk-lib/aws-budgets';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as snsSubs from 'aws-cdk-lib/aws-sns-subscriptions';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as cw from 'aws-cdk-lib/aws-cloudwatch';
import * as cwActions from 'aws-cdk-lib/aws-cloudwatch-actions';
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
 * The kill-switch Lambda itself lives in a later stack that has handles to ASGs
 * and Cognito. This stack creates the SNS topic; the Lambda subscribes to it.
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

    // ---------- AWS Budget ----------
    new budgets.CfnBudget(this, 'MonthlyBudget', {
      budget: {
        budgetName: `${config.projectName}-monthly`,
        budgetType: 'COST',
        timeUnit: 'MONTHLY',
        budgetLimit: {
          amount: config.cost.budgetAmountUsd,
          unit: 'USD',
        },
        costFilters: {
          // Filter to project-tagged resources only. Requires cost-allocation
          // tags activated in Billing console (manual, ~24h to populate).
          TagKeyValue: [`user:Project$${config.projectName}`],
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

    // The kill-switch Lambda is defined in a later stack (ApiStack) where it can
    // reference ASGs from ComputeStack and the Cognito user pool from AuthStack.
    // It will subscribe to this stack's killSwitchTopic.
  }
}
