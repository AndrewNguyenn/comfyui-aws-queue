#!/usr/bin/env node
import 'source-map-support/register';
import { App, Tags, Aspects } from 'aws-cdk-lib';
import { APP_CONFIG } from '../lib/config';
import { NetworkStack } from '../lib/stacks/network';
import { StorageStack } from '../lib/stacks/storage';
import { QueueStack } from '../lib/stacks/queue';
import { MonitoringStack } from '../lib/stacks/monitoring';

const app = new App();

const env = {
  region: APP_CONFIG.region,
  // account is intentionally unset — resolved at deploy time from CLI credentials,
  // never hardcoded so this app stays portable across accounts and safe in a public repo.
};

// Apply project-wide tags. Used for cost allocation in Billing console.
for (const [k, v] of Object.entries(APP_CONFIG.tags)) {
  Tags.of(app).add(k, v);
}

const network = new NetworkStack(app, 'ComfyNetworkStack', {
  env,
  config: APP_CONFIG,
  description: 'VPC, public subnets, security groups, S3 gateway endpoint',
});

const storage = new StorageStack(app, 'ComfyStorageStack', {
  env,
  config: APP_CONFIG,
  description: 'S3 buckets (models, outputs, uploads, frontend), DDB tables, Secrets Manager',
});

const queue = new QueueStack(app, 'ComfyQueueStack', {
  env,
  config: APP_CONFIG,
  description: 'SQS queues for image and video jobs, plus DLQs',
});

const monitoring = new MonitoringStack(app, 'ComfyMonitoringStack', {
  env,
  config: APP_CONFIG,
  description: 'AWS Budget, kill-switch Lambda, alarms, dashboards',
});

// Explicit dependency: monitoring references queue/storage resources for alarms.
monitoring.addDependency(queue);
monitoring.addDependency(storage);

// Future stacks (added in subsequent phases):
// - ComfyAuthStack (Cognito user pool)
// - ComfyApiStack (API Gateway + Lambdas)
// - ComfyCiStack (CodeBuild + ECR)
// - ComfyComputeStack (ECS, ASGs, capacity providers)
// - ComfyFrontendStack (S3 static website)

app.synth();
