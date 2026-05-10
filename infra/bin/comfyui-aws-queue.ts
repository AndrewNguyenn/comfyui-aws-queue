#!/usr/bin/env node
import 'source-map-support/register';
import { App, Tags } from 'aws-cdk-lib';
import { APP_CONFIG } from '../lib/config';
import { NetworkStack } from '../lib/stacks/network';
import { StorageStack } from '../lib/stacks/storage';
import { QueueStack } from '../lib/stacks/queue';
import { MonitoringStack } from '../lib/stacks/monitoring';
import { AuthStack } from '../lib/stacks/auth';
import { ApiStack } from '../lib/stacks/api';
import { CiStack } from '../lib/stacks/ci';
import { ComputeStack } from '../lib/stacks/compute';
import { FrontendStack } from '../lib/stacks/frontend';

const app = new App();

const env = {
  region: APP_CONFIG.region,
  // account intentionally unset — resolved at deploy time from CLI credentials.
  // Keeps this app portable across accounts and safe in a public repo.
};

// Project-wide tags. Used for cost allocation in Billing console
// (manual activation in Billing → Cost Allocation Tags, ~24h to populate).
for (const [k, v] of Object.entries(APP_CONFIG.tags)) {
  Tags.of(app).add(k, v);
}

// Phase 1 — free, deployable first
const network = new NetworkStack(app, 'ComfyNetworkStack', {
  env,
  config: APP_CONFIG,
  description: 'VPC, public subnets, security groups, S3+DDB gateway endpoints',
});
Tags.of(network).add('Component', 'network');

const storage = new StorageStack(app, 'ComfyStorageStack', {
  env,
  config: APP_CONFIG,
  description: 'S3 buckets (models, outputs, uploads, frontend), DDB tables, Secrets Manager',
});
Tags.of(storage).add('Component', 'storage');

const queue = new QueueStack(app, 'ComfyQueueStack', {
  env,
  config: APP_CONFIG,
  description: 'SQS image+video queues with DLQs',
});
Tags.of(queue).add('Component', 'queue');

const monitoring = new MonitoringStack(app, 'ComfyMonitoringStack', {
  env,
  config: APP_CONFIG,
  description: 'AWS Budget, kill-switch SNS topic, alarms',
});
Tags.of(monitoring).add('Component', 'monitoring');
monitoring.addDependency(queue);
monitoring.addDependency(storage);

// Phase 2 — auth + API
const auth = new AuthStack(app, 'ComfyAuthStack', {
  env,
  config: APP_CONFIG,
  description: 'Cognito user pool for API authentication',
});
Tags.of(auth).add('Component', 'auth');

const api = new ApiStack(app, 'ComfyApiStack', {
  env,
  config: APP_CONFIG,
  storage,
  queue,
  auth,
  description: 'API Gateway + Lambda functions (dispatcher, status, catalog, downloader)',
});
Tags.of(api).add('Component', 'api');

// Phase 3 — CI (CodeBuild + ECR)
const ci = new CiStack(app, 'ComfyCiStack', {
  env,
  config: APP_CONFIG,
  description: 'CodeBuild projects + ECR repos for worker images',
});
Tags.of(ci).add('Component', 'ci');

// Phase 4 — compute (depends on CI for ECR repos to exist, and on api for
// SSM parameters /comfy/api/url and /comfy/api/worker-key-id that the worker
// task reads at runtime).
const compute = new ComputeStack(app, 'ComfyComputeStack', {
  env,
  config: APP_CONFIG,
  network,
  storage,
  queue,
  ci,
  description: 'ECS cluster, image+video ASGs, capacity providers, services',
});
Tags.of(compute).add('Component', 'compute');
compute.addDependency(ci);
compute.addDependency(api);

// Phase 5 — frontend (depends on api+auth for runtime config)
const frontend = new FrontendStack(app, 'ComfyFrontendStack', {
  env,
  config: APP_CONFIG,
  storage,
  api,
  auth,
  description: 'S3 static-website hosting with auto-generated runtime config',
});
Tags.of(frontend).add('Component', 'frontend');
frontend.addDependency(api);
frontend.addDependency(auth);

app.synth();
