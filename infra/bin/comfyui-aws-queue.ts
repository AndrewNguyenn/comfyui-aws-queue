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
import { MetadataStack } from '../lib/stacks/metadata';
import { WebSocketStack } from '../lib/stacks/websocket';

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

// WebSocket stack is created BEFORE the REST API stack because the REST API
// references the WS API endpoint via SSM (for the /ws-ticket response). The
// SSM parameter is created during ws stack synth so REST stack can resolve it
// at deploy time.
const websocket = new WebSocketStack(app, 'ComfyWebSocketStack', {
  env,
  config: APP_CONFIG,
  storage,
  description: 'WebSocket API for live ComfyUI event streaming',
});
Tags.of(websocket).add('Component', 'websocket');

const api = new ApiStack(app, 'ComfyApiStack', {
  env,
  config: APP_CONFIG,
  storage,
  queue,
  auth,
  websocket,
  description: 'API Gateway + Lambda functions (dispatcher, status, catalog, downloader)',
});
api.addDependency(websocket);
// Note: dispatcher proxies /manager/* and /customnode/* to the metadata
// instance by reading /comfy/metadata/url from SSM at *runtime* (not synth).
// So api no longer depends on metadata's SSM at deploy time — avoids the
// circular dep where metadata also reads api's SSM (DISPATCHER_API_URL).
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

// Always-on metadata instance (CPU-only t3.small). Publishes object_info +
// extensions on boot so the editor always has node defs + Manager UI without
// waiting for a GPU spot worker. Depends on ci (needs ECR repo) and api
// (reads dispatcher URL + worker API key id via SSM).
const metadata = new MetadataStack(app, 'ComfyMetadataStack', {
  env,
  config: APP_CONFIG,
  network,
  storage,
  ci,
  description: 'Always-on t3.small running ComfyUI in CPU mode for editor metadata',
});
Tags.of(metadata).add('Component', 'metadata');
metadata.addDependency(ci);
metadata.addDependency(api);

// Phase 5 — frontend (depends on api+auth for runtime config).
// Only instantiate if frontend/dist/ exists so other stacks can synth/deploy
// before the frontend has been built. The build is a separate step
// (./frontend/build.sh) since it pulls vanilla ComfyUI from upstream.
const fs = require('fs');
const path = require('path');
const distExists = fs.existsSync(path.join(__dirname, '../../frontend/dist'));
if (distExists) {
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
} else {
  console.error(
    '[INFO] Skipping FrontendStack: frontend/dist not found. ' +
    'Run ./frontend/build.sh before deploying that stack.'
  );
}

app.synth();
