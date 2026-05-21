import { Stack, StackProps, CfnOutput, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as fs from 'fs';
import * as path from 'path';
import { AppConfig } from '../config';
import { StorageStack } from './storage';
import { ApiStack } from './api';
import { AuthStack } from './auth';

export interface FrontendStackProps extends StackProps {
  readonly config: AppConfig;
  readonly storage: StorageStack;
  readonly api: ApiStack;
  readonly auth: AuthStack;
}

/**
 * Static-website hosting for the frontend.
 *
 * Deploys the contents of `frontend/` (built artifact) to the public-read S3
 * bucket created in StorageStack. Also writes a runtime config.js with API URL,
 * Cognito user pool ID, and app client ID — populated at deploy time, never
 * committed to git.
 */
export class FrontendStack extends Stack {
  constructor(scope: Construct, id: string, props: FrontendStackProps) {
    super(scope, id, props);
    const { config, storage, api, auth } = props;

    // Generate runtime config.js content. Note these values are PUBLIC by design:
    // - API URL: world-discoverable anyway
    // - Cognito pool/client IDs: not secrets (safe to be in browser)
    const configJsContent = `
// AUTO-GENERATED at deploy time. Do not edit.
window.COMFY_CONFIG = {
  apiUrl: ${JSON.stringify(api.api.url)},
  region: ${JSON.stringify(this.region)},
  cognitoUserPoolId: ${JSON.stringify(auth.userPool.userPoolId)},
  cognitoUserPoolClientId: ${JSON.stringify(auth.userPoolClient.userPoolClientId)},
  projectName: ${JSON.stringify(config.projectName)},
};
`;

    // Use s3deploy to upload the frontend build artifact.
    // The build script (`frontend/build.sh`) outputs to `frontend/dist/`.
    // Fail fast at synth time with a clear message if dist is missing
    // (resolves code review N15).
    const distPath = path.join(__dirname, '../../../frontend/dist');
    if (!fs.existsSync(distPath)) {
      throw new Error(
        `FrontendStack: ${distPath} does not exist.\n` +
        `Run ./frontend/build.sh before deploying this stack.`
      );
    }

    new s3deploy.BucketDeployment(this, 'FrontendDeployment', {
      destinationBucket: storage.frontendBucket,
      sources: [
        s3deploy.Source.asset(distPath),
        s3deploy.Source.data('config.js', configJsContent),
      ],
      retainOnDelete: false,
      memoryLimit: 512,
      // no-cache — our standalone pages (viewer.js/.html, models.js, …) keep
      // fixed filenames, so without a revalidation directive a browser serves
      // a stale copy after a deploy (a new viewer.js never reaches the user
      // until they hard-refresh). `no-cache` makes the browser revalidate via
      // ETag every load: a 304 when unchanged (cheap), fresh bytes when not.
      // This also covers the editor's content-hashed assets/ bundles, which
      // could safely be cached `immutable` — splitting that into a second
      // BucketDeployment is a deferred optimization; the revalidation cost is
      // just cheap 304s, acceptable for a single-user tool.
      cacheControl: [s3deploy.CacheControl.fromString('no-cache')],
      // prune:false — the metadata/worker `extensions_publisher` uploads
      // custom-node JS to extensions/<pack>/ at runtime. With the default
      // prune:true, every frontend (re)deploy wipes those (only
      // extensions/core/ ships in the build) → the editor 404s
      // comfyui-manager.js etc. and the Manager UI disappears. Don't prune.
      prune: false,
    });

    new CfnOutput(this, 'FrontendUrl', {
      value: `http://${storage.frontendBucket.bucketWebsiteDomainName}`,
      description: 'Frontend website URL',
    });
    new CfnOutput(this, 'ApiUrl', {
      value: api.api.url,
      description: 'API Gateway base URL',
    });
    new CfnOutput(this, 'CognitoUserPoolId', {
      value: auth.userPool.userPoolId,
      description: 'Cognito User Pool ID',
    });
    new CfnOutput(this, 'CognitoUserPoolClientId', {
      value: auth.userPoolClient.userPoolClientId,
      description: 'Cognito App Client ID',
    });
  }
}
