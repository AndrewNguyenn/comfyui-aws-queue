import { Stack, StackProps, Duration, RemovalPolicy, CfnOutput } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as path from 'path';
import { AppConfig } from '../config';
import { StorageStack } from './storage';
import { QueueStack } from './queue';
import { AuthStack } from './auth';

export interface ApiStackProps extends StackProps {
  readonly config: AppConfig;
  readonly storage: StorageStack;
  readonly queue: QueueStack;
  readonly auth: AuthStack;
}

/**
 * REST API + Lambda functions.
 *
 * Routes:
 *   POST /prompt              → DispatcherFn  (route to image|video SQS)
 *   GET  /history/{id}        → DispatcherFn
 *   GET  /object_info         → DispatcherFn
 *   POST /internal/object_info→ DispatcherFn  (worker pushes its object_info on boot)
 *
 *   GET  /jobs/{id}           → StatusFn
 *   GET  /view                → StatusFn      (presigned-redirect to S3 output)
 *   POST /upload/image        → StatusFn      (presigned PUT for direct upload)
 *
 *   POST /models/download     → DownloadKickoffFn (returns download_id, async-invokes worker)
 *   GET  /downloads/{id}      → DownloadKickoffFn (status poll)
 *
 *   GET  /models              → CatalogFn
 *   POST /models              → CatalogFn (manual add)
 *   DELETE /models/{name}     → CatalogFn
 *
 * All routes require Cognito authentication (ID token in Authorization header).
 * /internal/* requires the API key in addition (worker-only).
 *
 * Throttling: 5 req/sec per source IP, burst 50. CloudWatch alarm at >100 calls/5min
 * triggers the kill-switch.
 */
export class ApiStack extends Stack {
  public readonly api: apigw.RestApi;
  public readonly dispatcherFn: lambda.Function;
  public readonly statusFn: lambda.Function;
  public readonly downloadKickoffFn: lambda.Function;
  public readonly downloadWorkerFn: lambda.Function;
  public readonly catalogFn: lambda.Function;

  constructor(scope: Construct, id: string, props: ApiStackProps) {
    super(scope, id, props);
    const { config, storage, queue, auth } = props;

    // ----- Common Lambda settings -----
    const commonLambdaProps = {
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64, // ~20% cheaper than x86 Lambda
      timeout: Duration.seconds(29), // Just under API GW 30s timeout
      memorySize: 512,
      logRetention: logs.RetentionDays.ONE_WEEK,
      tracing: lambda.Tracing.ACTIVE,
      environment: {
        REGION: this.region,
        MODELS_BUCKET: storage.modelsBucket.bucketName,
        OUTPUTS_BUCKET: storage.outputsBucket.bucketName,
        UPLOADS_BUCKET: storage.uploadsBucket.bucketName,
        MODELS_TABLE: storage.modelsTable.tableName,
        JOBS_TABLE: storage.jobsTable.tableName,
        DOWNLOADS_TABLE: storage.downloadsTable.tableName,
        OBJECT_INFO_TABLE: storage.objectInfoTable.tableName,
        IMAGE_QUEUE_URL: queue.imageJobsQueue.queueUrl,
        VIDEO_QUEUE_URL: queue.videoJobsQueue.queueUrl,
      },
    };

    // ----- Lambda: Dispatcher -----
    this.dispatcherFn = new lambda.Function(this, 'DispatcherFn', {
      ...commonLambdaProps,
      functionName: 'comfy-dispatcher',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/dispatcher')),
      handler: 'handler.lambda_handler',
      description: 'Routes ComfyUI workflows to image or video SQS queue',
    });
    queue.imageJobsQueue.grantSendMessages(this.dispatcherFn);
    queue.videoJobsQueue.grantSendMessages(this.dispatcherFn);
    storage.jobsTable.grantReadWriteData(this.dispatcherFn);
    storage.modelsTable.grantReadData(this.dispatcherFn);
    storage.objectInfoTable.grantReadWriteData(this.dispatcherFn);

    // ----- Lambda: Status -----
    this.statusFn = new lambda.Function(this, 'StatusFn', {
      ...commonLambdaProps,
      functionName: 'comfy-status',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/status')),
      handler: 'handler.lambda_handler',
      description: 'Job status reads, presigned URL generation for view/upload',
    });
    storage.jobsTable.grantReadData(this.statusFn);
    storage.outputsBucket.grantRead(this.statusFn);
    storage.uploadsBucket.grantPut(this.statusFn);

    // ----- Lambda: Catalog -----
    this.catalogFn = new lambda.Function(this, 'CatalogFn', {
      ...commonLambdaProps,
      functionName: 'comfy-catalog',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/catalog')),
      handler: 'handler.lambda_handler',
      description: 'Model catalog CRUD against DynamoDB',
    });
    storage.modelsTable.grantReadWriteData(this.catalogFn);

    // ----- Lambda: Download Kickoff -----
    // Fast Lambda that accepts the API GW request, writes a queued record, then
    // async-invokes the actual download worker. Returns in <1s to avoid API GW 30s timeout.
    this.downloadKickoffFn = new lambda.Function(this, 'DownloadKickoffFn', {
      ...commonLambdaProps,
      functionName: 'comfy-download-kickoff',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/downloader')),
      handler: 'kickoff.lambda_handler',
      description: 'Accepts download request, writes record, async-invokes worker',
      environment: {
        ...commonLambdaProps.environment,
        // Will be set after downloadWorkerFn is created (forward reference resolved below)
        DOWNLOAD_WORKER_FN: '',
      },
    });
    storage.downloadsTable.grantReadWriteData(this.downloadKickoffFn);

    // ----- Lambda: Download Worker -----
    // Long-running. 15 min timeout, 1 GB memory, streams CivitAI → S3.
    this.downloadWorkerFn = new lambda.Function(this, 'DownloadWorkerFn', {
      ...commonLambdaProps,
      functionName: 'comfy-download-worker',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/downloader')),
      handler: 'worker.lambda_handler',
      description: 'Streams model download from CivitAI to S3 multipart upload',
      timeout: Duration.minutes(15),
      memorySize: 1024,
      ephemeralStorageSize: require('aws-cdk-lib').Size.mebibytes(512), // No big disk needed; streaming
    });
    storage.modelsBucket.grantWrite(this.downloadWorkerFn);
    storage.modelsTable.grantWriteData(this.downloadWorkerFn);
    storage.downloadsTable.grantReadWriteData(this.downloadWorkerFn);
    storage.civitaiTokenSecret.grantRead(this.downloadWorkerFn);

    // Wire the kickoff → worker async invoke
    this.downloadWorkerFn.grantInvoke(this.downloadKickoffFn);
    this.downloadKickoffFn.addEnvironment('DOWNLOAD_WORKER_FN', this.downloadWorkerFn.functionName);

    // ----- API Gateway -----
    const apiLogs = new logs.LogGroup(this, 'ApiAccessLogs', {
      logGroupName: '/comfy/api/access',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    this.api = new apigw.RestApi(this, 'Api', {
      restApiName: 'comfy-api',
      description: 'comfyui-aws-queue REST API',
      endpointTypes: [apigw.EndpointType.REGIONAL],
      // Required for the access logs below to work. CDK creates an account-level
      // CloudWatch role and points API GW at it via UpdateAccount. One-time per
      // account; subsequent deploys are idempotent.
      cloudWatchRole: true,
      deployOptions: {
        stageName: 'v1',
        throttlingRateLimit: config.api.throttleRatePerSec,
        throttlingBurstLimit: config.api.throttleBurst,
        accessLogDestination: new apigw.LogGroupLogDestination(apiLogs),
        accessLogFormat: apigw.AccessLogFormat.jsonWithStandardFields({
          ip: true,
          caller: true,
          user: true,
          requestTime: true,
          httpMethod: true,
          resourcePath: true,
          status: true,
          protocol: true,
          responseLength: true,
        }),
        metricsEnabled: true,
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigw.Cors.ALL_ORIGINS, // Restricted later by Cognito
        allowMethods: apigw.Cors.ALL_METHODS,
        allowHeaders: ['Content-Type', 'Authorization', 'X-Api-Key', 'Comfy-User'],
      },
    });

    // Gateway responses: ensure CORS headers land on 4xx/5xx too.
    // Without these, a 403 from "Missing Authentication Token" (returned for
    // unmatched routes) lacks Access-Control-Allow-Origin and the browser
    // throws a CORS error before the JSON body is even readable.
    const corsResponseHeaders = {
      'Access-Control-Allow-Origin': "'*'",
      'Access-Control-Allow-Headers': "'Content-Type,Authorization,X-Api-Key,Comfy-User'",
      'Access-Control-Allow-Methods': "'GET,POST,PUT,DELETE,OPTIONS'",
    };
    this.api.addGatewayResponse('Default4xx', {
      type: apigw.ResponseType.DEFAULT_4XX,
      responseHeaders: corsResponseHeaders,
    });
    this.api.addGatewayResponse('Default5xx', {
      type: apigw.ResponseType.DEFAULT_5XX,
      responseHeaders: corsResponseHeaders,
    });

    // ----- Cognito authorizer -----
    const cognitoAuthorizer = new apigw.CognitoUserPoolsAuthorizer(this, 'CognitoAuthorizer', {
      cognitoUserPools: [auth.userPool],
      identitySource: 'method.request.header.Authorization',
    });

    const authMethodOptions: apigw.MethodOptions = {
      authorizer: cognitoAuthorizer,
      authorizationType: apigw.AuthorizationType.COGNITO,
    };

    // ----- Routes -----
    const dispatcherIntegration = new apigw.LambdaIntegration(this.dispatcherFn);
    const statusIntegration = new apigw.LambdaIntegration(this.statusFn);
    const catalogIntegration = new apigw.LambdaIntegration(this.catalogFn);
    const downloadKickoffIntegration = new apigw.LambdaIntegration(this.downloadKickoffFn);

    const root = this.api.root;

    // /prompt
    const prompt = root.addResource('prompt');
    prompt.addMethod('POST', dispatcherIntegration, authMethodOptions);

    // /history/{id}
    const history = root.addResource('history').addResource('{id}');
    history.addMethod('GET', dispatcherIntegration, authMethodOptions);

    // /object_info
    const objectInfo = root.addResource('object_info');
    objectInfo.addMethod('GET', dispatcherIntegration, authMethodOptions);

    // /jobs/{id}
    const jobs = root.addResource('jobs').addResource('{id}');
    jobs.addMethod('GET', statusIntegration, authMethodOptions);

    // /view
    const view = root.addResource('view');
    view.addMethod('GET', statusIntegration, authMethodOptions);

    // /upload/image
    const upload = root.addResource('upload').addResource('image');
    upload.addMethod('POST', statusIntegration, authMethodOptions);

    // /models (collection)
    const models = root.addResource('models');
    models.addMethod('GET', catalogIntegration, authMethodOptions);
    models.addMethod('POST', catalogIntegration, authMethodOptions);

    // /models/download
    const modelDownload = models.addResource('download');
    modelDownload.addMethod('POST', downloadKickoffIntegration, authMethodOptions);

    // /models/{name}
    const modelByName = models.addResource('{name}');
    modelByName.addMethod('DELETE', catalogIntegration, authMethodOptions);

    // /downloads/{id}
    const downloads = root.addResource('downloads').addResource('{id}');
    downloads.addMethod('GET', downloadKickoffIntegration, authMethodOptions);

    // ----- Worker-only internal routes (API key auth, not Cognito) -----
    // Used by workers to push their /object_info to the dispatcher on boot.
    // Workers can't easily do Cognito auth (no human user) so we use an API key.
    const workerApiKey = this.api.addApiKey('WorkerApiKey', {
      apiKeyName: 'comfy-worker-key',
      description: 'API key used by workers to publish /object_info',
    });
    const workerUsagePlan = this.api.addUsagePlan('WorkerUsagePlan', {
      name: 'comfy-worker-plan',
      throttle: { rateLimit: 5, burstLimit: 10 },
    });
    workerUsagePlan.addApiKey(workerApiKey);
    workerUsagePlan.addApiStage({ stage: this.api.deploymentStage });

    const internal = root.addResource('internal');
    const internalObjectInfo = internal.addResource('object_info');
    internalObjectInfo.addMethod(
      'POST',
      new apigw.LambdaIntegration(this.dispatcherFn),
      { apiKeyRequired: true } // No Cognito; API key only
    );

    // Stub endpoints that ComfyUI's frontend expects (resolves N16 from review).
    // Return minimal responses so the UI doesn't 404. dispatcher handler
    // detects these paths and returns the appropriate stub.
    for (const stubPath of ['queue', 'system_stats', 'embeddings', 'extensions']) {
      const r = root.addResource(stubPath);
      r.addMethod('GET', dispatcherIntegration, authMethodOptions);
    }

    // ----- ComfyUI editor (Comfy-Org/ComfyUI_frontend v1.x) compat routes -----
    // The new editor calls these on startup. Without them the editor blocks
    // on init or shows network errors. dispatcher handler implements them.

    // GET /history (list mode, no id) — the editor polls this for queue history
    const historyList = this.api.root.getResource('history')!;
    historyList.addMethod('GET', dispatcherIntegration, authMethodOptions);

    // /users
    const users = root.addResource('users');
    users.addMethod('GET', dispatcherIntegration, authMethodOptions);
    users.addMethod('POST', dispatcherIntegration, authMethodOptions);

    // /i18n + /i18n/{locale}
    const i18n = root.addResource('i18n');
    i18n.addMethod('GET', dispatcherIntegration, authMethodOptions);
    i18n.addResource('{locale}').addMethod('GET', dispatcherIntegration, authMethodOptions);

    // /free
    const freeRes = root.addResource('free');
    freeRes.addMethod('POST', dispatcherIntegration, authMethodOptions);

    // /workflow_templates
    root.addResource('workflow_templates').addMethod('GET', dispatcherIntegration, authMethodOptions);

    // /global_subgraphs
    root.addResource('global_subgraphs').addMethod('GET', dispatcherIntegration, authMethodOptions);

    // /experiment/models + /experiment/models/{folder}
    const experiment = root.addResource('experiment');
    const experimentModels = experiment.addResource('models');
    experimentModels.addMethod('GET', dispatcherIntegration, authMethodOptions);
    experimentModels.addResource('{folder}').addMethod('GET', dispatcherIntegration, authMethodOptions);

    // /settings, /settings/{id}
    const settings = root.addResource('settings');
    settings.addMethod('GET', dispatcherIntegration, authMethodOptions);
    settings.addMethod('POST', dispatcherIntegration, authMethodOptions);
    const settingById = settings.addResource('{id}');
    settingById.addMethod('GET', dispatcherIntegration, authMethodOptions);
    settingById.addMethod('POST', dispatcherIntegration, authMethodOptions);
    settingById.addMethod('DELETE', dispatcherIntegration, authMethodOptions);

    // /userdata, /userdata/{file+} (proxy for nested paths)
    const userdata = root.addResource('userdata');
    userdata.addMethod('GET', dispatcherIntegration, authMethodOptions);
    const userdataFile = userdata.addResource('{file+}');
    userdataFile.addMethod('GET', dispatcherIntegration, authMethodOptions);
    userdataFile.addMethod('POST', dispatcherIntegration, authMethodOptions);
    userdataFile.addMethod('DELETE', dispatcherIntegration, authMethodOptions);

    // Grant dispatcher Lambda S3 access to the outputs bucket for userdata storage.
    storage.outputsBucket.grantReadWrite(this.dispatcherFn);
    this.dispatcherFn.addEnvironment('OUTPUTS_BUCKET', storage.outputsBucket.bucketName);

    // Publish key API outputs to SSM so other stacks can read them without
    // creating direct CFN dependencies (which would cause cyclical imports
    // between Api ↔ Compute ↔ Frontend).
    new ssm.StringParameter(this, 'ApiUrlParam', {
      parameterName: '/comfy/api/url',
      stringValue: this.api.url,
      description: 'API Gateway base URL for comfy dispatcher',
    });

    // Worker API key value goes to SecretsManager (CDK can't easily put
    // an API key value into SSM as a String parameter without exposing it
    // in CFN templates). Workers fetch it at boot.
    // The key value is a runtime-generated token, not a secret in CFN.
    new CfnOutput(this, 'WorkerApiKeyId', {
      value: workerApiKey.keyId,
      description: 'API key ID for workers (look up value with: aws apigateway get-api-key --api-key <id> --include-value)',
    });
    new CfnOutput(this, 'ApiUrl', {
      value: this.api.url,
      description: 'API Gateway base URL',
    });
    new ssm.StringParameter(this, 'WorkerApiKeyIdParam', {
      parameterName: '/comfy/api/worker-key-id',
      stringValue: workerApiKey.keyId,
      description: 'API key ID — workers fetch the value at boot',
    });
  }
}
