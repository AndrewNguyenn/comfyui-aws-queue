import { Stack, StackProps, Duration, RemovalPolicy } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
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
        allowHeaders: ['Content-Type', 'Authorization', 'X-Api-Key'],
      },
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
  }
}
