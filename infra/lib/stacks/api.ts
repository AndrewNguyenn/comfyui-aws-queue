import { Stack, StackProps, Duration, RemovalPolicy, CfnOutput } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as eventsources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as path from 'path';
import { AppConfig } from '../config';
import { StorageStack } from './storage';
import { QueueStack } from './queue';
import { AuthStack } from './auth';
import { WebSocketStack } from './websocket';

export interface ApiStackProps extends StackProps {
  readonly config: AppConfig;
  readonly storage: StorageStack;
  readonly queue: QueueStack;
  readonly auth: AuthStack;
  readonly websocket: WebSocketStack;
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
 *   POST /jobs/{id}/cancel    → StatusFn      (cancel a queued or running job)
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
  public readonly editorCompatFn: lambda.Function;
  public readonly statusFn: lambda.Function;
  public readonly downloadKickoffFn: lambda.Function;
  public readonly downloadWorkerFn: lambda.Function;
  public readonly catalogFn: lambda.Function;
  public readonly reaperFn: lambda.Function;

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
      // X-Ray disabled: ACTIVE records a trace per invocation, and the
      // editor's constant polling (job status, /object_info, /view) burned
      // the 100k-traces/month free tier for traces nobody reads. Flip back
      // to lambda.Tracing.ACTIVE temporarily if you need to debug latency.
      tracing: lambda.Tracing.DISABLED,
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
        MINIMAX_QUEUE_URL: queue.minimaxJobsQueue.queueUrl,
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
    queue.minimaxJobsQueue.grantSendMessages(this.dispatcherFn);
    storage.jobsTable.grantReadWriteData(this.dispatcherFn);
    storage.modelsTable.grantReadData(this.dispatcherFn);
    storage.objectInfoTable.grantReadWriteData(this.dispatcherFn);
    // dispatcher writes object_info to outputs bucket (under metadata/ prefix)
    // because the JSON exceeds DDB's 400 KB item limit, then reads it back
    // when serving /object_info.
    storage.outputsBucket.grantReadWrite(this.dispatcherFn);
    // /ws-ticket needs to read the HMAC secret to sign tickets.
    storage.wsTicketSecret.grantRead(this.dispatcherFn);
    this.dispatcherFn.addEnvironment('WS_TICKET_SECRET_ARN', storage.wsTicketSecret.secretArn);
    // X-Comfy-Auth secret — sent on every proxied request to the metadata
    // instance so the in-container nginx gate lets the dispatcher through.
    storage.metadataAuthSecret.grantRead(this.dispatcherFn);
    this.dispatcherFn.addEnvironment('WS_API_ENDPOINT', props.websocket.wsStage.url);

    // ----- Lambda: EditorCompat -----
    // Same code asset as dispatcher (handler.py routes by event["resource"]),
    // but a separate Function so the editor-compat routes have their own
    // 20 KB Lambda resource policy budget — combining them on DispatcherFn
    // exceeded the limit.
    this.editorCompatFn = new lambda.Function(this, 'EditorCompatFn', {
      ...commonLambdaProps,
      functionName: 'comfy-editor-compat',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/dispatcher')),
      handler: 'handler.lambda_handler',
      description: 'ComfyUI editor compat endpoints (/users, /settings, /userdata, /i18n, etc.)',
    });
    storage.jobsTable.grantReadWriteData(this.editorCompatFn);
    storage.modelsTable.grantReadData(this.editorCompatFn);
    storage.objectInfoTable.grantReadData(this.editorCompatFn);
    storage.outputsBucket.grantReadWrite(this.editorCompatFn);
    queue.imageJobsQueue.grantSendMessages(this.editorCompatFn); // not used but cheaper than conditional env
    queue.videoJobsQueue.grantSendMessages(this.editorCompatFn);
    // EditorCompat shares the dispatcher code, including _proxy_to_metadata —
    // it also needs the X-Comfy-Auth secret for the nginx gate.
    storage.metadataAuthSecret.grantRead(this.editorCompatFn);

    // ----- Lambda: Status -----
    this.statusFn = new lambda.Function(this, 'StatusFn', {
      ...commonLambdaProps,
      functionName: 'comfy-status',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/status')),
      handler: 'handler.lambda_handler',
      description: 'Job status reads, presigned URL generation for view/upload',
    });
    // Read+write: DELETE /jobs/{id} removes the job record and its output
    // objects (the viewer's per-tile delete button).
    storage.jobsTable.grantReadWriteData(this.statusFn);
    storage.outputsBucket.grantReadWrite(this.statusFn);
    storage.uploadsBucket.grantPut(this.statusFn);
    // GET /backlog reads the live queue depth (ApproximateNumberOfMessages) so
    // the viewer's pending strip shows an exact, uncapped "in queue" count
    // instead of counting fetched job records. Queue URLs are already in env
    // (commonLambdaProps); this adds the read permission.
    queue.imageJobsQueue.grant(this.statusFn, 'sqs:GetQueueAttributes');
    queue.videoJobsQueue.grant(this.statusFn, 'sqs:GetQueueAttributes');
    queue.minimaxJobsQueue.grant(this.statusFn, 'sqs:GetQueueAttributes');

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
    // Long-running. 15 min timeout, streams CivitAI/HuggingFace → S3.
    //
    // 3008 MB is this account's ceiling, not a choice: Lambda rejects anything
    // higher with "'MemorySize' value failed to satisfy constraint: Member must
    // have value less than or equal to 3008". Accounts provisioned before the
    // 10 GB limit keep the old cap until a quota increase is requested. If
    // downloads ever outgrow this again, that request is the lever.
    //
    // Memory is not for buffering — the transfer is streamed through a
    // multipart upload and ephemeral storage stays at 512 MB — but because
    // Lambda allocates network bandwidth in proportion to memory. At 1 GB the
    // download worker sustained ~1.2 GiB/min, which is fine for a 200 MB LoRA
    // and hopeless for the model sizes video work needs: MiniMax H3's 19.5 GiB
    // transformer projected to ~23 min against a hard 15 min ceiling, and its
    // 25.3 GiB text encoder only just landed in 10.5 min. The ceiling cannot be
    // raised (15 min is Lambda's maximum), so the only lever is throughput.
    this.downloadWorkerFn = new lambda.Function(this, 'DownloadWorkerFn', {
      ...commonLambdaProps,
      functionName: 'comfy-download-worker',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/downloader')),
      handler: 'worker.lambda_handler',
      description: 'Streams model download from CivitAI to S3 multipart upload',
      timeout: Duration.minutes(15),
      memorySize: 3008,
      ephemeralStorageSize: require('aws-cdk-lib').Size.mebibytes(512), // No big disk needed; streaming
      // The kickoff Lambda invokes this asynchronously, so Lambda's default of
      // 2 automatic retries turns one timed-out multi-GB download into three
      // concurrent invocations fetching the SAME file. They then starve each
      // other of the very bandwidth that made the first one time out, and each
      // restarts from zero — observed as progress percentages going backwards.
      // A failed download should be re-requested deliberately, not multiplied.
      retryAttempts: 0,
    });
    storage.modelsBucket.grantWrite(this.downloadWorkerFn);
    storage.modelsTable.grantWriteData(this.downloadWorkerFn);
    storage.downloadsTable.grantReadWriteData(this.downloadWorkerFn);
    storage.civitaiTokenSecret.grantRead(this.downloadWorkerFn);

    // Wire the kickoff → worker async invoke
    this.downloadWorkerFn.grantInvoke(this.downloadKickoffFn);
    this.downloadKickoffFn.addEnvironment('DOWNLOAD_WORKER_FN', this.downloadWorkerFn.functionName);

    // Manager's Model Manager UI → /manager/queue/install_model is handled by
    // the dispatcher Lambda, which translates the request into a synchronous
    // RequestResponse invoke of the download-kickoff Lambda (same code path
    // as POST /models/download). Grant + wire.
    this.downloadKickoffFn.grantInvoke(this.dispatcherFn);
    this.dispatcherFn.addEnvironment('DOWNLOAD_KICKOFF_FN', this.downloadKickoffFn.functionName);

    // ----- Lambda: Reaper -----
    // SQS-tick-driven sweeper for zombie jobs (status stuck "running" after a
    // worker died). Not on a fixed cron — it runs only while there is work
    // (see ReaperTicksQueue / the dispatcher's arm-on-submit). Re-queues a
    // zombie up to 3× then parks its workflow JSON in the failed-workflows
    // bucket for analysis.
    this.reaperFn = new lambda.Function(this, 'ReaperFn', {
      ...commonLambdaProps,
      functionName: 'comfy-reaper',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/reaper')),
      handler: 'handler.lambda_handler',
      description: 'Sweeps zombie jobs: re-queues up to 3x, then archives to S3',
      timeout: Duration.seconds(60),
      // The reaper self-perpetuates by sending a delayed tick to the same SQS
      // queue that triggers it — an intentional, rate-limited loop (one tick
      // per 180 s, stops at idle). Lambda's recursive-loop detector flags any
      // Lambda↔own-SQS pattern and would otherwise terminate the chain, so
      // opt that one function out.
      recursiveLoop: lambda.RecursiveLoop.ALLOW,
      environment: {
        ...commonLambdaProps.environment,
        REAPER_QUEUE_URL: queue.reaperTicksQueue.queueUrl,
        FAILED_WORKFLOWS_BUCKET: storage.failedWorkflowsBucket.bucketName,
      },
    });
    storage.jobsTable.grantReadWriteData(this.reaperFn);          // query GSI + flip status
    queue.imageJobsQueue.grantSendMessages(this.reaperFn);        // re-queue image jobs
    queue.videoJobsQueue.grantSendMessages(this.reaperFn);        // re-queue video jobs
    queue.minimaxJobsQueue.grantSendMessages(this.reaperFn);      // re-queue minimax jobs
    queue.reaperTicksQueue.grantSendMessages(this.reaperFn);      // self-arm next tick
    storage.failedWorkflowsBucket.grantPut(this.reaperFn);        // archive give-ups
    this.reaperFn.addEventSource(
      new eventsources.SqsEventSource(queue.reaperTicksQueue, { batchSize: 10 }),
    );
    // Dispatcher arms the reaper when a job is submitted.
    queue.reaperTicksQueue.grantSendMessages(this.dispatcherFn);
    this.dispatcherFn.addEnvironment('REAPER_QUEUE_URL', queue.reaperTicksQueue.queueUrl);

    // ----- Lambda: WebSocket Forward (REST endpoint /internal/ws-event) -----
    // Workers POST events here; Lambda pushes them to the right WS connection.
    const wsForwardFn = new lambda.Function(this, 'WsForwardFn', {
      ...commonLambdaProps,
      functionName: 'comfy-ws-forward',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/websocket')),
      handler: 'forward.lambda_handler',
      description: 'Pushes ComfyUI events from worker to browser via WS API',
      environment: {
        REGION: this.region,
        CONNECTIONS_TABLE: props.websocket.connectionsTable.tableName,
        WS_API_ENDPOINT: `https://${props.websocket.wsApi.apiId}.execute-api.${this.region}.amazonaws.com/${props.websocket.wsStage.stageName}`,
      },
    });
    props.websocket.connectionsTable.grantReadWriteData(wsForwardFn);
    // Grant manage-connections on the WS API. CDK doesn't expose a helper for
    // this; we add the IAM policy explicitly.
    wsForwardFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['execute-api:ManageConnections'],
      resources: [
        `arn:aws:execute-api:${this.region}:${this.account}:${props.websocket.wsApi.apiId}/*`,
      ],
    }));

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
    const dispatcherIntegration = new apigw.LambdaIntegration(this.dispatcherFn, { allowTestInvoke: false });
    const statusIntegration = new apigw.LambdaIntegration(this.statusFn, { allowTestInvoke: false });
    const catalogIntegration = new apigw.LambdaIntegration(this.catalogFn, { allowTestInvoke: false });
    const downloadKickoffIntegration = new apigw.LambdaIntegration(this.downloadKickoffFn, { allowTestInvoke: false });

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

    // /jobs (list — editor queue/history) + /jobs/{id} (single)
    const jobsRoot = root.addResource('jobs');
    jobsRoot.addMethod('GET', statusIntegration, authMethodOptions);
    // /jobs/cancel-group — cancel an entire character stack (POST body: group
    // identity {type,model,character,subject}). Registered before /jobs/{id} so
    // the literal path wins over the {id} greedy param.
    jobsRoot.addResource('cancel-group').addMethod('POST', statusIntegration, authMethodOptions);
    const jobs = jobsRoot.addResource('{id}');
    jobs.addMethod('GET', statusIntegration, authMethodOptions);
    jobs.addMethod('DELETE', statusIntegration, authMethodOptions);
    // /jobs/{id}/cancel — cancel a queued or running job
    jobs.addResource('cancel').addMethod('POST', statusIntegration, authMethodOptions);

    // /backlog — live SQS backlog depth for the pending strip's count.
    // (/queue is taken by the ComfyUI editor-compat stub below.)
    const backlog = root.addResource('backlog');
    backlog.addMethod('GET', statusIntegration, authMethodOptions);

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

    // ----- ComfyUI-Manager passthrough (via dispatcher Lambda) -----
    // Manager's frontend calls /api/manager/* and /api/customnode/* paths.
    // Those need to hit a live ComfyUI instance with Manager loaded.
    // Dispatcher Lambda reads /comfy/metadata/url from SSM at *runtime* and
    // forwards the request to the metadata instance. This avoids a circular
    // synth-time dependency between ApiStack and MetadataStack (each reads
    // SSM parameters created by the other).
    this.dispatcherFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['ssm:GetParameter'],
      resources: [`arn:aws:ssm:${this.region}:${this.account}:parameter/comfy/metadata/*`],
    }));
    // The editor's Manager "Restart" button → dispatcher → SSM RunShellScript
    // (`docker restart comfy-metadata`) on the metadata instance. Scoped to
    // that instance via a Name-tag condition so the dispatcher can't run
    // shell on anything else.
    this.dispatcherFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['ssm:SendCommand'],
      resources: [`arn:aws:ssm:${this.region}::document/AWS-RunShellScript`],
    }));
    this.dispatcherFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['ssm:SendCommand'],
      resources: [`arn:aws:ec2:${this.region}:${this.account}:instance/*`],
      conditions: {
        StringEquals: { 'ssm:resourceTag/Name': 'ComfyMetadataStack/MetadataInstance' },
      },
    }));

    const proxyMethodOptions: apigw.MethodOptions = {
      ...authMethodOptions,
      requestParameters: { 'method.request.path.proxy': true },
    };

    // 'civitai' = the Civicomfy custom node's routes (search/browse/download/
    // status). Dispatcher proxies them to the metadata instance; POST
    // /civitai/download is intercepted there and redirected to /models/download.
    // 'externalmodel' = ComfyUI-Manager's Model Manager list (/externalmodel/getlist).
    for (const base of ['manager', 'customnode', 'snapshot', 'model-manager', 'civitai', 'externalmodel']) {
      const baseRes = root.addResource(base);
      const proxyRes = baseRes.addResource('{proxy+}');
      for (const method of ['GET', 'POST']) {
        proxyRes.addMethod(method, dispatcherIntegration, proxyMethodOptions);
      }
    }

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
      new apigw.LambdaIntegration(this.dispatcherFn, { allowTestInvoke: false }),
      { apiKeyRequired: true } // No Cognito; API key only
    );
    // Worker → server WebSocket event push (also API-key authed).
    const internalWsEvent = internal.addResource('ws-event');
    internalWsEvent.addMethod(
      'POST',
      new apigw.LambdaIntegration(wsForwardFn, { allowTestInvoke: false }),
      { apiKeyRequired: true }
    );
    // Worker → server custom-node extension publish (api-key authed).
    const internalExtensions = internal.addResource('extensions');
    internalExtensions.addMethod(
      'POST',
      new apigw.LambdaIntegration(this.dispatcherFn, { allowTestInvoke: false }),
      { apiKeyRequired: true }
    );
    // Editor LOGS panel → ComfyUI's /internal/logs/raw. Unlike the sibling
    // /internal routes above (worker-only, API-key auth), this is called by
    // the browser editor, so it's Cognito-authed. The dispatcher proxies it
    // to the metadata instance (the only full-time ComfyUI), whose log buffer
    // also shows custom-node install activity.
    const internalLogs = internal.addResource('logs');
    const internalLogsProxy = internalLogs.addResource('{proxy+}');
    // GET /internal/logs/raw (initial fetch) + PATCH /internal/logs/subscribe
    // (the editor's live-log toggle).
    for (const method of ['GET', 'POST', 'PATCH']) {
      internalLogsProxy.addMethod(method, dispatcherIntegration, proxyMethodOptions);
    }

    // ----- /ws-ticket (Cognito-authed) -----
    // Browser fetches this before opening a WebSocket; response includes the
    // WS URL and a 60-sec HMAC ticket the WS connect Lambda validates.
    const wsTicket = root.addResource('ws-ticket');
    wsTicket.addMethod('GET', dispatcherIntegration, authMethodOptions);

    // Stub endpoints that ComfyUI's frontend expects (resolves N16 from review).
    // Return minimal responses so the UI doesn't 404. dispatcher handler
    // detects these paths and returns the appropriate stub.
    for (const stubPath of ['queue', 'system_stats', 'embeddings', 'extensions']) {
      const r = root.addResource(stubPath);
      r.addMethod('GET', dispatcherIntegration, authMethodOptions);
    }

    // ----- ComfyUI editor (Comfy-Org/ComfyUI_frontend v1.x) compat routes -----
    // Routed to the separate EditorCompatFn Lambda to avoid blowing the 20 KB
    // Lambda resource policy limit on DispatcherFn.
    const editorIntegration = new apigw.LambdaIntegration(this.editorCompatFn, { allowTestInvoke: false });

    // GET /history (list mode, no id) — editor polls for queue history
    const historyList = this.api.root.getResource('history')!;
    historyList.addMethod('GET', editorIntegration, authMethodOptions);

    // /users
    const users = root.addResource('users');
    users.addMethod('GET', editorIntegration, authMethodOptions);
    users.addMethod('POST', editorIntegration, authMethodOptions);

    // /i18n + /i18n/{locale}
    const i18n = root.addResource('i18n');
    i18n.addMethod('GET', editorIntegration, authMethodOptions);
    i18n.addResource('{locale}').addMethod('GET', editorIntegration, authMethodOptions);

    // /free
    const freeRes = root.addResource('free');
    freeRes.addMethod('POST', editorIntegration, authMethodOptions);

    // /workflow_templates
    root.addResource('workflow_templates').addMethod('GET', editorIntegration, authMethodOptions);

    // /global_subgraphs
    root.addResource('global_subgraphs').addMethod('GET', editorIntegration, authMethodOptions);

    // /experiment/models + /experiment/models/{folder}
    const experiment = root.addResource('experiment');
    const experimentModels = experiment.addResource('models');
    experimentModels.addMethod('GET', editorIntegration, authMethodOptions);
    experimentModels.addResource('{folder}').addMethod('GET', editorIntegration, authMethodOptions);

    // /settings, /settings/{id}
    const settings = root.addResource('settings');
    settings.addMethod('GET', editorIntegration, authMethodOptions);
    settings.addMethod('POST', editorIntegration, authMethodOptions);
    const settingById = settings.addResource('{id}');
    settingById.addMethod('GET', editorIntegration, authMethodOptions);
    settingById.addMethod('POST', editorIntegration, authMethodOptions);
    settingById.addMethod('DELETE', editorIntegration, authMethodOptions);

    // /userdata, /userdata/{file+} (proxy for nested paths)
    const userdata = root.addResource('userdata');
    userdata.addMethod('GET', editorIntegration, authMethodOptions);
    const userdataFile = userdata.addResource('{file+}');
    userdataFile.addMethod('GET', editorIntegration, authMethodOptions);
    userdataFile.addMethod('POST', editorIntegration, authMethodOptions);
    userdataFile.addMethod('DELETE', editorIntegration, authMethodOptions);

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
