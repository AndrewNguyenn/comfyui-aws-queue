import { Stack, StackProps, Duration, RemovalPolicy, Aws } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as apigwv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as integrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as path from 'path';
import { AppConfig } from '../config';
import { StorageStack } from './storage';

export interface WebSocketStackProps extends StackProps {
  readonly config: AppConfig;
  readonly storage: StorageStack;
}

/**
 * API Gateway WebSocket API for live ComfyUI event streaming.
 *
 * Browser ──wss──▶ WS API ──▶ ControlFn ($connect / $disconnect) ──▶ DDB connections table
 *                                                                      ▲
 * Worker ──REST──▶ /internal/ws-event ──▶ ForwardFn ──post_to_connection──┘
 *
 * Auth: HMAC-signed tickets issued by REST /ws-ticket (Cognito-authed).
 * Tickets live 60 sec. Long enough for browser to open the WS, short enough
 * to limit replay risk.
 *
 * Connection table TTL: 6 hours (cleanup safety net; API GW idle-disconnects
 * after 10 min anyway).
 */
export class WebSocketStack extends Stack {
  public readonly wsApi: apigwv2.WebSocketApi;
  public readonly wsStage: apigwv2.WebSocketStage;
  public readonly connectionsTable: dynamodb.Table;
  public readonly controlFn: lambda.Function;

  constructor(scope: Construct, id: string, props: WebSocketStackProps) {
    super(scope, id, props);
    const { config, storage } = props;

    // ----- Connections table -----
    this.connectionsTable = new dynamodb.Table(this, 'ConnectionsTable', {
      tableName: 'comfy-ws-connections',
      partitionKey: { name: 'connection_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      removalPolicy: RemovalPolicy.DESTROY,
    });
    // GSI for ForwardFn to look up connections by the ComfyUI client_id.
    this.connectionsTable.addGlobalSecondaryIndex({
      indexName: 'client-id-index',
      partitionKey: { name: 'client_id', type: dynamodb.AttributeType.STRING },
    });

    // ----- Control Lambda ($connect / $disconnect) -----
    this.controlFn = new lambda.Function(this, 'WsControlFn', {
      functionName: 'comfy-ws-control',
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      timeout: Duration.seconds(10),
      memorySize: 256,
      logRetention: logs.RetentionDays.ONE_WEEK,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../../services/websocket')),
      handler: 'control.lambda_handler',
      description: 'WebSocket $connect / $disconnect — validates ticket, stores connection',
      environment: {
        CONNECTIONS_TABLE: this.connectionsTable.tableName,
        WS_TICKET_SECRET_ARN: storage.wsTicketSecret.secretArn,
      },
    });
    this.connectionsTable.grantReadWriteData(this.controlFn);
    storage.wsTicketSecret.grantRead(this.controlFn);

    // ----- WebSocket API -----
    this.wsApi = new apigwv2.WebSocketApi(this, 'WsApi', {
      apiName: 'comfy-ws',
      description: 'ComfyUI live event streaming',
      connectRouteOptions: {
        integration: new integrations.WebSocketLambdaIntegration(
          'ConnectIntegration',
          this.controlFn,
        ),
      },
      disconnectRouteOptions: {
        integration: new integrations.WebSocketLambdaIntegration(
          'DisconnectIntegration',
          this.controlFn,
        ),
      },
      // No $default — clients don't send anything (server push only).
    });

    this.wsStage = new apigwv2.WebSocketStage(this, 'WsStage', {
      webSocketApi: this.wsApi,
      stageName: 'v1',
      autoDeploy: true,
    });

    // Persist the WS endpoint so dispatcher's /ws-ticket can return it.
    new ssm.StringParameter(this, 'WsApiEndpointParam', {
      parameterName: '/comfy/ws/endpoint',
      stringValue: this.wsStage.url, // wss://<api-id>.execute-api.<region>.amazonaws.com/v1
    });
    // Also persist the management endpoint (https://, used by ForwardFn).
    new ssm.StringParameter(this, 'WsApiMgmtEndpointParam', {
      parameterName: '/comfy/ws/mgmt-endpoint',
      stringValue: `https://${this.wsApi.apiId}.execute-api.${this.region}.amazonaws.com/${this.wsStage.stageName}`,
    });
    new ssm.StringParameter(this, 'WsConnectionsTableParam', {
      parameterName: '/comfy/ws/connections-table',
      stringValue: this.connectionsTable.tableName,
    });
  }
}
