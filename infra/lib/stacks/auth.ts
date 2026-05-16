import { Stack, StackProps, RemovalPolicy, Duration } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import { AppConfig } from '../config';

export interface AuthStackProps extends StackProps {
  readonly config: AppConfig;
}

/**
 * Cognito user pool for API authentication.
 *
 * Single-user deployment — no self-signup. Operator creates the one user
 * after deploy via:
 *   aws cognito-idp admin-create-user \
 *       --user-pool-id <pool-id> --username <username> \
 *       --temporary-password 'TempPass123!' \
 *       --message-action SUPPRESS
 *
 * Frontend uses amazon-cognito-identity-js to sign in (USER_PASSWORD_AUTH flow),
 * gets an ID token, sends as `Authorization: Bearer <token>` to API Gateway.
 *
 * Cost: $0 under 50,000 monthly active users (we have 1).
 */
export class AuthStack extends Stack {
  public readonly userPool: cognito.UserPool;
  public readonly userPoolClient: cognito.UserPoolClient;

  constructor(scope: Construct, id: string, props: AuthStackProps) {
    super(scope, id, props);
    const { config } = props;

    this.userPool = new cognito.UserPool(this, 'UserPool', {
      userPoolName: `${config.projectName}-users`,
      selfSignUpEnabled: false, // Single-user deployment, admin creates user
      signInAliases: { username: true, email: true },
      autoVerify: { email: true },
      passwordPolicy: {
        // Single-user personal deployment — 8 char minimum is acceptable risk.
        minLength: 8,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
        tempPasswordValidity: Duration.days(7),
      },
      mfa: cognito.Mfa.OPTIONAL,
      mfaSecondFactor: { sms: false, otp: true },
      accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
      removalPolicy: RemovalPolicy.RETAIN, // User account is irreplaceable
    });

    // App client used by the browser frontend.
    // No client secret — public clients (browser, mobile) cannot keep secrets.
    this.userPoolClient = new cognito.UserPoolClient(this, 'UserPoolClient', {
      userPool: this.userPool,
      userPoolClientName: `${config.projectName}-web`,
      generateSecret: false,
      authFlows: {
        userPassword: true, // Allow USER_PASSWORD_AUTH from browser
        userSrp: true, // SRP is preferred — frontend lib supports it
      },
      preventUserExistenceErrors: true, // Don't leak whether a username exists
      idTokenValidity: Duration.hours(1),
      accessTokenValidity: Duration.hours(1),
      refreshTokenValidity: Duration.days(30),
      readAttributes: new cognito.ClientAttributes().withStandardAttributes({ email: true }),
    });
  }
}
