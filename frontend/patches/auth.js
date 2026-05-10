// Cognito sign-in wrapper. Loaded into the ComfyUI frontend via build.sh.
// Provides window.comfyAuth.{getIdToken, signIn, signOut, isSignedIn}.
//
// On page load: if no valid token, redirect to /login.html.
// Other scripts (api-shim.js, models.js, viewer.js) read window.comfyAuth.getIdToken()
// before making API calls.
//
// Storage: ID token in sessionStorage (cleared on tab close). Refresh token in
// localStorage (so user doesn't have to sign in every session).
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const REGION = cfg.region;
  const POOL = cfg.cognitoUserPoolId;
  const CLIENT = cfg.cognitoUserPoolClientId;

  if (!REGION || !POOL || !CLIENT) {
    console.error("COMFY_CONFIG missing — config.js not loaded?", cfg);
    return;
  }

  const COGNITO_URL = `https://cognito-idp.${REGION}.amazonaws.com/`;

  const TOKEN_KEY = "comfy.idToken";
  const TOKEN_EXP_KEY = "comfy.idToken.exp";
  const REFRESH_KEY = "comfy.refreshToken";

  function getIdToken() {
    const exp = parseInt(sessionStorage.getItem(TOKEN_EXP_KEY) || "0", 10);
    if (exp && Date.now() / 1000 < exp - 60) {
      return sessionStorage.getItem(TOKEN_KEY);
    }
    return null;
  }

  function isSignedIn() {
    return !!getIdToken();
  }

  async function signIn(username, password) {
    const r = await fetch(COGNITO_URL, {
      method: "POST",
      headers: {
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
        "Content-Type": "application/x-amz-json-1.1",
      },
      body: JSON.stringify({
        AuthFlow: "USER_PASSWORD_AUTH",
        ClientId: CLIENT,
        AuthParameters: { USERNAME: username, PASSWORD: password },
      }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.message || `auth failed: ${r.status}`);
    }
    const data = await r.json();
    if (data.ChallengeName) {
      throw new Error(
        `Cognito requires ${data.ChallengeName} — handle in console: ` +
          "aws cognito-idp admin-set-user-password ..."
      );
    }
    const auth = data.AuthenticationResult;
    if (!auth) throw new Error("no AuthenticationResult");

    sessionStorage.setItem(TOKEN_KEY, auth.IdToken);
    sessionStorage.setItem(
      TOKEN_EXP_KEY,
      String(Math.floor(Date.now() / 1000) + (auth.ExpiresIn || 3600))
    );
    if (auth.RefreshToken) {
      localStorage.setItem(REFRESH_KEY, auth.RefreshToken);
    }
  }

  async function refreshToken() {
    const refresh = localStorage.getItem(REFRESH_KEY);
    if (!refresh) return false;
    try {
      const r = await fetch(COGNITO_URL, {
        method: "POST",
        headers: {
          "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
          "Content-Type": "application/x-amz-json-1.1",
        },
        body: JSON.stringify({
          AuthFlow: "REFRESH_TOKEN_AUTH",
          ClientId: CLIENT,
          AuthParameters: { REFRESH_TOKEN: refresh },
        }),
      });
      if (!r.ok) return false;
      const data = await r.json();
      const auth = data.AuthenticationResult;
      if (!auth) return false;
      sessionStorage.setItem(TOKEN_KEY, auth.IdToken);
      sessionStorage.setItem(
        TOKEN_EXP_KEY,
        String(Math.floor(Date.now() / 1000) + (auth.ExpiresIn || 3600))
      );
      return true;
    } catch (e) {
      console.warn("refresh failed", e);
      return false;
    }
  }

  function signOut() {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(TOKEN_EXP_KEY);
    localStorage.removeItem(REFRESH_KEY);
    window.location.href = "/login.html";
  }

  // Auto-redirect to login if no valid token (skipped on /login.html itself).
  async function ensureSignedIn() {
    if (window.location.pathname.endsWith("/login.html")) return;
    if (isSignedIn()) return;
    const refreshed = await refreshToken();
    if (!refreshed) {
      window.location.href = "/login.html";
    }
  }

  window.comfyAuth = { getIdToken, signIn, signOut, isSignedIn, refreshToken };
  ensureSignedIn();
})();
