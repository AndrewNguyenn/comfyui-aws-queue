// API base URL + auth shim for the bundled ComfyUI workflow editor.
//
// The editor's `ComfyApi` class builds same-origin URLs via two helpers:
//   apiURL(path)  -> `${api_base}/api${path}`   (most calls — /prompt, /view,
//                                                /upload/image, /object_info,
//                                                /history -> /jobs, …)
//   fileURL(path) -> `${api_base}${path}`        (e.g. /templates/*.json)
// With api_base = '' (browser deployment) those resolve to bare paths like
// `/api/prompt`, `/api/view?...`, `/templates/index.json`, etc.
//
// This shim intercepts window.fetch for the `/api/*` paths and rewrites them to
// our API Gateway origin, attaching the Cognito ID token. Everything else
// (static assets under /assets/*, templates, materialdesignicons.min.css …)
// is left to the same S3 origin.
//
// We also strip the `Comfy-User` header the editor adds to every fetch — our
// API Gateway's CORS allowlist only includes Content-Type / Authorization /
// X-Api-Key, and a `Comfy-User` header would fail preflight.
//
// (The standalone /login, /models, /viewer pages call `${API}/path` directly
// using the absolute apiUrl from window.COMFY_CONFIG, so they don't depend on
// this shim — but we still rewrite bare relative paths like `/prompt` for
// good measure, in case a stray script in the editor uses fetch() directly.)
//
// WebSocket: intercepted below. The editor opens `wss://<host>/ws` (or
// `/api/ws`) for live progress events. We swap that for a connection to our
// API GW WebSocket API, after fetching a short-lived HMAC ticket from the
// REST /ws-ticket endpoint (Cognito-authed). Worker-side bridge forwards
// ComfyUI's local WS events through to the right browser by client_id.
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API_BASE = (cfg.apiUrl || "").replace(/\/$/, "");

  if (!API_BASE) {
    console.error("COMFY_CONFIG.apiUrl missing — config.js not loaded?");
    return;
  }

  // Bare-path allowlist (used by our standalone pages and as a fallback). The
  // editor itself almost always goes through `/api/<path>`; the leading-`/api`
  // case is handled separately below.
  //
  // NOTE: `/extensions` is intentionally NOT in this list. The editor loads
  // custom-node JS via raw `/extensions/<path>` URLs (not `/api/extensions/`).
  // Those need to hit the frontend S3 origin (where extensions_publisher
  // uploads them), NOT our REST API. The bare `/api/extensions` (the
  // discovery endpoint) is handled by the `/api/` prefix branch above.
  const COMFY_API_PATHS = [
    "/prompt",
    "/history",
    "/object_info",
    "/queue",
    "/view",
    "/upload/image",
    "/upload/mask",
    "/system_stats",
    "/embeddings",
    "/models",
    "/jobs",
    "/downloads",
    // ComfyUI-Manager UI fetches these as bare paths (no /api prefix).
    // Routed through dispatcher Lambda → metadata instance.
    "/manager",
    "/customnode",
    "/snapshot",
    "/model-manager",
    "/externalmodel",
    "/civitai",
    // Editor LOGS panel: getRawLogs()/subscribeLogs() build internalURL()
    // paths — bare `/internal/logs/raw` and `/internal/logs/subscribe` (no
    // /api prefix). Route them through the dispatcher → metadata instance.
    "/internal/logs",
  ];

  function rewriteUrl(url) {
    if (typeof url !== "string") return url;
    // Already absolute (other origin) — leave alone.
    if (/^https?:\/\//.test(url)) return url;

    // ComfyUI editor convention: every API call is `/api/<path>` (with optional
    // query string). Rewrite to `${API_BASE}/<path>` — our API GW does NOT
    // mount routes under /api, so we strip the prefix.
    if (url === "/api" || url.startsWith("/api/") || url.startsWith("/api?")) {
      return API_BASE + url.slice(4); // drop the leading "/api"
    }

    // Bare paths used by our standalone pages (and any extension that calls
    // fetch('/prompt') directly).
    for (const p of COMFY_API_PATHS) {
      if (url === p || url.startsWith(p + "/") || url.startsWith(p + "?")) {
        return API_BASE + url;
      }
    }
    return url;
  }

  const origFetch = window.fetch.bind(window);

  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : input.url;
    const newUrl = rewriteUrl(url);
    if (newUrl === url) {
      return origFetch(input, init);
    }

    const opts = init ? { ...init } : {};
    opts.headers = new Headers(opts.headers || {});

    // The editor adds `Comfy-User` to every request; our API GW CORS doesn't
    // permit it, which trips the browser's preflight. Strip it.
    opts.headers.delete("Comfy-User");

    const token = window.comfyAuth ? window.comfyAuth.getIdToken() : null;
    if (token) {
      opts.headers.set("Authorization", token);
    }

    const r = await origFetch(newUrl, opts);
    // A 401 means the ID token expired or was rejected. Try the refresh token
    // before giving up — the ID token only lives ~a day but the refresh token
    // is good for weeks, so an expired ID token should be renewed silently,
    // NOT treated as a sign-out. Only sign out if the refresh itself fails
    // (refresh token expired/revoked) — that's a genuinely dead session.
    //
    // 403 is NOT handled here: API Gateway returns 403 for unmatched routes
    // ("Missing Authentication Token") and per-IP throttling, neither of which
    // means the session is bad.
    if (r.status === 401 && window.comfyAuth) {
      const refreshed = await window.comfyAuth.refreshToken();
      if (!refreshed) {
        console.warn("auth rejected (401) and refresh failed; signing out");
        window.comfyAuth.signOut();
        return r;
      }
      // Retry once with the fresh token — but only when the request is safely
      // replayable. A FormData/Blob/stream body is consumed by the first send
      // and can't be re-fetched (image uploads), and a Request-object `input`
      // isn't fully captured by `opts`. In those cases the token is still
      // refreshed so the next request succeeds; this one is returned as-is.
      const body = opts.body;
      const replayable =
        typeof input === "string" &&
        (body == null || typeof body === "string" || body instanceof URLSearchParams);
      if (!replayable) return r;
      const fresh = window.comfyAuth.getIdToken();
      if (fresh) opts.headers.set("Authorization", fresh);
      const retry = await origFetch(newUrl, opts);
      if (retry.status === 401) console.warn("request still 401 after token refresh");
      return retry;
    }
    return r;
  };

  // ----- XMLHttpRequest interception -----
  //
  // The editor's newer queue/history menu uses axios, which uses XHR — not
  // fetch. Without this wrapper, /api/userdata/*, /api/i18n, etc. resolve
  // against the S3 frontend origin and 404. We patch XMLHttpRequest.prototype
  // to apply the same rewriteUrl + Authorization header logic as fetch above.
  //
  // We DON'T touch 401s on the XHR path: XHR can race in ways fetch can't
  // (multiple in-flight axios calls firing on page load), and re-sending an
  // XHR with a fresh token is awkward (XHR isn't promise-based, body replay
  // is worse). Known limitation: XHR requests (the axios-based queue/history
  // menu) don't get the refresh-and-retry the fetch path does — an expired
  // token makes them fail silently until the next page load refreshes it.
  // With a 24 h token that window is small. The fetch path carries sign-out.
  const XHRProto = XMLHttpRequest.prototype;
  const origOpen = XHRProto.open;
  const origSend = XHRProto.send;
  const origSetHeader = XHRProto.setRequestHeader;

  XHRProto.open = function (method, url, async, user, password) {
    const newUrl = rewriteUrl(url);
    this.__comfyRewritten = newUrl !== url;
    return origOpen.call(
      this,
      method,
      newUrl,
      async !== undefined ? async : true,
      user,
      password
    );
  };

  XHRProto.setRequestHeader = function (name, value) {
    // Drop Comfy-User on rewritten requests (API GW CORS doesn't permit it).
    if (this.__comfyRewritten && /^comfy-user$/i.test(name)) return;
    return origSetHeader.call(this, name, value);
  };

  XHRProto.send = function (body) {
    if (this.__comfyRewritten) {
      const token = window.comfyAuth ? window.comfyAuth.getIdToken() : null;
      if (token) {
        try {
          origSetHeader.call(this, "Authorization", token);
        } catch (_e) {
          // setRequestHeader can only be called after open() and before send();
          // we're between them so this should always work, but swallow defensively.
        }
      }
    }
    return origSend.call(this, body);
  };

  // ----- WebSocket interception -----
  //
  // The editor opens `ws(s)://<origin>/api/ws?...` (or sometimes `/ws`). We
  // detect that pattern, fetch a fresh HMAC ticket from /ws-ticket, then open
  // the real WebSocket against our API GW WebSocket endpoint with the ticket
  // in the query string. We expose the underlying WebSocket via a Proxy that
  // queues sends until the real socket is open.
  const OriginalWebSocket = window.WebSocket;

  function isComfyWsUrl(url) {
    if (typeof url !== "string") return false;
    // Same-origin path-style: '/api/ws?...' or '/ws?...'
    if (url.startsWith("/api/ws") || url === "/api/ws") return true;
    if (url.startsWith("/ws") || url === "/ws") return true;
    // Absolute URL whose path matches: 'wss://localhost/api/ws?...'
    try {
      const u = new URL(url, window.location.href);
      return u.pathname === "/api/ws" || u.pathname === "/ws";
    } catch (_e) {
      return false;
    }
  }

  // Cache the ticket+url for ~50 sec (ticket is valid for 60). Avoids hitting
  // /ws-ticket on every reconnect attempt during a flaky session.
  let _ticketCache = { value: null, expiresAt: 0 };
  async function fetchTicket() {
    const now = Date.now();
    if (_ticketCache.value && now < _ticketCache.expiresAt) {
      return _ticketCache.value;
    }
    const t = window.comfyAuth ? window.comfyAuth.getIdToken() : null;
    if (!t) throw new Error("not signed in");
    const r = await origFetch(`${API_BASE}/ws-ticket`, {
      headers: { Authorization: t },
    });
    if (!r.ok) throw new Error(`ws-ticket HTTP ${r.status}`);
    const data = await r.json();
    _ticketCache = {
      value: data,
      expiresAt: now + 50_000,
    };
    return data;
  }

  // Wrapper: behaves like a WebSocket but does the async ticket fetch first.
  // Implements the readyState/onopen/onmessage/onclose/onerror surface.
  function ComfyWebSocket(_url, _protocols) {
    const self = this;
    self.readyState = 0; // CONNECTING
    self.bufferedAmount = 0;
    self.url = _url;
    self.protocol = "";
    self.binaryType = "blob";

    const listeners = { open: [], message: [], close: [], error: [] };
    const pending = []; // queued sends until inner WS is ready
    let inner = null;
    let closed = false;

    function emit(name, evt) {
      // Direct .onX handler
      const h = self["on" + name];
      if (typeof h === "function") {
        try { h.call(self, evt); } catch (e) { console.error(e); }
      }
      // Listeners added via addEventListener
      for (const fn of listeners[name] || []) {
        try { fn.call(self, evt); } catch (e) { console.error(e); }
      }
    }

    self.addEventListener = (name, fn) => {
      (listeners[name] = listeners[name] || []).push(fn);
    };
    self.removeEventListener = (name, fn) => {
      listeners[name] = (listeners[name] || []).filter((f) => f !== fn);
    };
    self.send = (data) => {
      if (closed) throw new Error("WebSocket is closed");
      if (inner && inner.readyState === 1) {
        inner.send(data);
      } else {
        pending.push(data);
      }
    };
    self.close = (code, reason) => {
      closed = true;
      self.readyState = 3; // CLOSED
      if (inner) inner.close(code, reason);
    };

    fetchTicket()
      .then(({ ws_url, ticket, client_id }) => {
        if (closed) return;
        // Append client_id from the original URL if present (editor passes one).
        let originalClient = "";
        try {
          const u = new URL(_url, window.location.href);
          originalClient = u.searchParams.get("clientId") || "";
        } catch (_e) {}
        const cid = originalClient || client_id;
        const realUrl = `${ws_url}?ticket=${encodeURIComponent(ticket)}&client_id=${encodeURIComponent(cid)}`;
        // Stash so callers / debug can read it
        self.url = realUrl;
        // Expose client_id so calling code (and prompt submission) can use it
        window.comfyClientId = cid;

        inner = new OriginalWebSocket(realUrl);
        inner.binaryType = self.binaryType;
        inner.onopen = (e) => {
          self.readyState = 1; // OPEN
          // Flush any queued sends
          for (const d of pending.splice(0)) {
            try { inner.send(d); } catch (err) { console.error(err); }
          }
          emit("open", e);
        };
        inner.onmessage = (e) => emit("message", e);
        inner.onclose = (e) => {
          self.readyState = 3;
          emit("close", e);
        };
        inner.onerror = (e) => emit("error", e);
      })
      .catch((err) => {
        console.error("ws ticket fetch failed:", err);
        self.readyState = 3;
        emit("error", err);
        emit("close", { code: 1006, reason: "ticket fetch failed" });
      });
  }
  // Mirror the standard WebSocket constants on our wrapper.
  ComfyWebSocket.CONNECTING = 0;
  ComfyWebSocket.OPEN = 1;
  ComfyWebSocket.CLOSING = 2;
  ComfyWebSocket.CLOSED = 3;
  ComfyWebSocket.prototype.CONNECTING = 0;
  ComfyWebSocket.prototype.OPEN = 1;
  ComfyWebSocket.prototype.CLOSING = 2;
  ComfyWebSocket.prototype.CLOSED = 3;

  window.WebSocket = function (url, protocols) {
    if (isComfyWsUrl(url)) {
      return new ComfyWebSocket(url, protocols);
    }
    return new OriginalWebSocket(url, protocols);
  };
  // Inherit constants on the new constructor too.
  window.WebSocket.CONNECTING = 0;
  window.WebSocket.OPEN = 1;
  window.WebSocket.CLOSING = 2;
  window.WebSocket.CLOSED = 3;
})();
