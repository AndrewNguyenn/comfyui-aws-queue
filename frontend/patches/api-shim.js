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
// WebSocket interception is OUT OF SCOPE — we don't run a WS server. The
// editor's `${api_base}/ws` connection will fail silently; live preview frames
// won't render but workflows can still be queued via POST /api/prompt.
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
    "/extensions",
    "/models",
    "/jobs",
    "/downloads",
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
    if (r.status === 401 || r.status === 403) {
      console.warn("auth rejected; signing out");
      if (window.comfyAuth) window.comfyAuth.signOut();
    }
    return r;
  };
})();
