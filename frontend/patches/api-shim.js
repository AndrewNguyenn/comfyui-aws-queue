// API base URL + auth shim for the vanilla ComfyUI frontend.
//
// ComfyUI's frontend by default talks to a same-origin backend at /prompt,
// /history, etc. This shim intercepts fetch() so those relative URLs are
// rewritten to point at our API Gateway, with the Cognito ID token added
// to the Authorization header.
//
// (Note: when ComfyUI's frontend uses WebSocket — which we don't support in
// v1 — those would need a separate intercept.)
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API_BASE = (cfg.apiUrl || "").replace(/\/$/, "");

  if (!API_BASE) {
    console.error("COMFY_CONFIG.apiUrl missing — config.js not loaded?");
    return;
  }

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
  ];

  function rewriteUrl(url) {
    if (typeof url !== "string") return url;
    // Absolute URLs (other origins) — leave alone.
    if (/^https?:\/\//.test(url)) return url;
    // Relative URLs: rewrite if they look like a ComfyUI API call.
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
