// Outputs viewer — a live queue strip above a paginated grid of completed
// generations. Each tile has a delete button (removes the job + its S3
// objects). Presigned image URLs are cached client-side so reloads don't
// re-pull every image.
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API = (cfg.apiUrl || "").replace(/\/$/, "");
  const grid = document.getElementById("output-grid");
  const queueStrip = document.getElementById("queue-strip");
  const pager = document.getElementById("pager");

  const PER_PAGE = 105; // 7 across × 15 rows
  const URL_CACHE_KEY = "viewer.urlcache";
  const URL_CACHE_TTL = 3300 * 1000; // ~55 min; the presigned URL itself lives 1 h

  let allItems = []; // flat list of { key, job } — one entry per output file
  let page = 0;

  function authHeaders() {
    const h = {};
    const t = window.comfyAuth?.getIdToken();
    if (t) h["Authorization"] = t;
    return h;
  }

  // auth.js restores the token from the refresh token asynchronously — wait
  // for it before the first API call so a fresh tab doesn't 401.
  async function ensureAuth() {
    if (window.comfyAuth?.isSignedIn()) return true;
    if (window.comfyAuth?.refreshToken) {
      try {
        return await window.comfyAuth.refreshToken();
      } catch (_e) {
        return false;
      }
    }
    return false;
  }

  // fetch the API with auth; refresh the token and retry once on 401.
  async function authedFetch(path, opts = {}) {
    const build = () => ({ ...opts, headers: { ...(opts.headers || {}), ...authHeaders() } });
    let r = await fetch(`${API}${path}`, build());
    if (r.status === 401 && (await ensureAuth())) {
      r = await fetch(`${API}${path}`, build());
    }
    return r;
  }

  // ----- presigned-URL cache (localStorage, keyed by S3 key) -----
  // Generated images never change; only the presigned URL rotates. Caching
  // the URL for its lifetime means reloads reuse the same URL, so the
  // browser's image cache serves the bytes instead of re-downloading.
  function loadUrlCache() {
    try {
      return JSON.parse(localStorage.getItem(URL_CACHE_KEY) || "{}");
    } catch (_e) {
      return {};
    }
  }
  let urlCache = loadUrlCache();

  async function presignedUrl(key) {
    const hit = urlCache[key];
    if (hit && hit.exp > Date.now()) return hit.url;
    const r = await authedFetch(`/view?key=${encodeURIComponent(key)}&json=1`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const url = (await r.json()).url;
    urlCache[key] = { url, exp: Date.now() + URL_CACHE_TTL };
    try {
      localStorage.setItem(URL_CACHE_KEY, JSON.stringify(urlCache));
    } catch (_e) {
      // localStorage full — drop the cache and carry on uncached.
      urlCache = {};
    }
    return url;
  }

  // ----- delete -----
  async function deleteJob(jobId, tileEl) {
    if (!confirm("Delete this generation? It will be removed from S3 too.")) return;
    tileEl.classList.add("deleting");
    try {
      const r = await authedFetch(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      allItems = allItems.filter((it) => it.job.job_id !== jobId);
      if (page > 0 && page * PER_PAGE >= allItems.length) page -= 1;
      renderPage();
    } catch (e) {
      tileEl.classList.remove("deleting");
      alert(`Delete failed: ${e.message}`);
    }
  }

  // ----- one tile -----
  function renderTile(item) {
    const { key, job } = item;
    const filename = key.split("/").pop();
    const isVideo = /\.(mp4|webm|mov|gif|mkv)$/i.test(filename);
    const when = job?.created_at ? job.created_at.replace("T", " ").slice(0, 16) : "";
    const el = document.createElement("div");
    el.className = "thumb";
    el.innerHTML =
      `<button class="del" title="Delete">×</button>` +
      `<div class="media"><p class="loading">…</p></div>` +
      `<p>${escape(filename)}</p><p class="when">${escape(when)}</p>`;
    el.querySelector(".del").addEventListener("click", () => deleteJob(job.job_id, el));
    grid.appendChild(el); // appended in order synchronously; media fills in async
    presignedUrl(key)
      .then((url) => {
        el.querySelector(".media").innerHTML = isVideo
          ? `<video controls preload="metadata" src="${url}#t=0.1"></video>`
          : `<a href="${url}" target="_blank" rel="noopener">` +
            `<img src="${url}" loading="lazy" alt="${escape(filename)}" /></a>`;
      })
      .catch((e) => {
        el.querySelector(".media").innerHTML = `<p class="error">${escape(e.message)}</p>`;
      });
  }

  // ----- current page -----
  function renderPage() {
    const total = allItems.length;
    const pages = Math.max(1, Math.ceil(total / PER_PAGE));
    if (page >= pages) page = pages - 1;
    if (page < 0) page = 0;
    grid.innerHTML = "";
    const slice = allItems.slice(page * PER_PAGE, page * PER_PAGE + PER_PAGE);
    if (!slice.length) {
      grid.innerHTML = '<p class="hint">No images yet — generate one in the editor.</p>';
      pager.innerHTML = "";
      return;
    }
    for (const item of slice) renderTile(item);
    pager.innerHTML =
      `<button id="pg-prev"${page <= 0 ? " disabled" : ""}>‹ Prev</button>` +
      `<span>Page ${page + 1} / ${pages} · ${total} image${total !== 1 ? "s" : ""}</span>` +
      `<button id="pg-next"${page >= pages - 1 ? " disabled" : ""}>Next ›</button>`;
    const prev = document.getElementById("pg-prev");
    const next = document.getElementById("pg-next");
    if (prev) prev.onclick = () => { page -= 1; renderPage(); window.scrollTo(0, 0); };
    if (next) next.onclick = () => { page += 1; renderPage(); window.scrollTo(0, 0); };
  }

  // ----- load the gallery -----
  async function load() {
    grid.innerHTML = '<p class="loading">loading…</p>';
    if (!(await ensureAuth())) {
      grid.innerHTML =
        '<p class="error">Not signed in — <a href="/login.html">log in</a> and reopen.</p>';
      return;
    }
    try {
      const r = await authedFetch(`/jobs?status=complete&limit=500`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const jobs = ((await r.json()).jobs || []).filter((j) => (j.output_keys || []).length);
      allItems = [];
      for (const job of jobs) {
        for (const key of job.output_keys) allItems.push({ key, job });
      }
      renderPage();
    } catch (e) {
      grid.innerHTML = `<p class="error">load failed: ${e.message}</p>`;
    }
  }

  // ----- live queue strip -----
  function ago(iso) {
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return `${Math.floor(s)}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    return `${Math.floor(s / 3600)}h`;
  }

  let lastQueueIds = null;

  async function renderQueue() {
    if (!(await ensureAuth())) return;
    let jobs;
    try {
      const r = await authedFetch(`/jobs?status=queued,running&limit=50`);
      if (!r.ok) return;
      jobs = (await r.json()).jobs || [];
    } catch (_e) {
      return;
    }
    jobs.sort((a, b) =>
      a.status === b.status
        ? (a.created_at || "").localeCompare(b.created_at || "")
        : a.status === "running"
        ? -1
        : 1
    );
    if (!jobs.length) {
      queueStrip.innerHTML =
        '<span class="q-label">Queue</span><span class="q-empty">empty — nothing generating</span>';
    } else {
      let html = `<span class="q-label">Queue (${jobs.length})</span>`;
      for (const j of jobs) {
        const st = j.status === "running" ? "running" : "queued";
        const age = j.created_at ? ` · ${ago(j.created_at)}` : "";
        html += `<span class="q-chip ${st}"><span class="q-dot"></span>${escape(
          j.type || "job"
        )} · ${st}${age}</span>`;
      }
      queueStrip.innerHTML = html;
    }
    // A job leaving the queue (likely) finished — refresh the gallery.
    const curIds = new Set(jobs.map((j) => j.job_id));
    if (lastQueueIds && [...lastQueueIds].some((id) => !curIds.has(id))) {
      load();
    }
    lastQueueIds = curIds;
  }

  (async () => {
    await load();
    await renderQueue();
    setInterval(renderQueue, 6000);
  })();

  function escape(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();
