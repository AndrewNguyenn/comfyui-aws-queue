// Outputs viewer — a live queue strip (jobs waiting / generating) above a
// grid of completed generations. Polls GET /jobs; images come via /view.
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API = (cfg.apiUrl || "").replace(/\/$/, "");
  const grid = document.getElementById("output-grid");
  const queueStrip = document.getElementById("queue-strip");

  function authHeaders() {
    const h = {};
    const t = window.comfyAuth?.getIdToken();
    if (t) h["Authorization"] = t;
    return h;
  }

  // The ID token lives in sessionStorage (per-tab); a freshly opened viewer
  // tab has none. auth.js recovers it from the localStorage refresh token,
  // but that's async — wait for a usable token before the first API call.
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

  // fetch against the API with auth; refresh the token and retry once on 401.
  async function authedFetch(path) {
    let r = await fetch(`${API}${path}`, { headers: authHeaders() });
    if (r.status === 401 && (await ensureAuth())) {
      r = await fetch(`${API}${path}`, { headers: authHeaders() });
    }
    return r;
  }

  // Resolve a key → presigned S3 URL and append a thumbnail tile.
  async function renderThumb(key, job) {
    const filename = key.split("/").pop();
    const isVideo = /\.(mp4|webm|mov|gif|mkv)$/i.test(filename);
    const el = document.createElement("div");
    el.className = "thumb";
    try {
      // /view?json=1 returns {url: <presigned>} (a plain 302 can't be read
      // cross-origin from fetch). The presigned URL needs no auth header.
      const resp = await authedFetch(`/view?key=${encodeURIComponent(key)}&json=1`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const presigned = (await resp.json()).url;
      const when = job?.created_at ? job.created_at.replace("T", " ").slice(0, 16) : "";
      if (isVideo) {
        el.innerHTML = `
          <video controls preload="metadata" src="${presigned}#t=0.1"></video>
          <p>${escape(filename)}</p><p class="when">${escape(when)}</p>
        `;
      } else {
        el.innerHTML = `
          <a href="${presigned}" target="_blank" rel="noopener">
            <img src="${presigned}" alt="${escape(filename)}" loading="lazy" />
          </a>
          <p>${escape(filename)}</p><p class="when">${escape(when)}</p>
        `;
      }
    } catch (err) {
      el.innerHTML = `<p class="error">${escape(filename)}: ${err.message}</p>`;
    }
    grid.appendChild(el);
  }

  // List all completed jobs and render every output image/video.
  async function load() {
    grid.innerHTML = '<p class="loading">loading…</p>';
    if (!(await ensureAuth())) {
      grid.innerHTML =
        '<p class="error">Not signed in — <a href="/login.html">log in</a> and reopen.</p>';
      return;
    }
    try {
      const r = await authedFetch(`/jobs?status=complete&limit=100`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const jobs = (data.jobs || []).filter((j) => (j.output_keys || []).length);
      if (!jobs.length) {
        grid.innerHTML = '<p class="hint">No images yet — generate one in the editor.</p>';
        return;
      }
      grid.innerHTML = "";
      for (const job of jobs) {
        for (const key of job.output_keys) {
          await renderThumb(key, job);
        }
      }
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

  let lastQueueIds = null; // null until the first successful poll

  async function renderQueue() {
    if (!(await ensureAuth())) return;
    let jobs;
    try {
      const r = await authedFetch(`/jobs?status=queued,running&limit=50`);
      if (!r.ok) return; // keep the last strip on a transient error
      jobs = (await r.json()).jobs || [];
    } catch (_e) {
      return;
    }
    // running first, then queued; oldest-first within each group.
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
    // If a job left the queue since the last poll it (likely) finished —
    // refresh the gallery so the new image shows without a manual reload.
    const curIds = new Set(jobs.map((j) => j.job_id));
    if (lastQueueIds) {
      const finished = [...lastQueueIds].some((id) => !curIds.has(id));
      if (finished) load();
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
