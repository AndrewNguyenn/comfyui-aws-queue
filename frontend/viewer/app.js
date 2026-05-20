// ComfyUI · Outputs — Kenrokuen-language viewer.
// Live pending strip on top, then tabs / search / a paginated grid of
// generations. Backed by GET /jobs, /view, DELETE /jobs/{id}.
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API = (cfg.apiUrl || "").replace(/\/$/, "");

  const grid = document.getElementById("grid");
  const pager = document.getElementById("pager");
  const pending = document.getElementById("pending");
  const tabsEl = document.getElementById("tabs");
  const searchEl = document.getElementById("search");
  const hdrCount = document.getElementById("hdr-count");

  const PER_PAGE = 105; // 7 across × 15 rows
  const URL_CACHE_KEY = "viewer.urlcache";
  const URL_CACHE_TTL = 3300 * 1000; // ~55 min (presign lives 1 h)

  let allItems = []; // [{ key, job, isVideo }]
  let tab = "all";
  let query = "";
  let page = 0;
  let modalIdx = -1;

  /* ---------- auth ---------- */
  function authHeaders() {
    const h = {};
    const t = window.comfyAuth?.getIdToken();
    if (t) h["Authorization"] = t;
    return h;
  }
  async function ensureAuth() {
    if (window.comfyAuth?.isSignedIn()) return true;
    if (window.comfyAuth?.refreshToken) {
      try { return await window.comfyAuth.refreshToken(); } catch (_e) { return false; }
    }
    return false;
  }
  async function authedFetch(path, opts = {}) {
    const build = () => ({ ...opts, headers: { ...(opts.headers || {}), ...authHeaders() } });
    let r = await fetch(`${API}${path}`, build());
    if (r.status === 401 && (await ensureAuth())) r = await fetch(`${API}${path}`, build());
    return r;
  }

  /* ---------- presigned-URL cache ---------- */
  let urlCache = (() => {
    try { return JSON.parse(localStorage.getItem(URL_CACHE_KEY) || "{}"); }
    catch (_e) { return {}; }
  })();
  async function presignedUrl(key) {
    const hit = urlCache[key];
    if (hit && hit.exp > Date.now()) return hit.url;
    const r = await authedFetch(`/view?key=${encodeURIComponent(key)}&json=1`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const url = (await r.json()).url;
    urlCache[key] = { url, exp: Date.now() + URL_CACHE_TTL };
    try { localStorage.setItem(URL_CACHE_KEY, JSON.stringify(urlCache)); }
    catch (_e) { urlCache = {}; }
    return url;
  }

  /* ---------- helpers ---------- */
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const isVideoKey = (k) => /\.(mp4|webm|mov|gif|mkv)$/i.test(k);
  const fmtWhen = (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "");
  function elapsed(iso) {
    if (!iso) return "";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    const m = Math.floor(s / 60);
    return `${m}:${String(Math.floor(s - m * 60)).padStart(2, "0")}`;
  }
  function agoShort(iso) {
    if (!iso) return "";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return `${Math.floor(s)}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    return `${Math.floor(s / 3600)}h`;
  }
  const SVG_TRASH =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>';
  const SVG_PLAY =
    '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5l13 7.5-13 7.5v-15z"/></svg>';

  /* ---------- data ---------- */
  function filtered() {
    return allItems.filter((it) => {
      if (tab === "images" && it.isVideo) return false;
      if (tab === "videos" && !it.isVideo) return false;
      if (query) {
        const q = query.toLowerCase();
        const fn = it.key.split("/").pop().toLowerCase();
        return fn.includes(q) || (it.job.model || "").toLowerCase().includes(q);
      }
      return true;
    });
  }
  function refreshCounts() {
    const imgs = allItems.filter((it) => !it.isVideo).length;
    const vids = allItems.length - imgs;
    document.getElementById("ct-all").textContent = String(allItems.length).padStart(2, "0");
    document.getElementById("ct-images").textContent = String(imgs).padStart(2, "0");
    document.getElementById("ct-videos").textContent = String(vids).padStart(2, "0");
    hdrCount.textContent = `${allItems.length} file${allItems.length !== 1 ? "s" : ""}`;
  }

  async function loadJobs() {
    if (!(await ensureAuth())) {
      grid.outerHTML = '<div class="empty"><div class="e-title">Not signed in</div>' +
        '<div class="e-body"><a href="/login.html">Log in</a> and reopen.</div></div>';
      return;
    }
    try {
      const r = await authedFetch(`/jobs?status=complete&limit=500`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const jobs = ((await r.json()).jobs || []).filter((j) => (j.output_keys || []).length);
      allItems = [];
      for (const job of jobs) {
        for (const key of job.output_keys) {
          allItems.push({ key, job, isVideo: isVideoKey(key) });
        }
      }
      refreshCounts();
      renderGrid();
    } catch (e) {
      grid.innerHTML = `<div class="empty"><div class="e-title">Couldn't load</div>` +
        `<div class="e-body">${esc(e.message)}</div></div>`;
    }
  }

  /* ---------- grid ---------- */
  function renderGrid() {
    const items = filtered();
    const pages = Math.max(1, Math.ceil(items.length / PER_PAGE));
    if (page >= pages) page = pages - 1;
    if (page < 0) page = 0;
    const slice = items.slice(page * PER_PAGE, page * PER_PAGE + PER_PAGE);

    if (!slice.length) {
      grid.innerHTML =
        `<div class="empty" style="grid-column:1/-1">` +
        `<div class="e-eyebrow">${tab === "videos" ? "No videos" : tab === "images" ? "No images" : "Empty"}</div>` +
        `<div class="e-title">${query ? "Nothing matches that filter." : "Nothing here yet."}</div>` +
        `<div class="e-body">${query ? "Try a different filename or model." : "Generations land here as you run workflows."}</div></div>`;
      pager.innerHTML = "";
      return;
    }
    grid.innerHTML = "";
    slice.forEach((item, i) => grid.appendChild(makeCard(item, page * PER_PAGE + i)));

    pager.innerHTML =
      `<button id="pg-prev"${page <= 0 ? " disabled" : ""}>‹ Prev</button>` +
      `<span>Page ${page + 1} / ${pages} · ${items.length} item${items.length !== 1 ? "s" : ""}</span>` +
      `<button id="pg-next"${page >= pages - 1 ? " disabled" : ""}>Next ›</button>`;
    const pv = document.getElementById("pg-prev"), nx = document.getElementById("pg-next");
    if (pv) pv.onclick = () => { page--; renderGrid(); window.scrollTo(0, 0); };
    if (nx) nx.onclick = () => { page++; renderGrid(); window.scrollTo(0, 0); };
  }

  function makeCard(item, filteredIdx) {
    const { key, job, isVideo } = item;
    const filename = key.split("/").pop();
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML =
      `<div class="thumb-wrap">` +
        `<div class="ph">loading…</div>` +
        `<div class="typestamp ${isVideo ? "vid" : "img"}">${isVideo ? "MP4" : "PNG"}</div>` +
        (isVideo ? `<div class="play">${SVG_PLAY}</div>` : "") +
        `<div class="thumb-actions"><button class="danger" title="Delete">${SVG_TRASH}</button></div>` +
      `</div>` +
      `<div class="body">` +
        `<div class="name" title="${esc(filename)}">${esc(filename)}</div>` +
        `<div class="meta-row"><span>${esc(fmtWhen(job.created_at))}</span>` +
        (job.model ? `<span class="sep">·</span><span>${esc(job.model)}</span>` : "") +
        `</div>` +
      `</div>`;
    el.querySelector(".thumb-actions .danger").addEventListener("click", (e) => {
      e.stopPropagation();
      askDelete(job.job_id);
    });
    el.addEventListener("click", () => openModal(filteredIdx));
    presignedUrl(key)
      .then((url) => {
        const wrap = el.querySelector(".thumb-wrap");
        const ph = wrap.querySelector(".ph");
        const media = isVideo
          ? `<video src="${url}#t=0.1" preload="metadata" muted></video>`
          : `<img src="${url}" loading="lazy" alt="${esc(filename)}" />`;
        ph.insertAdjacentHTML("beforebegin", media);
        ph.remove();
      })
      .catch(() => {});
    return el;
  }

  /* ---------- pending strip ---------- */
  let lastPendingIds = null;
  async function renderPending() {
    if (!(await ensureAuth())) return;
    let jobs;
    try {
      const r = await authedFetch(`/jobs?status=queued,running&limit=50`);
      if (!r.ok) return;
      jobs = (await r.json()).jobs || [];
    } catch (_e) { return; }
    jobs.sort((a, b) =>
      a.status === b.status
        ? (a.created_at || "").localeCompare(b.created_at || "")
        : a.status === "running" ? -1 : 1);

    let rows;
    if (!jobs.length) {
      rows = `<div class="pend-empty">Idle — no active generations.</div>`;
    } else {
      rows = jobs.map((j) => {
        const running = j.status === "running";
        const since = running ? (j.started_at || j.created_at) : j.created_at;
        // Live sampling progress — j.progress is "value/max" (e.g. "7/20").
        let pct = null, frac = "";
        if (running && j.progress && j.progress.includes("/")) {
          const [v, m] = j.progress.split("/").map(Number);
          if (m > 0) { pct = Math.min(100, Math.round((v / m) * 100)); frac = `${v}/${m}`; }
        }
        const right = running
          ? elapsed(since) + (frac ? ` · ${frac}` : "")
          : `waiting ${agoShort(j.created_at)}`;
        const bar = !running
          ? `<div class="pend-bar queued"><i></i></div>`
          : pct != null
          ? `<div class="pend-bar det"><i style="width:${pct}%"></i></div>`
          : `<div class="pend-bar"><i></i></div>`; // indeterminate until 1st progress event
        return `<div class="pend-row">` +
          `<span class="pend-stamp ${running ? "running" : "queued"}">${running ? "running" : "queued"}</span>` +
          `<div><div class="pend-kind">${esc(j.type || "job")}${j.model ? " · " + esc(j.model) : ""}</div>` +
          `${bar}</div>` +
          `<div class="pend-elapsed">${esc(right)}</div></div>`;
      }).join("");
    }
    pending.innerHTML =
      `<div class="pend-card"><div class="pend-head">` +
      `<span>Pending</span>` +
      `<span class="live"><span class="dot"></span>${jobs.length} active</span>` +
      `</div>${rows}</div>`;

    // a job leaving the queue (likely) finished — refresh the gallery
    const cur = new Set(jobs.map((j) => j.job_id));
    if (lastPendingIds && [...lastPendingIds].some((id) => !cur.has(id))) loadJobs();
    lastPendingIds = cur;
  }

  /* ---------- modal ---------- */
  function openModal(filteredIdx) {
    modalIdx = filteredIdx;
    drawModal();
  }
  function closeModal() {
    modalIdx = -1;
    const s = document.querySelector(".scrim");
    if (s) s.remove();
  }
  function drawModal() {
    const items = filtered();
    if (modalIdx < 0 || modalIdx >= items.length) return closeModal();
    const { key, job, isVideo } = items[modalIdx];
    const filename = key.split("/").pop();
    let scrim = document.querySelector(".scrim");
    if (!scrim) {
      scrim = document.createElement("div");
      scrim.className = "scrim";
      scrim.addEventListener("click", (e) => { if (e.target === scrim) closeModal(); });
      document.body.appendChild(scrim);
    }
    scrim.innerHTML =
      `<div class="modal">` +
        `<div class="stage">` +
          `<div class="media-slot" style="display:flex;align-items:center;justify-content:center;width:100%;height:100%"></div>` +
          `<button class="nav prev"${modalIdx <= 0 ? " disabled" : ""}>‹</button>` +
          `<button class="nav next"${modalIdx >= items.length - 1 ? " disabled" : ""}>›</button>` +
        `</div>` +
        `<div class="info">` +
          `<button class="close" title="Close">✕</button>` +
          `<div class="info-head"><div class="eyebrow">${isVideo ? "Video" : "Image"}</div>` +
            `<h2>${esc(filename)}</h2></div>` +
          `<div class="info-body">` +
            `<div class="field"><div class="k">Model</div><div class="v">${esc(job.model || "—")}</div></div>` +
            `<div class="field"><div class="k">Type</div><div class="v">${esc(job.type || "—")}</div></div>` +
            `<div class="field"><div class="k">Created</div><div class="v">${esc(fmtWhen(job.created_at))}</div></div>` +
            `<div class="field"><div class="k">Job</div><div class="v">${esc(job.job_id.slice(0, 8))}</div></div>` +
          `</div>` +
          `<div class="info-foot">` +
            `<button class="dl">Download</button>` +
            `<button class="danger del">Delete</button>` +
          `</div>` +
        `</div>` +
      `</div>`;
    scrim.querySelector(".close").onclick = closeModal;
    scrim.querySelector(".nav.prev").onclick = () => { if (modalIdx > 0) { modalIdx--; drawModal(); } };
    scrim.querySelector(".nav.next").onclick = () => { if (modalIdx < items.length - 1) { modalIdx++; drawModal(); } };
    scrim.querySelector(".del").onclick = () => askDelete(job.job_id);
    presignedUrl(key).then((url) => {
      const slot = scrim.querySelector(".media-slot");
      if (slot) {
        slot.innerHTML = isVideo
          ? `<video src="${url}" controls autoplay muted loop playsinline></video>`
          : `<img src="${url}" alt="${esc(filename)}" />`;
      }
      const dl = scrim.querySelector(".dl");
      if (dl) dl.onclick = () => window.open(url, "_blank", "noopener");
    });
  }

  /* ---------- delete ---------- */
  function askDelete(jobId) {
    const scrim = document.createElement("div");
    scrim.className = "confirm-scrim";
    scrim.innerHTML =
      `<div class="confirm">` +
        `<div class="eyebrow">Delete permanently</div>` +
        `<h3>Delete this generation?</h3>` +
        `<p>The image/video is removed from S3 and the gallery. The output bucket ` +
        `is versioned, so it's technically recoverable — but treat this as permanent.</p>` +
        `<div class="cf-foot"><button class="cancel">Cancel</button>` +
        `<button class="danger ok">Delete</button></div>` +
      `</div>`;
    scrim.addEventListener("click", (e) => { if (e.target === scrim) scrim.remove(); });
    scrim.querySelector(".cancel").onclick = () => scrim.remove();
    scrim.querySelector(".ok").onclick = () => { scrim.remove(); doDelete(jobId); };
    document.body.appendChild(scrim);
  }
  async function doDelete(jobId) {
    try {
      const r = await authedFetch(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      allItems = allItems.filter((it) => it.job.job_id !== jobId);
      refreshCounts();
      if (modalIdx >= 0) closeModal();
      renderGrid();
      showToast("Deleted.");
    } catch (e) {
      showToast(`Delete failed: ${e.message}`);
    }
  }
  function showToast(msg) {
    document.querySelectorAll(".toast").forEach((t) => t.remove());
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3500);
  }

  /* ---------- wiring ---------- */
  tabsEl.addEventListener("click", (e) => {
    const t = e.target.closest(".tab");
    if (!t) return;
    tab = t.dataset.tab;
    page = 0;
    tabsEl.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    renderGrid();
  });
  searchEl.addEventListener("input", () => {
    query = searchEl.value.trim();
    page = 0;
    renderGrid();
  });
  document.addEventListener("keydown", (e) => {
    if (modalIdx < 0) return;
    if (e.key === "Escape") closeModal();
    else if (e.key === "ArrowLeft" && modalIdx > 0) { modalIdx--; drawModal(); }
    else if (e.key === "ArrowRight") { const n = filtered().length; if (modalIdx < n - 1) { modalIdx++; drawModal(); } }
  });

  /* ---------- init ---------- */
  (async () => {
    await loadJobs();
    await renderPending();
    setInterval(renderPending, 4000);
  })();
})();
