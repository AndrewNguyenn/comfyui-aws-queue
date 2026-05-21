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

  const PER_PAGE = 75; // 5 across × 15 rows
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
  // A titled prompt block for the modal info panel. `neg`-styled when the
  // label reads as a negative prompt (the label text comes from the API).
  function promptSection(label, text) {
    text = (text || "").trim();
    if (!text) return "";
    const neg = /negative/i.test(label) ? " neg" : "";
    return `<div class="section-ttl">${esc(label)}</div>` +
      `<div class="prompt${neg}">${esc(text)}</div>`;
  }
  // All prompt sections for a job — one per distinct prompt the workflow used
  // (detailer graphs carry several). Empty → a single muted note.
  function promptsBlock(prompts) {
    const blocks = (prompts || []).map((p) => promptSection(p.label, p.text)).join("");
    return blocks || (`<div class="section-ttl">Prompt</div>` +
      `<div class="prompt muted">Not recorded for this generation.</div>`);
  }
  // Generation parameters — only the fields the workflow actually carried.
  function paramsSection(params) {
    params = params || {};
    const rows = [
      ["Steps", params.steps], ["CFG", params.cfg],
      ["Sampler", params.sampler_name], ["Scheduler", params.scheduler],
      ["Denoise", params.denoise], ["Seed", params.seed],
    ].filter(([, v]) => v != null && v !== "");
    if (!rows.length) return "";
    return `<div class="section-ttl">Parameters</div>` +
      rows.map(([k, v]) =>
        `<div class="field"><div class="k">${k}</div>` +
        `<div class="v">${esc(String(v))}</div></div>`).join("");
  }
  function elapsed(iso) {
    if (!iso) return "";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    const m = Math.floor(s / 60);
    return `${m}:${String(Math.floor(s - m * 60)).padStart(2, "0")}`;
  }
  // Minutes a job has been waiting, and a compact "39m" / "1h 12m" rendering.
  const ageMin = (iso) =>
    iso ? Math.floor(Math.max(0, Date.now() - new Date(iso).getTime()) / 60000) : 0;
  function fmtWait(min) {
    if (min < 1) return "<1m";
    if (min < 60) return `${min}m`;
    return `${Math.floor(min / 60)}h ${min % 60}m`;
  }
  const SVG_TRASH =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>';
  const SVG_PLAY =
    '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5l13 7.5-13 7.5v-15z"/></svg>';
  const SVG_CLOSE =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';

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
      doDelete(job.job_id);
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

  /* ---------- pending workflows panel ----------
     The running job is featured with a live progress meter; the queue is
     collapsed to one row per model (consecutive same-model jobs read as one
     batched intent). Recently-failed jobs surface as a maple FAILED row.
     Every row carries a cancel / dismiss (×) action. */
  let lastPendingIds = null;
  let pendCollapsed = false;   // Hide/Show toggle
  let pendShowAll = false;     // "+ N more queued" reveal
  let pendGroups = [];         // queue groups for the current render (wired below)
  const cancelling = new Set(); // job_ids the user has cancelled, awaiting feed drop

  // Failure tracking. `failedSeen` is the failed job_ids observed as of the
  // previous poll — the toast is a live alert, so the first poll baselines it
  // silently and only later polls toast. `failedDismissed` (localStorage) is
  // failures the user has cleared from the panel; it survives reloads.
  const RECENT_FAIL_MS = 2 * 3600 * 1000; // a failure is "recent" for 2h
  const ROW_COLLAPSE_MS = 260;            // matches the pq-collapse CSS animation
  const FAIL_DISMISS_KEY = "viewer.faildismiss";
  let failedSeen = null;
  // job_id → dismissed-at epoch ms. Keyed by time so it self-prunes: an entry
  // is dropped once it outlives the recency window (past that the row it
  // hides can't render anyway), which keeps localStorage bounded.
  let failedDismissed = (() => {
    try {
      const v = JSON.parse(localStorage.getItem(FAIL_DISMISS_KEY) || "{}");
      return (v && typeof v === "object" && !Array.isArray(v)) ? v : {};
    } catch (_e) { return {}; }
  })();
  function persistDismissed() {
    try { localStorage.setItem(FAIL_DISMISS_KEY, JSON.stringify(failedDismissed)); }
    catch (_e) { /* storage full / disabled — dismissal just won't persist */ }
  }
  const failMsg = (j) => {
    const e = (j.error || "Unknown error.").trim();
    return e.length > 90 ? e.slice(0, 89) + "…" : e;
  };

  // Collapse consecutive same-(type, model) queued jobs into one ledger row.
  // `oldest` is kept as an ISO created_at string — ISO timestamps sort
  // lexically, so a string compare finds the earliest, and ageMin() consumes
  // it directly (no second time representation to drift from).
  function groupQueue(queued) {
    const out = [];
    for (const j of queued) {
      const model = j.model || j.type || "job";
      const last = out[out.length - 1];
      if (last && last.model === model && last.kind === j.type) {
        last.ids.push(j.job_id);
        last.count += 1;
        if (j.created_at && (!last.oldest || j.created_at < last.oldest)) {
          last.oldest = j.created_at;
        }
      } else {
        out.push({
          ids: [j.job_id], kind: j.type, model, count: 1,
          oldest: j.created_at || "",
        });
      }
    }
    return out;
  }

  // Rough "time remaining" — projects the running job's full duration from its
  // observed sampling pace, then assumes each queued job takes about as long.
  // Returns "" when there's no running job with progress to estimate from.
  function pendingEta(running, queuedCount) {
    const r = running.find(
      (j) => j.progress && j.progress.includes("/") && (j.started_at || j.created_at));
    if (!r) return "";
    const [v, m] = r.progress.split("/").map(Number);
    if (!(v > 0 && m > 0)) return "";
    const elapsedSec = Math.max(
      1, (Date.now() - new Date(r.started_at || r.created_at).getTime()) / 1000);
    const perJobSec = (elapsedSec / v) * m;
    const totalSec = Math.max(0, perJobSec - elapsedSec) + queuedCount * perJobSec;
    const min = Math.round(totalSec / 60);
    return min < 1 ? "<1m" : `${min}m`;
  }

  function runningRow(j) {
    const cancel = cancelling.has(j.job_id);
    const isVid = j.type === "video";
    const model = j.model || j.type || "job";
    let pct = null, frac = "";
    if (j.progress && j.progress.includes("/")) {
      const [v, m] = j.progress.split("/").map(Number);
      if (m > 0) { pct = Math.min(100, Math.round((v / m) * 100)); frac = `${v}/${m}`; }
    }
    const meter = pct != null
      ? `<div class="pq-meter"><div class="fill" style="--pct:${pct}%"></div></div>`
      : `<div class="pq-meter indet"><div class="fill"></div></div>`;
    const step = cancel ? "Stopping" : frac ? `Step ${frac}` : "Sampling";
    return `<div class="pq-row running${cancel ? " cancelling" : ""}">` +
      `<div class="pq-stamp">${cancel ? "Cancelling" : "Running"}</div>` +
      `<div class="pq-name">` +
        `<span class="kindchip ${isVid ? "vid" : "img"}">${isVid ? "Vid" : "Img"}</span>` +
        `<span class="model" title="${esc(model)}">${esc(model)}</span></div>` +
      meter +
      `<div class="pq-eta"><span class="step">${esc(step)}</span>` +
        `${esc(elapsed(j.started_at || j.created_at))}</div>` +
      `<button class="pq-act" data-act="cancel-run" data-id="${esc(j.job_id)}"` +
        `${cancel ? " disabled" : ""} aria-label="Cancel running workflow"` +
        ` title="Cancel running workflow">${SVG_CLOSE}</button>` +
    `</div>`;
  }

  function queuedRow(g, idx) {
    const isVid = g.kind === "video";
    const wait = fmtWait(ageMin(g.oldest));
    const batch = g.count > 1
      ? `<span class="batchchip">× <span class="ct">${g.count}</span></span>` : "";
    const label = g.count > 1
      ? `Remove ${g.count} queued items` : "Remove from queue";
    return `<div class="pq-row queued">` +
      `<div class="pq-stamp">Queued</div>` +
      `<div class="pq-name">` +
        `<span class="kindchip ${isVid ? "vid" : "img"}">${isVid ? "Vid" : "Img"}</span>` +
        `<span class="model" title="${esc(g.model)}">${esc(g.model)}</span>${batch}</div>` +
      `<div class="pq-meter"></div>` +
      `<div class="pq-eta">${esc(wait)}</div>` +
      `<button class="pq-act" data-act="cancel-group" data-group="${idx}"` +
        ` aria-label="${esc(label)}" title="${esc(label)}">${SVG_CLOSE}</button>` +
    `</div>`;
  }

  function failedRow(j) {
    const isVid = j.type === "video";
    const model = j.model || j.type || "job";
    const err = (j.error || "Unknown error.").trim();
    return `<div class="pq-row failed">` +
      `<div class="pq-stamp">Failed</div>` +
      `<div class="pq-name">` +
        `<span class="kindchip ${isVid ? "vid" : "img"}">${isVid ? "Vid" : "Img"}</span>` +
        `<span class="model" title="${esc(model)}">${esc(model)}</span></div>` +
      `<div class="pq-err" title="${esc(err)}">${esc(failMsg(j))}</div>` +
      `<button class="pq-act" data-act="dismiss-fail" data-id="${esc(j.job_id)}"` +
        ` aria-label="Dismiss" title="Dismiss">${SVG_CLOSE}</button>` +
    `</div>`;
  }

  function buildPendingHtml(running, queued, failed) {
    if (!running.length && !queued.length && !failed.length) {
      pendGroups = [];
      return `<section class="pending"><header class="pending-hd"><div class="lhs">` +
        `<span class="eyebrow">Pending</span>` +
        `<span class="summary">No active generations</span></div></header>` +
        `<div class="pq-empty">Idle — generations queue here as you run workflows.</div>` +
        `</section>`;
    }
    const groups = groupQueue(queued);
    const VISIBLE = 3;
    const visible = pendShowAll ? groups : groups.slice(0, VISIBLE);
    pendGroups = visible;
    const hidden = groups.slice(visible.length);
    const hiddenCount = hidden.reduce((s, g) => s + g.count, 0);

    const eta = pendingEta(running, queued.length);
    const summary =
      `${queued.length} in queue<span class="sep">·</span>${running.length} running` +
      (failed.length
        ? `<span class="sep">·</span><span class="failnote">${failed.length} failed</span>`
        : "") +
      (eta ? `<span class="eta">~${esc(eta)} remaining</span>` : "");

    const longestWait = groups.reduce(
      (m, g) => (g.oldest && (!m || g.oldest < m) ? g.oldest : m), "");
    const foot = hidden.length
      ? `<footer class="pq-foot">` +
        `<button class="more" data-act="more">` +
        `${pendShowAll ? "Show fewer" : "+ " + hiddenCount + " more queued"}</button>` +
        `<span class="total">${queued.length} queued · longest wait ` +
        `${esc(fmtWait(ageMin(longestWait)))}</span></footer>`
      : "";

    return `<section class="pending${pendCollapsed ? " collapsed" : ""}">` +
      `<header class="pending-hd"><div class="lhs">` +
        `<span class="eyebrow">Pending</span>` +
        `<span class="summary">${summary}</span></div>` +
        `<div class="rhs">` +
          `<span class="status-stamp"><span class="dot"></span>` +
          `${running.length + queued.length} Active</span>` +
          `<button class="collapse" data-act="collapse">` +
          `${pendCollapsed ? "Show" : "Hide"}</button></div></header>` +
      `<div class="pending-body">` +
        failed.map(failedRow).join("") +
        running.map(runningRow).join("") +
        visible.map(queuedRow).join("") +
      `</div>${foot}</section>`;
  }

  // POST a cancel for one job; resolves true on success.
  async function cancelJob(jobId) {
    try {
      const r = await authedFetch(
        `/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
      return r.ok;
    } catch (_e) { return false; }
  }

  // Cancel the running job — hold its row in a "cancelling" state until the
  // worker actually interrupts ComfyUI and the job drops out of the feed.
  async function cancelRunning(jobId) {
    cancelling.add(jobId);
    renderPending();
    if (!(await cancelJob(jobId))) {
      cancelling.delete(jobId);
      showToast("Couldn't cancel — try again.");
      renderPending();
    }
  }

  // Cancel a whole queued group — every job collapsed into the row. Optimistic:
  // collapse the row out, then re-render once the requests land. Rollback is
  // per-id: a job whose cancel succeeded stays hidden even if a sibling failed.
  async function cancelGroup(group, rowEl) {
    group.ids.forEach((id) => cancelling.add(id));
    if (rowEl) rowEl.classList.add("removing");
    const results = await Promise.all(
      group.ids.map(async (id) => [id, await cancelJob(id)]));
    const failed = results.filter(([, ok]) => !ok).map(([id]) => id);
    if (failed.length) {
      failed.forEach((id) => cancelling.delete(id));
      showToast("Couldn't cancel — try again.");
    }
    setTimeout(renderPending, rowEl ? ROW_COLLAPSE_MS : 0);
  }

  // Dismiss a failed row — collapse it out, then remember the id so it stays
  // gone across re-renders and reloads.
  function dismissFailed(jobId, rowEl) {
    if (rowEl) rowEl.classList.add("removing");
    setTimeout(() => {
      failedDismissed[jobId] = Date.now();
      persistDismissed();
      renderPending();
    }, rowEl ? ROW_COLLAPSE_MS : 0);
  }

  function wirePending() {
    const collapseBtn = pending.querySelector('[data-act="collapse"]');
    if (collapseBtn) collapseBtn.onclick = () => { pendCollapsed = !pendCollapsed; renderPending(); };
    const moreBtn = pending.querySelector('[data-act="more"]');
    if (moreBtn) moreBtn.onclick = () => { pendShowAll = !pendShowAll; renderPending(); };
    pending.querySelectorAll('[data-act="cancel-run"]').forEach((btn) => {
      btn.onclick = () => cancelRunning(btn.dataset.id);
    });
    pending.querySelectorAll('[data-act="cancel-group"]').forEach((btn) => {
      btn.onclick = () => {
        const g = pendGroups[Number(btn.dataset.group)];
        if (g) cancelGroup(g, btn.closest(".pq-row"));
      };
    });
    pending.querySelectorAll('[data-act="dismiss-fail"]').forEach((btn) => {
      btn.onclick = () => dismissFailed(btn.dataset.id, btn.closest(".pq-row"));
    });
  }

  // A failure is "recent" if it started within RECENT_FAIL_MS. The worker
  // doesn't stamp completed_at on failure, so started_at is the freshest time
  // a failed job carries (image jobs run ~1-2 min, so it's close enough).
  const failRecent = (j) =>
    Date.now() - new Date(j.started_at || j.created_at).getTime() < RECENT_FAIL_MS;

  async function renderPending() {
    if (!(await ensureAuth())) return;
    let jobs;
    try {
      const r = await authedFetch(`/jobs?status=queued,running,failed&limit=100`);
      if (!r.ok) return;
      jobs = (await r.json()).jobs || [];
    } catch (_e) { return; }

    // Forget cancelling-ids that have left the feed (the cancel finished).
    const feedIds = new Set(jobs.map((j) => j.job_id));
    for (const id of [...cancelling]) if (!feedIds.has(id)) cancelling.delete(id);

    const running = jobs.filter((j) => j.status === "running");
    // Drop queued jobs the user just cancelled — gives instant feedback before
    // the server-side status flip lands in the next poll.
    const queued = jobs
      .filter((j) => j.status === "queued" && !cancelling.has(j.job_id))
      .sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));

    const failedAll = jobs.filter((j) => j.status === "failed");
    // Toast genuinely new failures. The toast is a *live* alert — for a
    // failure that happens while the viewer is open. The first poll baselines
    // failedSeen silently: pre-existing failures (however recent) surface as
    // panel rows, not toasts, and aren't re-toasted on reload. failedSeen is
    // reset to the current feed each poll, so it stays bounded.
    if (failedSeen !== null) {
      const fresh = failedAll.filter(
        (j) => !failedSeen.has(j.job_id) && failRecent(j));
      if (fresh.length === 1) {
        showToast(`Workflow failed — ${fresh[0].model || fresh[0].type || "job"}: ${failMsg(fresh[0])}`);
      } else if (fresh.length > 1) {
        showToast(`${fresh.length} workflows failed.`);
      }
    }
    failedSeen = new Set(failedAll.map((j) => j.job_id));
    // Drop dismissed entries older than the recency window — past that the
    // row can't render anyway, so the entry is dead weight.
    let pruned = false;
    for (const id of Object.keys(failedDismissed)) {
      if (Date.now() - failedDismissed[id] > RECENT_FAIL_MS) {
        delete failedDismissed[id]; pruned = true;
      }
    }
    if (pruned) persistDismissed();
    // Panel shows recent, undismissed failures (newest first).
    const failed = failedAll
      .filter((j) => failRecent(j) && !failedDismissed[j.job_id])
      .sort((a, b) => (b.started_at || b.created_at || "")
        .localeCompare(a.started_at || a.created_at || ""));

    pending.innerHTML = buildPendingHtml(running, queued, failed);
    wirePending();

    // A job leaving the queue (likely) finished — refresh the gallery.
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
            paramsSection(job.params) +
            promptsBlock(job.prompts) +
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
    scrim.querySelector(".del").onclick = () => doDelete(job.job_id);
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

  /* ---------- delete (no confirm — one click) ---------- */
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
