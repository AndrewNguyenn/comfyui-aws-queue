// ComfyUI · Outputs — Kenrokuen-language viewer.
// Live pending strip on top, then tabs / search / a paginated grid of
// generations. Backed by GET /jobs, /view, DELETE /jobs/{id}.
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API = (cfg.apiUrl || "").replace(/\/$/, "");

  const grid = document.getElementById("grid");
  const pager = document.getElementById("pager");
  const pagerTop = document.getElementById("pager-top");
  const pending = document.getElementById("pending");
  const tabsEl = document.getElementById("tabs");
  const searchEl = document.getElementById("search");
  const hdrCount = document.getElementById("hdr-count");

  const PER_PAGE = 75; // 5 across × 15 rows
  // Jobs are fetched lazily: a small first batch paints the gallery fast,
  // then larger batches in the background until the tail is reached. Keeps
  // first-paint cost flat as the job history grows.
  const INITIAL_FETCH = 150;  // enough to fill page 1 + a buffer for filters
  const BG_FETCH = 1000;
  const URL_CACHE_KEY = "viewer.urlcache";
  const URL_CACHE_TTL = 3300 * 1000; // ~55 min (presign lives 1 h)

  let allItems = []; // [{ key, job, isVideo }]
  let tab = "all";
  let query = "";
  let page = 0;
  let modalIdx = -1;
  // Workflow for the open modal's job, pre-fetched on open so the "Copy JSON"
  // button can write to the clipboard synchronously inside the click gesture
  // (execCommand — the only option on this plain-HTTP viewer — won't copy
  // outside an active user gesture, so it can't run after an awaited fetch).
  let modalWf = { jobId: null, wf: null, full: false };
  // Multi-select. `selected` holds image keys; `selectMode` keeps the
  // checkboxes always visible; `lastSelIdx` anchors shift-click range select.
  let selectMode = false;
  const selected = new Set();
  let lastSelIdx = -1;
  // Bumped on every grid render; queued presign calls for a superseded
  // render are skipped so a fast pager doesn't back the queue up with
  // requests for cards that are no longer on screen.
  let gridGen = 0;
  // Bumped on every loadJobs() — the background-loader loop checks it and
  // aborts when a fresh reload supersedes it (e.g. a pending job completed
  // and triggered a refetch).
  let loadGen = 0;

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
  // /view presign calls are concurrency-limited + retried. A grid page fires
  // one per card; bursting 75+ at once made some fail — and the failures were
  // silent (blank cards that only a page refresh recovered). Cap the in-flight
  // count and retry transient failures.
  const VIEW_MAX = 6;
  let _viewActive = 0;
  const _viewQ = [];
  function _viewRun(fn, stale) {
    return new Promise((resolve, reject) => {
      _viewQ.push({ fn, resolve, reject, stale });
      _viewPump();
    });
  }
  function _viewPump() {
    while (_viewActive < VIEW_MAX && _viewQ.length) {
      const { fn, resolve, reject, stale } = _viewQ.shift();
      if (stale && stale()) { reject(new Error("stale")); continue; }
      _viewActive++;
      fn().then(resolve, reject).finally(() => { _viewActive--; _viewPump(); });
    }
  }
  async function _fetchPresign(path) {
    let lastErr;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const r = await authedFetch(path);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const u = (await r.json()).url;
        if (!u) throw new Error("no url");
        return u;
      } catch (e) {
        lastErr = e;
        await new Promise((res) => setTimeout(res, 250 * (attempt + 1)));
      }
    }
    throw lastErr;
  }
  async function presignedUrl(key, download, stale) {
    // Display URLs are cached (stable URL → the browser image cache hits);
    // download URLs carry a content-disposition and are one-shot — skip cache.
    if (!download) {
      const hit = urlCache[key];
      if (hit && hit.exp > Date.now()) return hit.url;
    }
    const path = `/view?key=${encodeURIComponent(key)}&json=1` +
      (download ? "&download=1" : "");
    const url = await _viewRun(() => _fetchPresign(path), stale);
    if (!download) {
      urlCache[key] = { url, exp: Date.now() + URL_CACHE_TTL };
      try { localStorage.setItem(URL_CACHE_KEY, JSON.stringify(urlCache)); }
      catch (_e) { urlCache = {}; }
    }
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
    return `<div class="section-ttl prompt-ttl">${esc(label)}` +
      `<button class="prompt-copy" type="button" title="Copy ${esc(label)}">${SVG_COPY}</button>` +
      `</div>` +
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
  const SVG_CHECK =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l5 5 9-11"/></svg>';
  // Canonical copy glyph — front square + offset back sheet.
  const SVG_COPY =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="1.5"/><path d="M5 15V6.5A1.5 1.5 0 0 1 6.5 5H15"/></svg>';

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

  // Returns the raw job list for the page (no client filtering). offset is
  // over raw jobs so the caller's offset arithmetic and EOF check stay
  // aligned with what the API paginated.
  async function _fetchJobsPage(offset, limit) {
    const r = await authedFetch(
      `/jobs?status=complete&limit=${limit}&offset=${offset}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return (await r.json()).jobs || [];
  }
  function _appendJobsAsItems(jobs) {
    for (const job of jobs) {
      if (!(job.output_keys || []).length) continue;
      for (const key of job.output_keys) {
        allItems.push({ key, job, isVideo: isVideoKey(key) });
      }
    }
  }

  async function loadJobs() {
    const myGen = ++loadGen;
    if (!(await ensureAuth())) {
      if (myGen !== loadGen) return;
      grid.outerHTML = '<div class="empty"><div class="e-title">Not signed in</div>' +
        '<div class="e-body"><a href="/login.html">Log in</a> and reopen.</div></div>';
      return;
    }
    try {
      const initial = await _fetchJobsPage(0, INITIAL_FETCH);
      if (myGen !== loadGen) return;
      allItems = [];
      _appendJobsAsItems(initial);
      // Drop any selected keys that no longer exist after the reload, and
      // reset the shift-select anchor — filtered() indices have shifted.
      lastSelIdx = -1;
      if (selected.size) {
        const live = new Set(allItems.map((it) => it.key));
        for (const k of [...selected]) if (!live.has(k)) selected.delete(k);
        renderSelbar();
      }
      refreshCounts();
      renderGrid();
      // Tail of the history streams in below the fold. A short initial fetch
      // means filter tabs / search start sparse and fill as batches arrive.
      if (initial.length >= INITIAL_FETCH) _bgLoadRest(myGen, initial.length);
    } catch (e) {
      if (myGen !== loadGen) return;
      grid.innerHTML = `<div class="empty"><div class="e-title">Couldn't load</div>` +
        `<div class="e-body">${esc(e.message)}</div></div>`;
    }
  }

  // Page through the rest of the job history in the background. Stops when
  // an empty batch comes back (table exhausted) or a fresh loadJobs()
  // supersedes this run. After each batch, re-renders the grid only when
  // the user's current page slice actually gains items — appended items are
  // older than everything in allItems, so a slice that's already full of
  // newer items is byte-identical; just the pagers update.
  //
  // Cost note: offset/limit pagination re-queries the per-status pool of
  // size (offset + limit) per request, so total backend work grows with N
  // batches — the tradeoff for fast first paint.
  async function _bgLoadRest(myGen, startOffset) {
    let offset = startOffset;
    while (true) {
      let batch;
      try { batch = await _fetchJobsPage(offset, BG_FETCH); }
      catch (_e) { return; } // give up silently — the page already painted
      if (myGen !== loadGen) return;
      if (!batch.length) return;
      const sliceStart = page * PER_PAGE;
      const sliceEnd = sliceStart + PER_PAGE;
      const beforeFiltered = filtered().length;
      const beforePages = Math.max(1, Math.ceil(beforeFiltered / PER_PAGE));
      const beforeInSlice = Math.max(
        0, Math.min(beforeFiltered, sliceEnd) - sliceStart);
      _appendJobsAsItems(batch);
      refreshCounts();
      const afterFiltered = filtered().length;
      const afterPages = Math.max(1, Math.ceil(afterFiltered / PER_PAGE));
      const afterInSlice = Math.max(
        0, Math.min(afterFiltered, sliceEnd) - sliceStart);
      if (afterInSlice !== beforeInSlice) {
        // Current page slice grew — re-render the grid so the new cards land.
        renderGrid();
      } else if (afterPages !== beforePages) {
        // Slice unchanged, just more pages now — update the pagers in place.
        renderPager(pagerTop, afterPages, afterFiltered);
        renderPager(pager, afterPages, afterFiltered);
      }
      offset += batch.length;
    }
  }

  /* ---------- grid ---------- */
  function pageWindow(cur, total) {
    // Show every page number when there aren't many; past that, a wide window
    // (±5 around the current page) plus the first and last, with … gaps.
    if (total <= 16) return Array.from({ length: total }, (_, i) => i);
    const keep = [...new Set([
      0, total - 1,
      ...Array.from({ length: 11 }, (_, i) => cur - 5 + i),
    ])].filter((n) => n >= 0 && n < total).sort((a, b) => a - b);
    const out = [];
    keep.forEach((n, i) => {
      if (i > 0 && n - keep[i - 1] > 1) out.push(null);
      out.push(n);
    });
    return out;
  }

  function renderGrid() {
    gridGen++; // supersede any in-flight presign calls from the previous render
    const items = filtered();
    grid.classList.toggle("selectmode", selectMode);
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
      pagerTop.innerHTML = "";
      return;
    }
    grid.innerHTML = "";
    slice.forEach((item, i) => grid.appendChild(makeCard(item, page * PER_PAGE + i)));

    renderPager(pagerTop, pages, items.length);
    renderPager(pager, pages, items.length);
  }

  // The pager is rendered both above and below the grid — both call this.
  function renderPager(el, pages, total) {
    const nums = pageWindow(page, pages).map((p) =>
      p === null
        ? `<span class="pg-gap">…</span>`
        : `<button class="pg-num${p === page ? " active" : ""}" data-pg="${p}">${p + 1}</button>`
    ).join("");
    el.innerHTML =
      `<button class="pg-prev"${page <= 0 ? " disabled" : ""}>‹ Prev</button>` +
      `<span class="pg-nums">${nums}</span>` +
      `<button class="pg-next"${page >= pages - 1 ? " disabled" : ""}>Next ›</button>` +
      `<span class="pg-count">${total} item${total !== 1 ? "s" : ""}</span>`;
    const go = (p) => {
      const prev = page;
      page = Math.max(0, Math.min(pages - 1, p));
      renderGrid();
      // Coming back to page 0 picks up anything the pending-poll's refresh
      // was suppressing while we were away (see the page === 0 guard in
      // renderPending). renderGrid() above paints with current data first
      // so the user sees something immediately; loadJobs() re-renders when
      // the response lands.
      if (prev !== 0 && page === 0) loadJobs();
      window.scrollTo(0, 0);
    };
    el.querySelector(".pg-prev").onclick = () => go(page - 1);
    el.querySelector(".pg-next").onclick = () => go(page + 1);
    el.querySelectorAll(".pg-num").forEach((b) => {
      b.onclick = () => go(+b.dataset.pg);
    });
  }

  function loadCardMedia(el, item) {
    const { key, isVideo } = item;
    const filename = key.split("/").pop();
    const gen = gridGen; // the render this card belongs to
    presignedUrl(key, false, () => gen !== gridGen)
      .then((url) => {
        if (gen !== gridGen) return; // grid re-rendered — this card is gone
        const ph = el.querySelector(".thumb-wrap .ph");
        if (!ph) return;
        const media = isVideo
          ? `<video src="${url}#t=0.1" preload="metadata" muted></video>`
          : `<img src="${url}" loading="lazy" alt="${esc(filename)}" />`;
        ph.insertAdjacentHTML("beforebegin", media);
        ph.remove();
      })
      .catch(() => {
        if (gen !== gridGen) return; // superseded render — nothing to show
        // Presign failed even after retries — surface it instead of leaving a
        // dead "loading…" placeholder; one click retries this card.
        const ph = el.querySelector(".thumb-wrap .ph");
        if (!ph) return;
        ph.textContent = "failed — click to retry";
        ph.classList.add("ph-failed");
        ph.onclick = (e) => {
          e.stopPropagation();
          ph.textContent = "loading…";
          ph.classList.remove("ph-failed");
          ph.onclick = null;
          loadCardMedia(el, item);
        };
      });
  }
  function makeCard(item, filteredIdx) {
    const { key, job, isVideo } = item;
    const filename = key.split("/").pop();
    const el = document.createElement("div");
    el.className = "card" + (selected.has(key) ? " selected" : "");
    el.dataset.key = key;
    el.innerHTML =
      `<div class="thumb-wrap">` +
        `<div class="ph">loading…</div>` +
        `<div class="checkbox" title="Select">${SVG_CHECK}</div>` +
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
    el.querySelector(".checkbox").addEventListener("click", (e) => {
      e.stopPropagation();
      if (!selectMode) setSelectMode(true);
      toggleSelect(key, filteredIdx, e.shiftKey);
    });
    el.addEventListener("click", (e) => {
      if (e.target.closest(".thumb-actions") || e.target.closest(".checkbox")) return;
      if (selectMode) toggleSelect(key, filteredIdx, e.shiftKey);
      else openModal(filteredIdx);
    });
    loadCardMedia(el, item);
    return el;
  }

  /* ---------- multi-select ---------- */
  function setSelectMode(on) {
    selectMode = on;
    grid.classList.toggle("selectmode", on);
    const btn = document.getElementById("select-toggle");
    if (btn) btn.classList.toggle("active", on);
    if (!on) clearSelection();
  }
  function syncSelectionClasses() {
    grid.querySelectorAll(".card").forEach((el) => {
      el.classList.toggle("selected", selected.has(el.dataset.key));
    });
  }
  function clearSelection() {
    selected.clear();
    lastSelIdx = -1;
    syncSelectionClasses();
    renderSelbar();
  }
  function toggleSelect(key, idx, shift) {
    if (shift && lastSelIdx >= 0) {
      // Range-select across the filtered list from the last anchor.
      const items = filtered();
      const lo = Math.min(lastSelIdx, idx), hi = Math.max(lastSelIdx, idx);
      for (let i = lo; i <= hi && i < items.length; i++) selected.add(items[i].key);
    } else {
      if (selected.has(key)) selected.delete(key);
      else selected.add(key);
      lastSelIdx = idx;
    }
    syncSelectionClasses();
    renderSelbar();
  }
  function renderSelbar() {
    let bar = document.getElementById("selbar");
    if (!selected.size) { if (bar) bar.remove(); return; }
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "selbar";
      bar.className = "selbar";
      document.body.appendChild(bar);
    }
    const n = selected.size;
    bar.innerHTML =
      `<div class="sb-lhs"><span class="sb-ct">${n}</span>selected</div>` +
      `<div class="sb-rhs">` +
        `<button class="sb-clear" type="button">Clear</button>` +
        `<button class="sb-dl primary" type="button">Download ${n}</button>` +
      `</div>`;
    bar.querySelector(".sb-clear").onclick = clearSelection;
    bar.querySelector(".sb-dl").onclick = (e) => downloadSelected(e.currentTarget);
  }
  async function downloadSelected(btn) {
    const keys = [...selected];
    if (!keys.length) return;
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Preparing…";
    // Resolve every download URL first (concurrency-limited).
    const urls = (await Promise.all(
      keys.map((k) => presignedUrl(k, true).catch(() => null))
    )).filter(Boolean);
    btn.disabled = false;
    btn.textContent = orig;
    // Fire the downloads in one tight burst so the browser treats them as a
    // single batch — it shows one "allow multiple downloads" prompt and lets
    // them all through. Spacing the clicks with timers makes the browser see
    // separate requests and block all but the first.
    for (const url of urls) {
      const a = document.createElement("a");
      a.href = url;
      a.rel = "noopener";
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
    showToast(urls.length < keys.length
      ? `Downloading ${urls.length} of ${keys.length} — ${keys.length - urls.length} failed`
      : `Downloading ${urls.length} file${urls.length !== 1 ? "s" : ""}` +
        (urls.length > 1 ? " — allow multiple downloads if asked" : ""));
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

  // Bucket queued jobs by character (the server derives it from the prompt),
  // keeping the checkpoint on the row. When a prompt names no character the
  // server hands back an appearance `subject` hint instead (e.g. "black_hair")
  // — we group on that so unnamed figures still cluster ("black_hair figure")
  // rather than collapsing into one model row. Falls back to the model when
  // neither is present (scenery prompts, video jobs). Unlike a consecutive-run
  // collapse, this groups every job of the same (type, model, key) into ONE row
  // even when the queue interleaves them. `oldest` is an ISO created_at string
  // — ISO timestamps sort lexically, so a string compare finds the earliest.
  function groupQueue(queued) {
    const map = new Map();
    for (const j of queued) {
      const model = j.model || j.type || "job";
      const character = j.character || "";
      const subject = j.subject || "";
      const key = `${j.type}|${model}|${character || subject}`;
      let g = map.get(key);
      if (!g) {
        g = { ids: [], kind: j.type, model, character, subject, count: 0,
              oldest: j.created_at || "" };
        map.set(key, g);
      }
      g.ids.push(j.job_id);
      g.count += 1;
      if (j.created_at && (!g.oldest || j.created_at < g.oldest)) {
        g.oldest = j.created_at;
      }
    }
    // Largest batch first — what you most likely want to bulk-cancel — with the
    // oldest batch breaking ties.
    return [...map.values()].sort(
      (a, b) => b.count - a.count || (a.oldest || "").localeCompare(b.oldest || ""));
  }

  // Rough "time remaining" — projects the running job's full duration from its
  // observed sampling pace, then assumes each *same-type* queued job takes about
  // as long. Image vs video pace differ a lot, so the queued count must match
  // the running job's type: prefer the SQS per-type backlog (queuedByType),
  // else fall back to counting the fetched sample by type. Returns "" when
  // there's no running job with progress to estimate from.
  function pendingEta(running, queuedSample, queuedByType) {
    const r = running.find(
      (j) => j.progress && j.progress.includes("/") && (j.started_at || j.created_at));
    if (!r) return "";
    const [v, m] = r.progress.split("/").map(Number);
    if (!(v > 0 && m > 0)) return "";
    const queuedCount = queuedByType && queuedByType[r.type] != null
      ? queuedByType[r.type]
      : queuedSample.filter((j) => j.type === r.type).length;
    const elapsedSec = Math.max(
      1, (Date.now() - new Date(r.started_at || r.created_at).getTime()) / 1000);
    const perJobSec = (elapsedSec / v) * m;
    const totalSec = Math.max(0, perJobSec - elapsedSec) + queuedCount * perJobSec;
    const min = Math.round(totalSec / 60);
    if (min < 1) return "<1m";
    if (min < 90) return `${min}m`;  // render hours past 90m so a deep backlog
    const h = Math.floor(min / 60), mm = min % 60;  // doesn't print 4-digit min
    return mm ? `${h}h${mm}m` : `${h}h`;
  }

  function runningRow(j) {
    const cancel = cancelling.has(j.job_id);
    let pct = null, frac = "";
    if (j.progress && j.progress.includes("/")) {
      const [v, m] = j.progress.split("/").map(Number);
      if (m > 0) { pct = Math.min(100, Math.round((v / m) * 100)); frac = `${v}/${m}`; }
    }
    const meter = pct != null
      ? `<div class="pq-meter"><div class="fill" style="--pct:${pct}%"></div></div>`
      : `<div class="pq-meter indet"><div class="fill"></div></div>`;
    const step = cancel ? "Stopping" : frac ? `Step ${frac}` : "Sampling";
    // The GPU instance this job is generating on, when known.
    const instance = j.instance_type
      ? `<span class="pq-instance" title="GPU instance">${esc(j.instance_type)}</span>`
      : "";
    return `<div class="pq-row running${cancel ? " cancelling" : ""}">` +
      `<div class="pq-stamp">${cancel ? "Cancelling" : "Running"}</div>` +
      `<div class="pq-name">${nameCell(j.type, j.model || j.type || "job", j.character, j.subject)}${instance}</div>` +
      meter +
      `<div class="pq-eta"><span class="step">${esc(step)}</span>` +
        `${esc(elapsed(j.started_at || j.created_at))}</div>` +
      `<button class="pq-act" data-act="cancel-run" data-id="${esc(j.job_id)}"` +
        `${cancel ? " disabled" : ""} aria-label="Cancel running workflow"` +
        ` title="Cancel running workflow">${SVG_CLOSE}</button>` +
    `</div>`;
  }

  // Human headline for a group/job: the named character, else an unnamed-figure
  // appearance hint rendered as "<tag> figure" (e.g. "black_hair figure"), else
  // "" (caller falls back to the model).
  function subjectLabel(character, subject) {
    return character || (subject ? `${subject} figure` : "");
  }

  // The name cell: the character (or unnamed-figure hint) as the headline with
  // the checkpoint as a quiet trailing tag (`narberal_gamma · catponyDark…`).
  // Falls back to just the model when the prompt yielded neither.
  function nameCell(kind, model, character, subject) {
    const isVid = kind === "video";
    const named = subjectLabel(character, subject);
    const head = named || model || "job";
    const sub = named
      ? `<span class="submodel" title="${esc(model)}">· ${esc(model)}</span>` : "";
    return `<span class="kindchip ${isVid ? "vid" : "img"}">${isVid ? "Vid" : "Img"}</span>` +
      `<span class="model" title="${esc(head)}">${esc(head)}</span>${sub}`;
  }

  function queuedRow(g, idx) {
    const wait = fmtWait(ageMin(g.oldest));
    // g.shown is the true depth (scaled to the SQS backlog); g.count/g.ids are
    // the actual sampled rows the cancel button can act on. When the displayed
    // count exceeds what one cancel can remove, keep the label generic.
    const n = g.shown != null ? g.shown : g.count;
    const removable = g.ids ? g.ids.length : g.count;
    const batch = n > 1
      ? `<span class="batchchip">× <span class="ct">${n}</span></span>` : "";
    const named = subjectLabel(g.character, g.subject);
    const who = named ? `${named} · ${g.model}` : g.model;
    const label = n > 1
      ? (n > removable ? `Remove queued ${who} items` : `Remove ${n} queued ${who} items`)
      : `Remove ${who} from queue`;
    return `<div class="pq-row queued">` +
      `<div class="pq-stamp">Queued</div>` +
      `<div class="pq-name">${nameCell(g.kind, g.model, g.character, g.subject)}${batch}</div>` +
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

  function buildPendingHtml(running, queued, failed, queuedTotal, queuedByType) {
    if (!running.length && !queued.length && !failed.length && !queuedTotal) {
      pendGroups = [];
      return `<section class="pending"><header class="pending-hd"><div class="lhs">` +
        `<span class="eyebrow">Pending</span>` +
        `<span class="summary">No active generations</span></div></header>` +
        `<div class="pq-empty">Idle — generations queue here as you run workflows.</div>` +
        `</section>`;
    }
    const groups = groupQueue(queued);
    // The fetched sample caps at 500 rows, but queuedTotal is the true SQS
    // backlog. Scale each group's *displayed* count up to that total so the
    // chips reflect real queue depth — exact for a single-model batch, a
    // proportional estimate from the newest-sample when models are mixed.
    // g.count/g.ids stay the real sampled rows so the cancel button still acts
    // on actual job ids. (scale == 1 when the sample already covers the queue.)
    const sampleTotal = queued.length || 1;
    let assigned = 0;
    groups.forEach((g) => {
      g.shown = Math.round(g.count * queuedTotal / sampleTotal);
      assigned += g.shown;
    });
    if (groups.length) {  // park rounding drift on the largest group
      const big = groups.reduce((a, b) => (b.shown > a.shown ? b : a));
      big.shown += queuedTotal - assigned;
    }
    const VISIBLE = 3;
    const visible = pendShowAll ? groups : groups.slice(0, VISIBLE);
    pendGroups = visible;
    // hiddenCount is the true remainder beyond the visible chips (their scaled
    // counts), so chips + "+N more" reconcile to queuedTotal.
    const visibleShown = visible.reduce((s, g) => s + g.shown, 0);
    const hiddenCount = Math.max(0, queuedTotal - visibleShown);

    const eta = pendingEta(running, queued, queuedByType);
    const summary =
      `${queuedTotal} in queue<span class="sep">·</span>${running.length} running` +
      (failed.length
        ? `<span class="sep">·</span><span class="failnote">${failed.length} failed</span>`
        : "") +
      (eta ? `<span class="eta">~${esc(eta)} remaining</span>` : "");

    const longestWait = groups.reduce(
      (m, g) => (g.oldest && (!m || g.oldest < m) ? g.oldest : m), "");
    // The expand/collapse toggle only makes sense when there are extra sample
    // groups to reveal. A remainder that ISN'T a collapsible group — the sample
    // truncated at 500, or a small SQS overcount — gets no button (clicking
    // would reveal nothing); the foot's queuedTotal still tells the true count.
    const hasCollapsible = groups.length > VISIBLE;
    const moreBtn = hasCollapsible
      ? `<button class="more" data-act="more">` +
        `${pendShowAll ? "Show fewer" : "+ " + hiddenCount + " more queued"}</button>`
      : "";
    // Show the foot whenever there's a queue (it carries the longest-wait); the
    // toggle/“+N more” only appears when there are collapsible groups.
    const foot = groups.length
      ? `<footer class="pq-foot">${moreBtn}` +
        `<span class="total">${queuedTotal} queued · longest wait ` +
        `${esc(fmtWait(ageMin(longestWait)))}</span></footer>`
      : "";

    return `<section class="pending${pendCollapsed ? " collapsed" : ""}">` +
      `<header class="pending-hd"><div class="lhs">` +
        `<span class="eyebrow">Pending</span>` +
        `<span class="summary">${summary}</span></div>` +
        `<div class="rhs">` +
          `<span class="status-stamp"><span class="dot"></span>` +
          `${running.length + queuedTotal} Active</span>` +
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

  // A small yes/no confirm overlay (reuses the .scrim backdrop). Resolves true
  // on confirm, false on Keep / Escape / backdrop click. Enter confirms.
  function confirmDialog({ title, sub, confirmLabel, cancelLabel }) {
    return new Promise((resolve) => {
      // Its OWN backdrop class — NOT the modal's `.scrim` — so the gallery
      // modal's document.querySelector(".scrim") can never grab this overlay.
      const scrim = document.createElement("div");
      scrim.className = "confirm-scrim";
      scrim.innerHTML =
        `<div class="confirm-card" role="alertdialog" aria-modal="true">` +
          `<div class="confirm-title">${esc(title)}</div>` +
          (sub ? `<div class="confirm-sub">${esc(sub)}</div>` : "") +
          `<div class="confirm-actions">` +
            `<button class="confirm-keep" data-act="keep">${esc(cancelLabel || "Keep")}</button>` +
            `<button class="confirm-go" data-act="go">${esc(confirmLabel || "Confirm")}</button>` +
          `</div></div>`;
      const done = (val) => {
        document.removeEventListener("keydown", onKey, true);
        scrim.remove();
        resolve(val);
      };
      // Capture phase + stopPropagation so a modal confirm swallows the key —
      // Escape here must NOT also reach the global handler (which closes the
      // image modal / exits select mode).
      const onKey = (e) => {
        if (e.key !== "Escape" && e.key !== "Enter") return;
        e.preventDefault();
        e.stopPropagation();
        done(e.key === "Enter");
      };
      scrim.addEventListener("click", (e) => { if (e.target === scrim) done(false); });
      scrim.querySelector('[data-act="keep"]').onclick = () => done(false);
      scrim.querySelector('[data-act="go"]').onclick = () => done(true);
      document.addEventListener("keydown", onKey, true);
      document.body.appendChild(scrim);
      scrim.querySelector('[data-act="go"]').focus();
    });
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
      btn.onclick = async () => {
        const g = pendGroups[Number(btn.dataset.group)];
        if (!g) return;
        // Confirm before nuking a whole character's batch — one click could
        // otherwise cancel dozens of jobs that can't be un-cancelled (only
        // re-queued). n is the displayed (SQS-scaled) count; `removable` is the
        // sampled ids the cancel can actually act on — if the queue ran past
        // the 500-row sample, say what the cancel really removes.
        const n = g.shown != null ? g.shown : g.count;
        const removable = g.ids ? g.ids.length : g.count;
        const named = subjectLabel(g.character, g.subject);
        const who = named ? `${named} · ${g.model}` : g.model;
        const sub = n > removable
          ? `${who}  —  cancels the ${removable} loaded so far (of ~${n})`
          : who;
        const ok = await confirmDialog({
          title: `Cancel ${n} queued ${n === 1 ? "job" : "jobs"}?`,
          sub,
          confirmLabel: "Cancel all",
          cancelLabel: "Keep",
        });
        if (ok) cancelGroup(g, btn.closest(".pq-row"));
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
    let jobs, queuedTotal, queuedByType;
    try {
      // The "in queue" count comes from the live SQS backlog (/backlog) — exact
      // and uncapped. Counting fetched job records instead clamps at the page
      // limit, and a single combined /jobs query let a big queued batch (newest
      // created_at) crowd the running/failed rows out of the window ("0
      // running"). So: /backlog gives the count; per-status /jobs queries give
      // the rows — running/failed in full (only a handful exist), queued as a
      // display sample (500) for the grouped rows. The four calls are a
      // non-atomic snapshot (a job mid-flip can be briefly absent from all of
      // them) — fine; the gallery refresh below is best-effort and self-corrects.
      const [qDepth, qd, rn, fl] = await Promise.all([
        authedFetch(`/backlog`),
        authedFetch(`/jobs?status=queued&limit=500`),
        authedFetch(`/jobs?status=running&limit=50`),
        authedFetch(`/jobs?status=failed&limit=50`),
      ]);
      // Degrade per status: a hiccup on the low-value running/failed list
      // shouldn't blank the whole strip. But if the queued rows fail, skip the
      // tick — the gallery-refresh diff below would otherwise read "every queued
      // job left" and fire a spurious reload.
      if (!qd.ok) return;
      const bucket = async (r) => (r.ok ? ((await r.json()).jobs || []) : []);
      const [qj, rj, fj] = await Promise.all([bucket(qd), bucket(rn), bucket(fl)]);
      jobs = [...rj, ...qj, ...fj];
      // SQS backlog: total drives the count, per-type drives the ETA. null if
      // /backlog failed → both fall back to the fetched sample downstream.
      const depth = qDepth.ok ? await qDepth.json() : null;
      queuedTotal = depth ? (depth.queued ?? null) : null;
      queuedByType = depth
        ? { image: depth.image?.queued, video: depth.video?.queued } : null;
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

    // Authoritative queue count from SQS; if /backlog failed, fall back to the
    // fetched sample. Never show fewer than we actually fetched (guards the
    // case where SQS lags behind the freshly-written queued records).
    const qTotal = queuedTotal == null
      ? queued.length : Math.max(queuedTotal, queued.length);
    pending.innerHTML = buildPendingHtml(running, queued, failed, qTotal, queuedByType);
    wirePending();

    // A job leaving the queue (likely) finished — refresh the gallery. Skip
    // when the user is past page 0: the new job lands at the top of page 0,
    // and loadJobs() resets allItems to the initial-fetch window — too small
    // to cover a late page, so we'd bump the user back. A later return to
    // page 0 (or the next completion while there) picks the new job up.
    const cur = new Set(jobs.map((j) => j.job_id));
    if (page === 0 && lastPendingIds &&
        [...lastPendingIds].some((id) => !cur.has(id))) loadJobs();
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
            // Filled in by the /jobs/{id} pre-fetch below — the /jobs list
            // response is lite and carries no prompts/params.
            `<div class="m-detail"></div>` +
          `</div>` +
          `<div class="info-foot">` +
            `<button class="dl">Download</button>` +
            `<button class="copywf">Copy JSON</button>` +
            `<button class="danger del">Delete</button>` +
          `</div>` +
        `</div>` +
      `</div>`;
    scrim.querySelector(".close").onclick = closeModal;
    scrim.querySelector(".nav.prev").onclick = () => { if (modalIdx > 0) { modalIdx--; drawModal(); } };
    scrim.querySelector(".nav.next").onclick = () => { if (modalIdx < items.length - 1) { modalIdx++; drawModal(); } };
    scrim.querySelector(".del").onclick = () => doDelete(job.job_id);
    const copyBtn = scrim.querySelector(".copywf");
    copyBtn.onclick = () => doCopyWorkflow(job, copyBtn);
    // Pre-fetch the workflow so the copy is gesture-synchronous (see modalWf).
    modalWf = { jobId: job.job_id, wf: null, full: false };
    authedFetch(`/jobs/${encodeURIComponent(job.job_id)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && modalWf.jobId === job.job_id) {
          // Prefer the full editor workflow; older jobs only have the API prompt.
          modalWf.wf = d.workflow_ui || d.workflow || null;
          modalWf.full = !!d.workflow_ui;
          // Fill the params/prompts sections — the /jobs list is lite.
          const det = scrim.querySelector(".m-detail");
          if (det) {
            det.innerHTML = paramsSection(d.params) + promptsBlock(d.prompts);
            det.querySelectorAll(".prompt-copy").forEach((btn) => {
              btn.onclick = (e) => {
                e.stopPropagation();
                // The prompt body is the element right after the title row.
                const body = btn.closest(".prompt-ttl") &&
                  btn.closest(".prompt-ttl").nextElementSibling;
                const text = body ? body.textContent : "";
                if (text && copyText(text)) {
                  btn.innerHTML = SVG_CHECK;
                  btn.classList.add("copied");
                  setTimeout(() => {
                    if (!btn.isConnected) return;
                    btn.innerHTML = SVG_COPY;
                    btn.classList.remove("copied");
                  }, 1400);
                  showToast("Prompt copied");
                } else {
                  showToast("Couldn't copy the prompt");
                }
              };
            });
          }
        }
      })
      .catch(() => {});
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
      allItems
        .filter((it) => it.job.job_id === jobId)
        .forEach((it) => selected.delete(it.key));
      allItems = allItems.filter((it) => it.job.job_id !== jobId);
      renderSelbar();
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

  /* ---------- copy workflow (seed-randomized) ---------- */
  // < 2^50 — within rgthree's Seed-node cap (1125899906842624).
  const randSeed = () => Math.floor(Math.random() * 0x4000000000000);
  // Replace every seed so the copy is seed-canonical: copying any of N
  // identical-but-different-seed runs yields the same workflow, with a fresh
  // random seed each click. Handles both shapes the button may receive:
  //   - full editor workflow: { nodes: [ { type, widgets_values:[...] } ] } —
  //     the seed is a positional widget value (first integer on a sampler /
  //     seed / noise node).
  //   - API prompt: { nodeId: { inputs:{ seed|noise_seed } } } — a named input
  //     (skip [nodeId, idx] link arrays — those aren't literal seeds).
  function randomizeSeeds(wf) {
    let n = 0;
    if (Array.isArray(wf.nodes)) {
      // The UI workflow carries no widget names — match known seed-bearing
      // node types and take the first integer widget (the seed slot for all
      // of them). Anchored/exact rather than a loose substring so a non-seed
      // integer on an unrelated node isn't randomized by mistake.
      const isSeedNode = (t) =>
        /^(KSampler|KSamplerAdvanced|SamplerCustom|SamplerCustomAdvanced|RandomNoise)$/i.test(t) ||
        /seed/i.test(t);
      for (const node of wf.nodes) {
        const wv = node && node.widgets_values;
        if (!Array.isArray(wv) || !isSeedNode(node.type || "")) continue;
        for (let i = 0; i < wv.length; i++) {
          if (typeof wv[i] === "number" && Number.isInteger(wv[i]) && wv[i] >= 0) {
            wv[i] = randSeed(); // first non-negative integer widget is the seed
            n++;
            break;
          }
        }
      }
      return n;
    }
    for (const node of Object.values(wf)) {
      const inp = node && node.inputs;
      if (!inp || typeof inp !== "object") continue;
      for (const k of ["seed", "noise_seed"]) {
        if (typeof inp[k] === "number") {
          inp[k] = randSeed();
          n++;
        }
      }
    }
    return n;
  }
  // Synchronous so it runs inside the click gesture. The async Clipboard API
  // needs a secure context, which this plain-HTTP viewer is not — execCommand
  // is the only option and it must not be reached via an await.
  function copyText(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.top = "0";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (_e) { ok = false; }
    ta.remove();
    return ok;
  }
  function doCopyWorkflow(job, btn) {
    const wf = modalWf.jobId === job.job_id ? modalWf.wf : null;
    if (!wf || typeof wf !== "object" || !Object.keys(wf).length) {
      // Either the pre-fetch is still in flight, or the job has no workflow.
      showToast("Workflow still loading — try again in a moment");
      return;
    }
    // Clone so the seed randomization never mutates the cached workflow.
    const copy = JSON.parse(JSON.stringify(wf));
    const n = randomizeSeeds(copy);
    if (!copyText(JSON.stringify(copy, null, 2))) {
      showToast("Couldn't write to clipboard");
      return;
    }
    btn.textContent = "Copied ✓";
    setTimeout(() => { if (btn.isConnected) btn.textContent = "Copy JSON"; }, 2000);
    const seeds = n
      ? `${n} seed${n > 1 ? "s" : ""} randomized`
      : "no seed found to randomize";
    showToast(modalWf.full
      ? `Full workflow copied — ${seeds}`
      : `Copied API prompt (full graph unavailable for this run) — ${seeds}`);
  }

  /* ---------- wiring ---------- */
  tabsEl.addEventListener("click", (e) => {
    const t = e.target.closest(".tab");
    if (!t) return;
    tab = t.dataset.tab;
    page = 0;
    lastSelIdx = -1; // filtered() changed — the shift-select anchor is stale
    tabsEl.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    renderGrid();
  });
  searchEl.addEventListener("input", () => {
    query = searchEl.value.trim();
    page = 0;
    lastSelIdx = -1; // filtered() changed — the shift-select anchor is stale
    renderGrid();
  });
  document.getElementById("select-toggle").addEventListener("click", () => {
    setSelectMode(!selectMode);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && selectMode) { setSelectMode(false); return; }
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
