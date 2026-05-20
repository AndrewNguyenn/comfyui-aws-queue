// Outputs viewer — lists recent completed jobs and their generated outputs.
// Pulls the job list from GET /jobs?status=complete and renders each output
// via /view (a Cognito-authed 302 → presigned S3 GET).
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API = (cfg.apiUrl || "").replace(/\/$/, "");
  const grid = document.getElementById("output-grid");

  function authHeaders() {
    const h = {};
    const t = window.comfyAuth?.getIdToken();
    if (t) h["Authorization"] = t;
    return h;
  }

  // Resolve a /view 302 → presigned URL and append a thumbnail tile.
  async function renderThumb(key, job) {
    const filename = key.split("/").pop();
    const isVideo = /\.(mp4|webm|mov|gif|mkv)$/i.test(filename);
    const el = document.createElement("div");
    el.className = "thumb";
    try {
      // Browsers don't send headers on <img>/<video> loads, so resolve the
      // presigned URL ourselves (Authorization on the fetch) and use that.
      const resp = await fetch(`${API}/view?key=${encodeURIComponent(key)}`, {
        headers: authHeaders(),
        redirect: "manual",
      });
      const presigned = resp.headers.get("Location") || resp.url;
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
    try {
      const r = await fetch(`${API}/jobs?status=complete&limit=100`, {
        headers: authHeaders(),
      });
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

  load();

  function escape(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();
