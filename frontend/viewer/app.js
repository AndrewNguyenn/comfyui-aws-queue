// Outputs viewer — lists recent completed jobs and their generated outputs.
// Pulls from /jobs and renders each output via /view (presigned redirect).
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

  async function load() {
    grid.innerHTML = '<p class="loading">loading…</p>';
    try {
      // Note: a /jobs collection endpoint is not implemented in v1 — viewer
      // currently relies on the user knowing job IDs. v2 should add a recent-
      // jobs endpoint backed by the status-index GSI.
      grid.innerHTML = `
        <p class="hint">v1 limitation: viewer needs a /jobs collection endpoint
        to list recent outputs. Until then, fetch a specific job by ID:</p>
        <form id="job-form">
          <input type="text" id="job-id" placeholder="job_id (uuid)" required />
          <button type="submit">Load</button>
        </form>
      `;
      document.getElementById("job-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const id = document.getElementById("job-id").value.trim();
        await loadJob(id);
      });
    } catch (e) {
      grid.innerHTML = `<p class="error">load failed: ${e.message}</p>`;
    }
  }

  async function loadJob(id) {
    grid.innerHTML = '<p class="loading">loading job…</p>';
    try {
      const r = await fetch(`${API}/jobs/${encodeURIComponent(id)}`, {
        headers: authHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const job = await r.json();
      if (!job.output_keys?.length) {
        grid.innerHTML = `<p>no outputs (status: ${job.status})</p>`;
        return;
      }
      grid.innerHTML = "";
      for (const key of job.output_keys) {
        const filename = key.split("/").pop();
        const isVideo = /\.(mp4|webm|mov|gif|mkv)$/i.test(filename);
        const el = document.createElement("div");
        el.className = "thumb";
        // Resolve the presigned URL ourselves so we can attach Authorization
        // (browsers don't send headers on <img>/<video> requests).
        // /view returns 302 → presigned. We follow manually via fetch.
        const viewApiUrl = `${API}/view?key=${encodeURIComponent(key)}`;
        try {
          const resp = await fetch(viewApiUrl, {
            headers: authHeaders(),
            redirect: "manual",
          });
          // Cognito-auth GET → API GW returns the 302; browser sets opaque-redirect status.
          // Read Location header from the response.
          const presigned = resp.headers.get("Location") || resp.url;
          if (isVideo) {
            el.innerHTML = `
              <video controls preload="metadata" src="${presigned}#t=0.1"></video>
              <p>${escape(filename)}</p>
            `;
          } else {
            el.innerHTML = `
              <a href="${presigned}" target="_blank" rel="noopener">
                <img src="${presigned}" alt="${escape(filename)}" loading="lazy" />
              </a>
              <p>${escape(filename)}</p>
            `;
          }
        } catch (err) {
          el.innerHTML = `<p class="error">${escape(filename)}: ${err.message}</p>`;
        }
        grid.appendChild(el);
      }
    } catch (e) {
      grid.innerHTML = `<p class="error">job load failed: ${e.message}</p>`;
    }
  }

  load();

  function escape(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  }
})();
