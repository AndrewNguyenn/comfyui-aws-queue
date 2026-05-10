// Standalone /models page — paste CivitAI URL, browse catalog.
// Calls our dispatcher API directly (rather than ComfyUI's /prompt endpoint).
(function () {
  const cfg = window.COMFY_CONFIG || {};
  const API = (cfg.apiUrl || "").replace(/\/$/, "");

  const downloadForm = document.getElementById("download-form");
  const errorEl = document.getElementById("download-error");
  const listEl = document.getElementById("download-list");
  const catalogList = document.getElementById("model-list");
  const catalogSummary = document.getElementById("catalog-summary");
  const typeFilter = document.getElementById("type-filter");

  const activeDownloads = new Map(); // download_id → element

  function authHeaders() {
    const h = { "Content-Type": "application/json" };
    const t = window.comfyAuth?.getIdToken();
    if (t) h["Authorization"] = t;
    return h;
  }

  downloadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.textContent = "";
    const civitai_url = document.getElementById("civitai-url").value;
    const model_type = document.getElementById("model-type").value;

    try {
      const r = await fetch(`${API}/models/download`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ civitai_url, model_type }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const { download_id } = await r.json();
      addToList(download_id, civitai_url);
      pollDownload(download_id);
    } catch (err) {
      errorEl.textContent = err.message;
    }
  });

  function addToList(download_id, url) {
    const li = document.createElement("li");
    li.dataset.id = download_id;
    li.innerHTML = `
      <span class="dl-name">${escape(url)}</span>
      <progress value="0" max="100"></progress>
      <span class="dl-status">queued</span>
    `;
    listEl.prepend(li);
    activeDownloads.set(download_id, li);
  }

  async function pollDownload(download_id) {
    const li = activeDownloads.get(download_id);
    if (!li) return;
    try {
      const r = await fetch(`${API}/downloads/${download_id}`, {
        headers: authHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const prog = li.querySelector("progress");
      const status = li.querySelector(".dl-status");
      const name = li.querySelector(".dl-name");
      prog.value = data.percent || 0;
      status.textContent = `${data.status}${data.percent ? ` (${data.percent}%)` : ""}`;
      if (data.model_name) name.textContent = data.model_name;
      if (data.status === "complete") {
        status.classList.add("ok");
        await loadCatalog();
        return;
      }
      if (data.status === "failed") {
        status.classList.add("err");
        status.textContent = `failed: ${data.error || "unknown error"}`;
        return;
      }
    } catch (e) {
      console.error("poll failed", e);
    }
    setTimeout(() => pollDownload(download_id), 2000);
  }

  async function loadCatalog() {
    const t = typeFilter.value;
    const url = t ? `${API}/models?type=${encodeURIComponent(t)}` : `${API}/models`;
    catalogList.innerHTML = '<li class="loading">loading…</li>';
    try {
      const r = await fetch(url, { headers: authHeaders() });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const total = data.models.reduce((s, m) => s + (m.size_gb || 0), 0);
      catalogSummary.textContent = `${data.count} models, ${total.toFixed(1)} GB total`;
      catalogList.innerHTML = "";
      for (const m of data.models) {
        const li = document.createElement("li");
        li.innerHTML = `
          <strong>${escape(m.name)}</strong>
          <span class="meta">${escape(m.type)} · ${m.size_gb.toFixed(2)} GB${m.pinned ? ' · <span class="pin">pinned</span>' : ''}</span>
          <button class="del" data-name="${escape(m.name)}">remove from catalog</button>
        `;
        catalogList.appendChild(li);
      }
      catalogList.addEventListener("click", onCatalogClick, { once: true });
    } catch (e) {
      catalogList.innerHTML = `<li class="error">load failed: ${escape(e.message)}</li>`;
    }
  }

  async function onCatalogClick(e) {
    const btn = e.target.closest(".del");
    if (!btn) {
      catalogList.addEventListener("click", onCatalogClick, { once: true });
      return;
    }
    const name = btn.dataset.name;
    if (!confirm(`Remove ${name} from catalog? (S3 file is NOT deleted.)`)) {
      catalogList.addEventListener("click", onCatalogClick, { once: true });
      return;
    }
    try {
      const r = await fetch(`${API}/models/${encodeURIComponent(name)}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      await loadCatalog();
    } catch (err) {
      alert(`delete failed: ${err.message}`);
    }
  }

  typeFilter.addEventListener("change", loadCatalog);
  loadCatalog();

  function escape(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  }
})();
