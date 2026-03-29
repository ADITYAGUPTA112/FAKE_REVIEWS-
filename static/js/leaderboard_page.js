(function () {
  "use strict";

  const init = window.__LEADERBOARD_INIT__ || {};
  const state = {
    tab: "trusted",
    search: "",
    platform: "all",
    dateDays: "all",
    minScans: 1,
    sort: "trust_desc",
    fraudBand: "all",
    page: 1,
    pageSize: 12,
    maxProducts: 600,
  };
  const runtime = { req: 0, totalPages: 1 };

  const $ = (id) => document.getElementById(id);
  const el = {
    sumProducts: $("sum-products"),
    sumScans: $("sum-scans"),
    sumClean: $("sum-clean"),
    sumRisk: $("sum-risk"),
    tabTrusted: $("tab-trusted"),
    tabRisk: $("tab-risk"),
    resultMeta: $("result-meta"),
    search: $("search"),
    platform: $("platform"),
    dateDays: $("date-days"),
    minScans: $("min-scans"),
    sort: $("sort"),
    fraudBand: $("fraud-band"),
    reset: $("reset"),
    export: $("export"),
    top3: $("top3"),
    rows: $("rows"),
    pageMeta: $("page-meta"),
    pageNo: $("page-no"),
    prev: $("prev"),
    next: $("next"),
  };

  function fmtPct(v) {
    const n = Number(v);
    return Number.isFinite(n) ? `${n.toFixed(1)}%` : "-";
  }

  function adjustedRating(trust) {
    const n = Number(trust);
    if (!Number.isFinite(n)) return "-";
    return `${(1 + (Math.max(0, Math.min(100, n)) / 100) * 4).toFixed(1)} / 5`;
  }

  function statusClass(tier) {
    if (tier === "Clean") return "border-emerald-400/35 text-emerald-300 bg-emerald-500/15";
    if (tier === "Watchlist") return "border-amber-400/35 text-amber-300 bg-amber-500/15";
    return "border-rose-400/35 text-rose-300 bg-rose-500/15";
  }

  function confidenceClass(conf, scans) {
    if (Number(scans) < 3 || conf === "Low") return "border-rose-400/35 text-rose-300 bg-rose-500/15";
    if (conf === "Medium") return "border-amber-400/35 text-amber-300 bg-amber-500/15";
    return "border-emerald-400/35 text-emerald-300 bg-emerald-500/15";
  }

  function spark(values) {
    const arr = Array.isArray(values) ? values.map(Number).filter(Number.isFinite) : [];
    if (!arr.length) return '<span class="text-xs text-slate-500">no trend</span>';
    const w = 110, h = 30;
    const min = Math.min(...arr), max = Math.max(...arr), range = Math.max(max - min, 1);
    const pts = arr.map((v, i) => {
      const x = (i / Math.max(arr.length - 1, 1)) * (w - 8) + 4;
      const y = h - (((v - min) / range) * (h - 8) + 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const color = arr[arr.length - 1] >= arr[0] ? "#22c55e" : "#ff5d67";
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline points="${pts}" stroke="${color}" stroke-width="2" fill="none" stroke-linecap="round"/></svg>`;
  }

  function tabUi() {
    if (state.tab === "trusted") {
      el.tabTrusted.className = "tab-active border rounded px-3 py-2 text-sm font-semibold";
      el.tabRisk.className = "tab-idle border rounded px-3 py-2 text-sm font-semibold";
    } else {
      el.tabRisk.className = "tab-active border rounded px-3 py-2 text-sm font-semibold";
      el.tabTrusted.className = "tab-idle border rounded px-3 py-2 text-sm font-semibold";
    }
  }

  function ensurePlatforms(platforms) {
    const cur = el.platform.value || "all";
    const unique = Array.isArray(platforms) ? [...new Set(platforms.map((p) => String(p).toLowerCase()).filter(Boolean))].sort() : [];
    el.platform.innerHTML = '<option value="all">All Platforms</option>';
    unique.forEach((p) => {
      const o = document.createElement("option");
      o.value = p;
      o.textContent = p[0].toUpperCase() + p.slice(1);
      el.platform.appendChild(o);
    });
    if ([...el.platform.options].some((o) => o.value === cur)) el.platform.value = cur;
  }

  function renderSummary(summary) {
    const s = summary || {};
    el.sumProducts.textContent = Number(s.products || 0).toLocaleString();
    el.sumScans.textContent = Number(s.scans || 0).toLocaleString();
    el.sumClean.textContent = Number(s.clean || 0).toLocaleString();
    el.sumRisk.textContent = Number(s.high_risk || 0).toLocaleString();
  }

  function renderTop(rows) {
    const top = (Array.isArray(rows) ? rows : []).slice(0, 3);
    if (!top.length) {
      el.top3.innerHTML = '<div class="panel rounded p-3 text-slate-400">No entries</div>';
      return;
    }
    el.top3.innerHTML = top.map((r, i) => {
      const reasons = (r.reasons || []).slice(0, 2).map((x) => `<span class="chip rounded-full px-2 py-0.5 text-[11px] text-teal-200">${x}</span>`).join(" ");
      return `
      <article class="panel rounded p-3">
        <p class="text-xs text-slate-400 uppercase">Case ${i + 1}</p>
        <p class="text-[11px] text-slate-500 uppercase">Category</p>
        <p class="font-semibold mt-1">${r.product_id || "-"}</p>
        <p class="text-xs text-slate-400">${r.platform || "-"} - ${r.last_scan_label || "-"}</p>
        <div class="grid grid-cols-2 gap-2 mt-2 text-xs">
          <div class="rounded bg-[#050b14] border border-white/10 p-2">Trust <b class="text-teal-300">${fmtPct(r.trust_score)}</b></div>
          <div class="rounded bg-[#050b14] border border-white/10 p-2">Fraud <b class="text-rose-300">${fmtPct(r.fraud_score)}</b></div>
          <div class="rounded bg-[#050b14] border border-white/10 p-2">Adjusted <b>${adjustedRating(r.trust_score)}</b></div>
          <div class="rounded bg-[#050b14] border border-white/10 p-2">Evidence <b>${Number(r.scans_count || 0)}</b></div>
          <div class="rounded bg-[#050b14] border border-white/10 p-2">Genuine Share <b>${fmtPct(r.genuine_share)}</b></div>
          <div class="rounded bg-[#050b14] border border-white/10 p-2">Sample Confidence <b>${fmtPct(r.sample_confidence)}</b></div>
        </div>
        <div class="mt-2 flex flex-wrap gap-1">${reasons}</div>
      </article>`;
    }).join("");
  }

  function renderTable(payload) {
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    if (!rows.length) {
      el.rows.innerHTML = '<tr><td colspan="8" class="px-3 py-8 text-center text-slate-400">No rows found</td></tr>';
    } else {
      el.rows.innerHTML = rows.map((r) => {
        const rank = Number(r.rank || 0);
        const low = Number(r.scans_count || 0) < 3;
        const chips = (r.reasons || []).map((x) => `<span class="inline-block mr-1 mb-1 rounded-full chip px-2 py-0.5 text-[11px] text-teal-200">${x}</span>`).join("");
        return `<tr class="hover:bg-white/5">
          <td class="px-3 py-3 font-semibold ${rank <= 3 ? "text-teal-300" : "text-slate-300"}">#${rank || "-"}</td>
          <td class="px-3 py-3"><div class="font-mono font-semibold">${r.product_id || "-"}</div><div class="text-xs text-slate-400">${r.platform || "-"}</div></td>
          <td class="px-3 py-3"><span class="px-2 py-0.5 rounded border border-teal-300/35 bg-teal-400/10 text-teal-200">${fmtPct(r.trust_score)}</span></td>
          <td class="px-3 py-3 font-semibold text-rose-300">${fmtPct(r.fraud_score)}</td>
          <td class="px-3 py-3">${spark(r.trend)}</td>
          <td class="px-3 py-3"><div>${Number(r.scans_count || 0)} samples</div><div class="text-[11px] text-slate-400">Genuine: ${fmtPct(r.genuine_share)} | Sample conf: ${fmtPct(r.sample_confidence)}</div><span class="inline-block mt-1 rounded-full border px-2 py-0.5 text-[11px] ${confidenceClass(r.confidence, r.scans_count)}">${low ? "Low Confidence" : `${r.confidence} Confidence`}</span></td>
          <td class="px-3 py-3 text-xs">${chips || '<span class="text-slate-500">-</span>'}</td>
          <td class="px-3 py-3"><span class="inline-block rounded-full border px-2 py-0.5 text-xs ${statusClass(r.tier)}">${r.tier || "-"}</span></td>
        </tr>`;
      }).join("");
    }
    const total = Number(payload.total_filtered || 0);
    const start = total ? Number(payload.rank_offset || 0) + 1 : 0;
    const end = total ? Number(payload.rank_offset || 0) + rows.length : 0;
    const page = Number(payload.page || 1);
    const pages = Number(payload.total_pages || 1);
    runtime.totalPages = pages;
    el.pageMeta.textContent = `${start}-${end} of ${total}`;
    el.pageNo.textContent = `${page} / ${pages}`;
    el.prev.disabled = page <= 1;
    el.next.disabled = page >= pages;
    el.resultMeta.textContent = `${total} entries after filters`;
  }

  function csvOut(rows) {
    const headers = ["rank","category","platform","trust_score","fraud_score","sample_size","confidence","tier","reasons","last_scan"];
    const lines = [headers.join(",")];
    (rows || []).forEach((r) => {
      const vals = [r.rank || "", r.product_id || "", r.platform || "", r.trust_score ?? "", r.fraud_score ?? "", r.scans_count ?? "", r.confidence || "", r.tier || "", (r.reasons || []).join("|"), r.last_scan_iso || ""]
        .map((v) => `"${String(v).replace(/"/g, '""')}"`);
      lines.push(vals.join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `trustlens_global_category_leaderboard_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function query(over = {}) {
    const q = new URLSearchParams({
      tab: over.tab ?? state.tab,
      search: over.search ?? state.search,
      platform: over.platform ?? state.platform,
      date_days: over.dateDays ?? state.dateDays,
      min_scans: String(over.minScans ?? state.minScans),
      fraud_band: over.fraudBand ?? state.fraudBand,
      sort: over.sort ?? state.sort,
      page: String(over.page ?? state.page),
      page_size: String(over.pageSize ?? state.pageSize),
      max_products: String(over.maxProducts ?? state.maxProducts),
    });
    return q.toString();
  }

  async function fetchData(over = {}) {
    const req = ++runtime.req;
    el.resultMeta.textContent = "Loading...";
    const res = await fetch(`/api/leaderboard?${query(over)}`);
    if (!res.ok) throw new Error(`Leaderboard API ${res.status}`);
    const data = await res.json();
    if (req !== runtime.req) return null;
    return data;
  }

  function render(payload) {
    if (!payload) return;
    ensurePlatforms(payload.platforms || []);
    renderSummary(payload.summary || {});
    renderTop(payload.top3 || []);
    renderTable(payload);
  }

  async function refresh(over = {}) {
    try {
      const data = await fetchData(over);
      if (!data) return;
      state.page = Number(data.page || state.page);
      render(data);
    } catch (e) {
      el.resultMeta.textContent = "Failed to load leaderboard";
      console.error(e);
    }
  }

  async function doExport() {
    try {
      const data = await fetchData({ page: 1, pageSize: 500, maxProducts: 3000 });
      if (data && Array.isArray(data.rows)) csvOut(data.rows);
    } catch (e) { console.error(e); }
  }

  function bind() {
    let t = null;
    el.tabTrusted.addEventListener("click", async () => {
      state.tab = "trusted";
      state.page = 1;
      if (state.sort === "fraud_desc") { state.sort = "trust_desc"; el.sort.value = state.sort; }
      tabUi();
      await refresh();
    });
    el.tabRisk.addEventListener("click", async () => {
      state.tab = "risk";
      state.page = 1;
      if (state.sort === "trust_desc") { state.sort = "fraud_desc"; el.sort.value = state.sort; }
      tabUi();
      await refresh();
    });
    el.search.addEventListener("input", () => {
      if (t) clearTimeout(t);
      t = setTimeout(async () => { state.search = el.search.value || ""; state.page = 1; await refresh(); }, 220);
    });
    el.platform.addEventListener("change", async () => { state.platform = el.platform.value; state.page = 1; await refresh(); });
    el.dateDays.addEventListener("change", async () => { state.dateDays = el.dateDays.value; state.page = 1; await refresh(); });
    el.minScans.addEventListener("change", async () => { state.minScans = Number(el.minScans.value || 1); state.page = 1; await refresh(); });
    el.sort.addEventListener("change", async () => { state.sort = el.sort.value; state.page = 1; await refresh(); });
    el.fraudBand.addEventListener("change", async () => { state.fraudBand = el.fraudBand.value; state.page = 1; await refresh(); });
    el.reset.addEventListener("click", async () => {
      state.search = ""; state.platform = "all"; state.dateDays = "all"; state.minScans = 1;
      state.sort = state.tab === "risk" ? "fraud_desc" : "trust_desc";
      state.fraudBand = "all"; state.page = 1;
      el.search.value = ""; el.platform.value = "all"; el.dateDays.value = "all"; el.minScans.value = "1"; el.sort.value = state.sort; el.fraudBand.value = "all";
      await refresh();
    });
    el.prev.addEventListener("click", async () => { if (state.page > 1) { state.page -= 1; await refresh(); } });
    el.next.addEventListener("click", async () => { if (state.page < runtime.totalPages) { state.page += 1; await refresh(); } });
    el.export.addEventListener("click", doExport);
  }

  tabUi();
  render(init);
  bind();
  refresh();
})();
