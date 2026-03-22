const API_BASE = "http://127.0.0.1:5000";

const productMetaEl = document.getElementById("productMeta");
const trustScoreEl = document.getElementById("trustScore");
const ratingMetaEl = document.getElementById("ratingMeta");
const statusEl = document.getElementById("status");

let trendChart = null;

function parseContextFromUrl(url) {
  if (!url) return null;
  let u;
  try {
    u = new URL(url);
  } catch (e) {
    return null;
  }

  const host = u.hostname.toLowerCase();
  if (host.includes("amazon.")) {
    const m = url.match(/\/(?:dp|gp\/product)\/([A-Z0-9]{10})/i) || url.match(/\b([A-Z0-9]{10})\b/i);
    return m ? { platform: "amazon", productId: m[1].toUpperCase() } : null;
  }

  if (host.includes("flipkart.com")) {
    const pid = u.searchParams.get("pid");
    if (pid) return { platform: "flipkart", productId: pid };
    const m = url.match(/\/p\/([^/?#]+)/i);
    return m ? { platform: "flipkart", productId: m[1] } : null;
  }

  return null;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function renderTrendChart(history) {
  const labels = history.map((p) => p.month);
  const trust = history.map((p) => p.trust_score);
  const fraud = history.map((p) => p.fraud_score);

  const ctx = document.getElementById("trendChart").getContext("2d");
  if (trendChart) {
    trendChart.destroy();
  }

  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Trust Score",
          data: trust,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56,189,248,0.2)",
          tension: 0.25,
          spanGaps: true,
          pointRadius: 3
        },
        {
          label: "Fraud Risk",
          data: fraud,
          borderColor: "#f87171",
          backgroundColor: "rgba(248,113,113,0.2)",
          tension: 0.25,
          spanGaps: true,
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#cbd5e1", boxWidth: 10 }
        }
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", maxRotation: 60, minRotation: 40 },
          grid: { color: "rgba(148,163,184,0.15)" }
        },
        y: {
          min: 0,
          max: 100,
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.15)" }
        }
      }
    }
  });
}

async function loadPopup() {
  statusEl.textContent = "Reading active tab...";
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const context = parseContextFromUrl(tab?.url || "");

  if (!context) {
    statusEl.textContent = "Open an Amazon/Flipkart product page.";
    productMetaEl.textContent = "No supported product found.";
    trustScoreEl.textContent = "-";
    ratingMetaEl.textContent = "-";
    renderTrendChart([]);
    return;
  }

  const { platform, productId } = context;
  productMetaEl.textContent = `${platform.toUpperCase()} | ${productId}`;

  statusEl.textContent = "Fetching trust badge...";
  const badgeData = await fetchJson(
    `${API_BASE}/api/extension/trust-badge?platform=${encodeURIComponent(platform)}&product_id=${encodeURIComponent(productId)}&pages=10`
  );

  if (badgeData.error) {
    throw new Error(badgeData.error);
  }

  const trust = Number(badgeData.trust_score ?? 0);
  const fraud = Number(badgeData.fraud_risk ?? 0);
  trustScoreEl.textContent = `Trust ${trust.toFixed(1)}%`;

  if (typeof badgeData.authentic_rating === "number" && typeof badgeData.adjusted_rating === "number") {
    ratingMetaEl.textContent = `Authentic ${badgeData.authentic_rating.toFixed(1)} -> Adjusted ${badgeData.adjusted_rating.toFixed(1)} | Fraud ${fraud.toFixed(1)}%`;
  } else {
    ratingMetaEl.textContent = `Fraud risk ${fraud.toFixed(1)}%`;
  }

  statusEl.textContent = "Fetching 12-month history...";
  const trendData = await fetchJson(
    `${API_BASE}/api/trust-history?platform=${encodeURIComponent(platform)}&product_id=${encodeURIComponent(productId)}&months=12`
  );
  renderTrendChart(trendData.history || []);
  statusEl.textContent = "Updated";
}

loadPopup().catch((err) => {
  statusEl.textContent = `Error: ${err.message}`;
});
