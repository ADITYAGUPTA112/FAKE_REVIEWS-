(function () {
  const API_BASE = "http://127.0.0.1:5000";
  const BADGE_ID = "trustlens-authentic-badge";
  const BADGE_STYLE_ID = "trustlens-badge-style";

  function injectStyles() {
    if (document.getElementById(BADGE_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = BADGE_STYLE_ID;
    style.textContent = `
      #${BADGE_ID} {
        margin-top: 10px;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid #7dd3fc;
        background: #082f49;
        color: #e0f2fe;
        font-family: Arial, sans-serif;
        font-size: 13px;
        line-height: 1.35;
        max-width: 480px;
      }
      #${BADGE_ID} .trustlens-title {
        font-weight: 700;
        margin-bottom: 2px;
      }
      #${BADGE_ID} .trustlens-meta {
        opacity: 0.95;
      }
      #${BADGE_ID}.risk-high {
        border-color: #f87171;
        background: #450a0a;
        color: #fee2e2;
      }
      #${BADGE_ID}.risk-mid {
        border-color: #fbbf24;
        background: #451a03;
        color: #fef3c7;
      }
      #${BADGE_ID}.risk-low {
        border-color: #4ade80;
        background: #052e16;
        color: #dcfce7;
      }
    `;
    document.head.appendChild(style);
  }

  function getPlatform() {
    const host = location.hostname.toLowerCase();
    if (host.includes("flipkart.com")) return "flipkart";
    if (host.includes("amazon.")) return "amazon";
    return null;
  }

  function extractProductId(platform) {
    const url = location.href;
    if (platform === "amazon") {
      const m = url.match(/\/(?:dp|gp\/product)\/([A-Z0-9]{10})/i) || url.match(/\b([A-Z0-9]{10})\b/i);
      return m ? m[1].toUpperCase() : null;
    }
    if (platform === "flipkart") {
      const pid = new URL(location.href).searchParams.get("pid");
      if (pid) return pid;
      const m = url.match(/\/p\/([^/?#]+)/i);
      return m ? m[1] : null;
    }
    return null;
  }

  function findInsertionNode(platform) {
    if (platform === "amazon") {
      return (
        document.querySelector("#title") ||
        document.querySelector("#productTitle") ||
        document.querySelector("h1.a-size-large")
      );
    }
    if (platform === "flipkart") {
      return (
        document.querySelector("span.B_NuCI") ||
        document.querySelector("h1._6EBuvT span") ||
        document.querySelector("h1")
      );
    }
    return null;
  }

  function ensureBadge(targetNode) {
    let badge = document.getElementById(BADGE_ID);
    if (!badge) {
      badge = document.createElement("div");
      badge.id = BADGE_ID;
      targetNode.parentElement.insertBefore(badge, targetNode.nextSibling);
    }
    return badge;
  }

  function setBadgeState(badge, text, riskClass) {
    badge.classList.remove("risk-low", "risk-mid", "risk-high");
    if (riskClass) badge.classList.add(riskClass);
    badge.innerHTML = text;
  }

  async function fetchBadgeData(platform, productId) {
    const qs = new URLSearchParams({
      platform: platform,
      product_id: productId,
      pages: "10"
    });
    const response = await fetch(`${API_BASE}/api/extension/trust-badge?${qs.toString()}`);
    if (!response.ok) {
      throw new Error(`API ${response.status}`);
    }
    return response.json();
  }

  function riskClassFromFraud(fraudRisk) {
    if (fraudRisk >= 45) return "risk-high";
    if (fraudRisk >= 20) return "risk-mid";
    return "risk-low";
  }

  async function renderBadge() {
    const platform = getPlatform();
    if (!platform) return;
    const productId = extractProductId(platform);
    if (!productId) return;

    const target = findInsertionNode(platform);
    if (!target || !target.parentElement) return;

    injectStyles();
    const badge = ensureBadge(target);
    setBadgeState(
      badge,
      `<div class="trustlens-title">TrustLens</div><div class="trustlens-meta">Analyzing reviews...</div>`,
      null
    );

    try {
      const data = await fetchBadgeData(platform, productId);
      if (data.error) {
        setBadgeState(
          badge,
          `<div class="trustlens-title">TrustLens</div><div class="trustlens-meta">${data.error}</div>`,
          null
        );
        return;
      }

      const trust = Number(data.trust_score ?? 0);
      const fraud = Number(data.fraud_risk ?? 0);
      const ar = data.authentic_rating;
      const adj = data.adjusted_rating;

      const ratingText =
        typeof ar === "number" && typeof adj === "number"
          ? `Authentic Rating: <b>${ar.toFixed(1)}</b> -> Adjusted: <b>${adj.toFixed(1)}</b>`
          : `Trust Score: <b>${trust.toFixed(1)}</b>`;

      const html = `
        <div class="trustlens-title">TrustLens Authenticity Badge</div>
        <div class="trustlens-meta">${ratingText}</div>
        <div class="trustlens-meta">Fraud Risk: ${fraud.toFixed(1)}% | Trust: ${trust.toFixed(1)}%</div>
      `;
      setBadgeState(badge, html, riskClassFromFraud(fraud));
    } catch (err) {
      setBadgeState(
        badge,
        `<div class="trustlens-title">TrustLens</div><div class="trustlens-meta">Could not fetch trust score from API.</div>`,
        null
      );
    }
  }

  let rerenderTimer = null;
  function scheduleRender() {
    if (rerenderTimer) clearTimeout(rerenderTimer);
    rerenderTimer = setTimeout(renderBadge, 450);
  }

  renderBadge();
  const observer = new MutationObserver(scheduleRender);
  observer.observe(document.documentElement || document.body, {
    childList: true,
    subtree: true
  });
})();
