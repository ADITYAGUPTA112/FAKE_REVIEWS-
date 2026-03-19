import torch
import numpy as np
import pandas as pd
import os
import glob
import re
import time
import random
import joblib
import scipy.sparse
import requests
import dateparser

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import RobertaForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
SCRAPINGDOG_KEY   = os.getenv("SCRAPINGDOG_KEY", "")   # optional — free at scrapingdog.com
SERP_API_KEY      = os.getenv("SERP_API_KEY", "262ad8b51b449946485141e9ee2521a8d0120bd6b0ba609c667ed3a3d56d0495")
RAPIDAPI_KEY      = os.getenv("RAPIDAPI_KEY", "")       # optional — free at rapidapi.com

DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENUINE_THRESHOLD = 60.0
MAX_REVIEWS       = 500   # safe limit for direct scraping

# Browser headers — rotate to avoid blocks
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

def _headers():
    return {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Referer":         "https://www.amazon.in/",
    }

# =============================================================================
# LOCATE MODELS
# =============================================================================
_CHECKPOINT_BASE = "./roberta_checkpoints"
_checkpoint_dirs = glob.glob(os.path.join(_CHECKPOINT_BASE, "checkpoint-*"))
MODEL_PATH = (
    sorted(_checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))[-1]
    if _checkpoint_dirs else _CHECKPOINT_BASE
)
print(f"[INFO] RoBERTa checkpoint : {MODEL_PATH}")

for _candidate in ["./models/ensemble", "./model/ensemble", "./model"]:
    if os.path.exists(_candidate):
        ENSEMBLE_PATH = _candidate
        break
else:
    ENSEMBLE_PATH = "./models/ensemble"
print(f"[INFO] Ensemble path      : {ENSEMBLE_PATH}")

# =============================================================================
# LOAD MODELS
# =============================================================================
MODELS_LOADED = False
roberta_model = tokenizer = None
rf = lr_base = xgb = meta_learner = tfidf = None


def _pkl(new_name, old_name=None):
    p = os.path.join(ENSEMBLE_PATH, new_name)
    if os.path.exists(p):
        return joblib.load(p)
    if old_name:
        p2 = os.path.join(ENSEMBLE_PATH, old_name)
        if os.path.exists(p2):
            return joblib.load(p2)
    raise FileNotFoundError(f"{new_name} not found in {ENSEMBLE_PATH}")


try:
    roberta_model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    roberta_model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    rf           = _pkl("rf_classifier.pkl",   "random_forest.pkl")
    lr_base      = _pkl("lr_classifier.pkl",   "logistic_regression.pkl")
    xgb          = _pkl("xgb_classifier.pkl",  "xgboost.pkl")
    meta_learner = _pkl("meta_learner.pkl",     "meta_learner.pkl")
    tfidf        = _pkl("tfidf_vectorizer.pkl", "tfidf.pkl")

    MODELS_LOADED = True
    print("[INFO] All models loaded successfully.")

except Exception as e:
    print(f"[WARN] Models not fully loaded: {e}")

# =============================================================================
# LIME
# =============================================================================
explainer = LimeTextExplainer(class_names=["Fake", "Genuine"])

# =============================================================================
# TEXT CLEANING
# =============================================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",               " ", text)
    text = re.sub(r"https?://\S+|www\.\S+",  " ", text)
    text = re.sub(r"[^\w\s]",               " ", text)
    text = re.sub(r"\s+",                   " ", text).strip()
    return text

# =============================================================================
# PREDICTION HELPERS
# =============================================================================
def _get_genuine_proba(model, X):
    col = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(X)[:, col]


def get_prediction_probs(texts):
    if not MODELS_LOADED:
        return np.array([[0.5, 0.5]] * len(texts))
    cleaned    = [clean_text(t) for t in texts]
    lengths    = np.array([[len(t.split())] for t in cleaned])
    X_tfidf    = tfidf.transform(cleaned)
    X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(lengths)])
    p_rf  = _get_genuine_proba(rf,      X_combined)
    p_lr  = _get_genuine_proba(lr_base, X_combined)
    p_xgb = _get_genuine_proba(xgb,     X_combined)
    all_rob = []
    for i in range(0, len(cleaned), 32):
        batch  = cleaned[i: i + 32]
        inputs = tokenizer(batch, truncation=True, padding=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = roberta_model(**inputs).logits
        all_rob.extend(torch.softmax(logits, dim=1).cpu().numpy()[:, 1])
    p_roberta = np.array(all_rob)
    return meta_learner.predict_proba(np.column_stack([p_rf, p_lr, p_xgb, p_roberta]))


def predict_review(text):
    if not MODELS_LOADED:
        return "Unknown", 0.0, 0.0
    p = get_prediction_probs([text])[0]
    fp, gp = p[0]*100, p[1]*100
    return ("Genuine", round(gp,2), round(fp,2)) if gp >= GENUINE_THRESHOLD else ("Fake", round(fp,2), round(fp,2))


def predict_reviews_batch(texts):
    if not texts or not MODELS_LOADED:
        return [], [], []
    all_probs = []
    for s in range(0, len(texts), 256):
        all_probs.append(get_prediction_probs(texts[s:s+256]))
        print(f"[INFO]   Predicted {min(s+256, len(texts))}/{len(texts)} …")
    labels, confs, fps = [], [], []
    for p in np.vstack(all_probs):
        fp, gp = p[0]*100, p[1]*100
        if gp >= GENUINE_THRESHOLD:
            labels.append("Genuine"); confs.append(round(gp,2))
        else:
            labels.append("Fake");    confs.append(round(fp,2))
        fps.append(round(fp,2))
    return labels, confs, fps


def explain_review(text, label):
    if not MODELS_LOADED:
        return []
    try:
        exp = explainer.explain_instance(text, get_prediction_probs, labels=(0, 1),
                                          num_features=5, num_samples=50)
        idx = 0 if label == "Fake" else 1
        return [{"word": w, "weight": float(wt)} for w, wt in exp.as_list(label=idx)]
    except Exception as e:
        print(f"[WARN] LIME: {e}")
        return []

# =============================================================================
# METHOD 1 — Direct Amazon scraping (FREE, no API key needed)
# =============================================================================
def _scrape_amazon_direct(asin: str,
                           domain: str = "amazon.in",
                           max_reviews: int = MAX_REVIEWS) -> list:
    """
    Directly scrape Amazon review pages using requests + BeautifulSoup.
    Works without any API key. Uses rotating user agents.
    """
    reviews = []
    seen    = set()
    session = requests.Session()

    # First visit the product page to get cookies (avoids 503 blocks)
    try:
        product_url = f"https://www.{domain}/dp/{asin}"
        session.get(product_url, headers=_headers(), timeout=15)
        time.sleep(random.uniform(1.0, 2.0))
    except Exception:
        pass

    sort_filters = [
        ("helpful",  "helpful"),
        ("recent",   "recent"),
        ("critical", "critical"),
        ("positive", "positive"),
    ]

    for sort_key, sort_val in sort_filters:
        if len(reviews) >= max_reviews:
            break

        page = 1
        consecutive_empty = 0

        while len(reviews) < max_reviews:
            url = (f"https://www.{domain}/product-reviews/{asin}"
                   f"?pageNumber={page}&sortBy={sort_val}&reviewerType=all_reviews")
            try:
                resp = session.get(url, headers=_headers(), timeout=20)
                time.sleep(random.uniform(0.8, 1.8))   # polite delay

                if resp.status_code == 503:
                    print(f"[WARN] Amazon 503 — backing off sort={sort_key} page={page}")
                    time.sleep(5)
                    break

                if resp.status_code != 200:
                    print(f"[WARN] HTTP {resp.status_code} sort={sort_key} page={page}")
                    break

                soup   = BeautifulSoup(resp.text, "html.parser")

                # Extract review bodies
                blocks = soup.select("div[data-hook='review']")
                added  = 0

                for block in blocks:
                    # Text
                    body_el = (block.select_one("span[data-hook='review-body']") or
                               block.select_one(".review-text-content span"))
                    if not body_el:
                        continue
                    text = body_el.get_text(separator=" ", strip=True)
                    if not text or text in seen or len(text) < 10:
                        continue

                    # Date
                    date_el = block.select_one("span[data-hook='review-date']")
                    date    = date_el.get_text(strip=True) if date_el else ""

                    seen.add(text)
                    reviews.append({"text": text, "date": date})
                    added += 1

                if added == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        print(f"[INFO] No more reviews sort={sort_key} at page={page}")
                        break
                else:
                    consecutive_empty = 0
                    print(f"[INFO] Direct sort={sort_key} page={page}: +{added} → total={len(reviews)}")

                # Check if next page exists
                next_btn = soup.select_one("li.a-last a")
                if not next_btn:
                    break

                page += 1

            except Exception as e:
                print(f"[WARN] Direct scrape error sort={sort_key} page={page}: {e}")
                break

    print(f"[INFO] Direct scrape total: {len(reviews)} reviews for ASIN={asin}")
    return reviews


# =============================================================================
# METHOD 2 — ScrapingDog API (optional — better success rate, 1000 free credits)
# =============================================================================
def _scrape_amazon_scrapingdog(asin: str,
                                domain: str = "amazon.in",
                                max_reviews: int = MAX_REVIEWS) -> list:
    reviews = []
    seen    = set()
    country = domain.replace("amazon.", "")

    for sort in ["helpful", "recent"]:
        page = 1
        consecutive_empty = 0

        while len(reviews) < max_reviews:
            try:
                resp = requests.get(
                    "https://api.scrapingdog.com/amazon/reviews",
                    params={
                        "api_key": SCRAPINGDOG_KEY,
                        "asin":    asin,
                        "country": country,
                        "page":    page,
                        "sort_by": sort,
                    },
                    timeout=30
                )
                if resp.status_code != 200:
                    break

                data = resp.json()
                raw  = data.get("reviews", [])

                if not raw:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        break
                    page += 1
                    continue

                consecutive_empty = 0
                added = 0
                for rev in raw:
                    text = (rev.get("body") or rev.get("text") or
                            rev.get("review_body") or "")
                    date = rev.get("date") or ""
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    reviews.append({"text": text.strip(), "date": str(date)})
                    added += 1

                print(f"[INFO] ScrapingDog sort={sort} page={page}: +{added} → total={len(reviews)}")

                if not data.get("next_page", False):
                    break
                page += 1

            except Exception as e:
                print(f"[WARN] ScrapingDog error: {e}")
                break

    print(f"[INFO] ScrapingDog total: {len(reviews)} reviews")
    return reviews


# =============================================================================
# METHOD 3 — Flipkart via RapidAPI (free tier)
# =============================================================================
def _scrape_flipkart(product_url: str, max_reviews: int = MAX_REVIEWS) -> list:
    reviews = []
    seen    = set()
    page    = 1

    while len(reviews) < max_reviews:
        try:
            resp = requests.get(
                "https://flipkart-reviews.p.rapidapi.com/reviews",
                params={"url": product_url, "page": page},
                headers={
                    "X-RapidAPI-Key":  RAPIDAPI_KEY,
                    "X-RapidAPI-Host": "flipkart-reviews.p.rapidapi.com",
                },
                timeout=30
            )
            if resp.status_code != 200:
                break

            data = resp.json()
            raw  = data.get("reviews", []) or data.get("data", [])
            if not raw:
                break

            added = 0
            for rev in raw:
                text = (rev.get("review") or rev.get("text") or
                        rev.get("body")   or rev.get("content") or "")
                date = rev.get("date") or ""
                if not text or text in seen:
                    continue
                seen.add(text)
                reviews.append({"text": text.strip(), "date": str(date)})
                added += 1

            print(f"[INFO] Flipkart API page={page}: +{added} → total={len(reviews)}")
            if not data.get("next_page") and added == 0:
                break
            page += 1

        except Exception as e:
            print(f"[WARN] Flipkart API error page={page}: {e}")
            break

    print(f"[INFO] Flipkart total: {len(reviews)} reviews")
    return reviews


# =============================================================================
# URL / ASIN HELPERS
# =============================================================================
def extract_domain(url: str) -> str:
    m = re.search(r"(amazon\.[a-z\.]+)", url)
    return m.group(1) if m else "amazon.in"


def extract_asin(url_or_asin: str) -> str:
    url_or_asin = url_or_asin.strip()
    if re.fullmatch(r"[A-Z0-9]{10}", url_or_asin):
        return url_or_asin
    m = re.search(r"/dp/([A-Z0-9]{10})", url_or_asin)
    if m:
        return m.group(1)
    m = re.search(r"([A-Z0-9]{10})", url_or_asin)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract ASIN from: {url_or_asin}")


def is_flipkart_url(url: str) -> bool:
    return "flipkart.com" in url.lower()


def is_amazon_url(url: str) -> bool:
    return "amazon." in url.lower() or bool(re.fullmatch(r"[A-Z0-9]{10}", url.strip()))


# =============================================================================
# PRODUCT METADATA via SerpAPI (free tier fallback)
# =============================================================================
def _extract_reviews_from_result(results: dict, add_review_fn) -> int:
    extracted = 0
    ri = results.get("reviews_information", {})
    for key in ("authors_reviews", "top_reviews", "other_countries_reviews"):
        for rev in ri.get(key, []):
            text = rev.get("text") or rev.get("body") or rev.get("content") or ""
            add_review_fn(text, rev.get("date"))
            if text: extracted += 1

    for insight in ri.get("summary", {}).get("insights", []):
        for ex in insight.get("examples", []):
            snippet = ex.get("snippet", "")
            add_review_fn(snippet, "")
            if snippet: extracted += 1
    return extracted


def _fetch_metadata(asin: str, domain: str = "amazon.in") -> tuple:
    try:
        from serpapi import GoogleSearch
        res  = GoogleSearch({
            "engine":        "amazon_product",
            "amazon_domain": domain,
            "asin":          asin,
            "api_key":       SERP_API_KEY,
        }).get_dict()
        prod = res.get("product_results", {})
        
        fallback_reviews = []
        seen = set()
        def add_review(text, date_str):
            if text and text not in seen:
                fallback_reviews.append({"text": text, "date": date_str or ""})
                seen.add(text)
                
        _extract_reviews_from_result(res, add_review)
        
        return {
            "title":         prod.get("title",     f"Amazon Product ({asin})"),
            "thumbnail":     prod.get("thumbnail", ""),
            "rating":        prod.get("rating",    0),
            "reviews_total": prod.get("reviews",   0),
        }, fallback_reviews
    except Exception as e:
        print(f"[WARN] Metadata fetch failed: {e}")
        return {"title": f"Amazon Product ({asin})",
                "thumbnail": "", "rating": 0, "reviews_total": 0}, []


# =============================================================================
# MASTER FETCH — auto-detects platform, uses best available method
# =============================================================================
def fetch_reviews_multi_page(url_or_asin: str,
                              target_domain: str = "amazon.in",
                              pages: int = 50,
                              max_reviews: int = MAX_REVIEWS):
    """
    Fetches reviews from Amazon or Flipkart automatically.

    Priority order (Amazon):
      1. ScrapingDog API  — if SCRAPINGDOG_KEY is set in .env
      2. Direct scraping  — always works, no API key needed
    """
    url_or_asin = url_or_asin.strip()

    # ── Flipkart ──────────────────────────────────────────────────────────────
    if is_flipkart_url(url_or_asin):
        print("[INFO] Detected Flipkart — using RapidAPI")
        if not RAPIDAPI_KEY or RAPIDAPI_KEY == "YOUR_RAPIDAPI_KEY_HERE":
            print("[WARN] RAPIDAPI_KEY not set. Get free key at rapidapi.com")
            return [], {"title": "Flipkart Product", "thumbnail": "",
                        "rating": 0, "reviews_total": 0}
        reviews = _scrape_flipkart(url_or_asin, max_reviews)
        return reviews, {"title": "Flipkart Product", "thumbnail": "",
                         "rating": 0, "reviews_total": len(reviews)}

    # ── Amazon ────────────────────────────────────────────────────────────────
    try:
        asin = extract_asin(url_or_asin)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return [], {}

    domain = extract_domain(url_or_asin) if "amazon." in url_or_asin else target_domain

    # Metadata (SerpAPI free plan — works fine)
    print(f"[INFO] Fetching product metadata for ASIN={asin} …")
    product_details, fallback_reviews = _fetch_metadata(asin, domain)
    print(f"[INFO] Product : {product_details.get('title','?')[:70]}")
    print(f"[INFO] Ratings : {product_details.get('reviews_total','?')}")

    # Choose best available review method
    if SCRAPINGDOG_KEY and SCRAPINGDOG_KEY not in ("", "YOUR_SCRAPINGDOG_KEY_HERE"):
        print("[INFO] Using ScrapingDog API (premium scraping) …")
        reviews = _scrape_amazon_scrapingdog(asin, domain, max_reviews)
        if not reviews:
            print("[INFO] ScrapingDog returned 0 — falling back to direct scraping …")
            reviews = _scrape_amazon_direct(asin, domain, max_reviews)
    else:
        print("[INFO] SCRAPINGDOG_KEY not set — using direct scraping (free) …")
        reviews = _scrape_amazon_direct(asin, domain, max_reviews)

    if not reviews:
        print("[WARN] Blocked by Amazon. Falling back to SerpAPI metadata reviews …")
        reviews = fallback_reviews

    if not reviews:
        print("[WARN] Could not fetch any reviews at all.")
        print("       Tip: sign up free at scrapingdog.com and add SCRAPINGDOG_KEY to .env")

    return reviews[:max_reviews], product_details


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================
def analyze_product(asin: str, pages: int = 50) -> tuple:
    reviews, product_details = fetch_reviews_multi_page(
        asin, pages=pages, max_reviews=MAX_REVIEWS
    )

    if not reviews:
        return {
            "total_reviews":         0,
            "fake_percent":          0.0,
            "genuine_percent":       0.0,
            "avg_confidence":        0.0,
            "avg_fake_probability":  0.0,
            "product_title":         product_details.get("title",         "Unknown Product"),
            "product_image":         product_details.get("thumbnail",     ""),
            "product_rating":        product_details.get("rating",        0),
            "product_reviews_total": product_details.get("reviews_total", 0),
        }, pd.DataFrame()

    texts = [r["text"] for r in reviews]
    print(f"[INFO] Running inference on {len(texts)} reviews …")
    labels, confidences, fake_probs = predict_reviews_batch(texts)

    results_data     = []
    fake_count       = 0
    total_confidence = 0.0
    total_fake_prob  = 0.0

    for i, rev in enumerate(reviews):
        label      = labels[i]
        confidence = confidences[i]
        fake_prob  = fake_probs[i]
        if label == "Fake":
            fake_count += 1
        total_confidence += confidence
        total_fake_prob  += fake_prob

        explanation = []
        if i < 5:
            try:
                explanation = explain_review(rev["text"], label)
            except Exception:
                pass

        results_data.append({
            "review":           rev["text"],
            "date":             rev["date"],
            "prediction":       label,
            "confidence":       round(confidence, 2),
            "fake_probability": round(fake_prob, 2),
            "explanation":      explanation,
        })

    total    = len(reviews)
    fake_pct = (fake_count / total * 100) if total else 0.0
    avg_conf = total_confidence / total   if total else 0.0
    avg_fake = total_fake_prob  / total   if total else 0.0

    df_results = pd.DataFrame(results_data)
    df_results.to_csv("analysis_results.csv", index=False)
    print(f"[INFO] Done — {total} reviews, {fake_pct:.1f}% fake.")

    return {
        "total_reviews":         int(total),
        "fake_percent":          float(round(fake_pct, 2)),
        "genuine_percent":       float(round(100 - fake_pct, 2)),
        "avg_confidence":        float(round(avg_conf, 2)),
        "avg_fake_probability":  float(round(avg_fake, 2)),
        "product_title":         product_details.get("title",         "Product"),
        "product_image":         product_details.get("thumbnail",     ""),
        "product_rating":        product_details.get("rating",        0),
        "product_reviews_total": product_details.get("reviews_total", 0),
    }, df_results