import torch
import numpy as np
import pandas as pd
import os
import glob
import re
import joblib
import scipy.sparse
import dateparser

from dotenv import load_dotenv
from serpapi import GoogleSearch
from transformers import RobertaForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
SERP_API_KEY         = os.getenv("SERP_API_KEY", "262ad8b51b449946485141e9ee2521a8d0120bd6b0ba609c667ed3a3d56d0495")
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENUINE_THRESHOLD    = 60.0   # P(Genuine) % threshold
MAX_REVIEWS          = 2000   # hard ceiling — fetches all pages until this
MAX_PAGES_PER_DOMAIN = 50     # pages to try per domain before moving on

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

# Try common ensemble paths
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


def _pkl(name):
    return joblib.load(os.path.join(ENSEMBLE_PATH, name))


try:
    roberta_model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    roberta_model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    rf           = _pkl("random_forest.pkl")
    lr_base      = _pkl("logistic_regression.pkl")
    xgb          = _pkl("xgboost.pkl")
    meta_learner = _pkl("meta_learner.pkl")
    tfidf        = _pkl("tfidf.pkl")

    MODELS_LOADED = True
    print("[INFO] All models loaded successfully.")

except Exception as e:
    print(f"[WARN] Models not fully loaded. Run the export cell in the notebook.\n       {e}")

# =============================================================================
# LIME EXPLAINER
# =============================================================================
explainer = LimeTextExplainer(class_names=["Fake", "Genuine"])

# =============================================================================
# TEXT CLEANING  (mirrors notebook clean_text exactly)
# =============================================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",              " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]",              " ", text)
    text = re.sub(r"\s+",                  " ", text).strip()
    return text

# =============================================================================
# CLASS-AWARE PROBABILITY EXTRACTION
# =============================================================================
def _get_genuine_proba(model, X):
    """P(Genuine) regardless of sklearn internal class ordering."""
    col = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(X)[:, col]

# =============================================================================
# CORE PREDICTION  →  returns (N, 2) [P(Fake), P(Genuine)]
# =============================================================================
def get_prediction_probs(texts):
    if not MODELS_LOADED:
        return np.array([[0.5, 0.5]] * len(texts))

    cleaned = [clean_text(t) for t in texts]
    lengths  = np.array([[len(t.split())] for t in cleaned])

    X_tfidf    = tfidf.transform(cleaned)
    X_len_sp   = scipy.sparse.csr_matrix(lengths)
    X_combined = scipy.sparse.hstack([X_tfidf, X_len_sp])

    p_rf  = _get_genuine_proba(rf,      X_combined)
    p_lr  = _get_genuine_proba(lr_base, X_combined)
    p_xgb = _get_genuine_proba(xgb,     X_combined)

    # RoBERTa — process in small batches to avoid OOM
    all_p_roberta = []
    for i in range(0, len(cleaned), 32):
        batch  = cleaned[i: i + 32]
        inputs = tokenizer(batch, truncation=True, padding=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = roberta_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_p_roberta.extend(probs[:, 1])

    p_roberta     = np.array(all_p_roberta)
    meta_features = np.column_stack([p_rf, p_lr, p_xgb, p_roberta])
    return meta_learner.predict_proba(meta_features)   # (N, 2)

# =============================================================================
# SINGLE-REVIEW PREDICTION
# =============================================================================
def predict_review(text: str):
    if not MODELS_LOADED:
        return "Unknown", 0.0, 0.0
    probs    = get_prediction_probs([text])[0]
    fake_pct = probs[0] * 100
    gen_pct  = probs[1] * 100
    if gen_pct >= GENUINE_THRESHOLD:
        return "Genuine", round(gen_pct, 2), round(fake_pct, 2)
    return "Fake", round(fake_pct, 2), round(fake_pct, 2)

# =============================================================================
# BATCH PREDICTION
# =============================================================================
def predict_reviews_batch(texts):
    if not texts or not MODELS_LOADED:
        return [], [], []

    # Process in chunks of 256 to print progress for large datasets
    chunk = 256
    all_probs = []
    for start in range(0, len(texts), chunk):
        batch_texts = texts[start: start + chunk]
        all_probs.append(get_prediction_probs(batch_texts))
        print(f"[INFO]   Predicted {min(start + chunk, len(texts))}/{len(texts)} …")

    probs = np.vstack(all_probs)
    labels, confs, fake_probs = [], [], []
    for p in probs:
        fake_pct = p[0] * 100
        gen_pct  = p[1] * 100
        if gen_pct >= GENUINE_THRESHOLD:
            labels.append("Genuine")
            confs.append(round(gen_pct, 2))
        else:
            labels.append("Fake")
            confs.append(round(fake_pct, 2))
        fake_probs.append(round(fake_pct, 2))
    return labels, confs, fake_probs

# =============================================================================
# LIME EXPLANATION  (called for first 5 reviews only — expensive)
# =============================================================================
def explain_review(text: str, label: str):
    if not MODELS_LOADED:
        return []
    try:
        exp = explainer.explain_instance(
            text, get_prediction_probs, num_features=5, num_samples=100
        )
        target_idx = 0 if label == "Fake" else 1
        return [{"word": w, "weight": float(wt)}
                for w, wt in exp.as_list(label=target_idx)]
    except Exception as e:
        print(f"[WARN] LIME error: {e}")
        return []

# =============================================================================
# REVIEW EXTRACTION — handles every known SerpAPI shape
# =============================================================================
def _extract_reviews_from_result(results: dict, add_review_fn) -> int:
    extracted = 0

    # 1. Primary: amazon_reviews engine
    for rev in results.get("reviews", []):
        text = (rev.get("body") or rev.get("content") or
                rev.get("text") or rev.get("snippet") or "")
        add_review_fn(text, rev.get("date") or rev.get("date_string"))
        if text:
            extracted += 1

    # 2. Nested reviews_information (amazon_product engine)
    ri = results.get("reviews_information", {})
    for key in ("authors_reviews", "top_reviews", "other_countries_reviews"):
        for rev in ri.get(key, []):
            text = rev.get("text") or rev.get("body") or rev.get("content") or ""
            add_review_fn(text, rev.get("date"))
            if text:
                extracted += 1

    # 3. Insight snippets
    for insight in ri.get("summary", {}).get("insights", []):
        for ex in insight.get("examples", []):
            snippet = ex.get("snippet", "")
            add_review_fn(snippet, "")
            if snippet:
                extracted += 1

    return extracted

# =============================================================================
# FETCH ALL REVIEWS — exhaustive pagination across all domains
# =============================================================================
def fetch_reviews_multi_page(asin: str,
                              target_domain: str = "amazon.com",
                              pages: int = 20,
                              max_reviews: int = MAX_REVIEWS):
    """
    Fetches as many reviews as possible for a given ASIN.

    Strategy
    --------
    1. amazon_product engine  → product metadata + embedded reviews.
    2. amazon_reviews engine  → paginate every page across 8 domains until
       max_reviews reached or 2 consecutive empty pages per domain.
    3. Fallback               → paginate amazon_product if step 2 yields zero.
    """
    all_reviews:     list = []
    seen_texts:      set  = set()
    product_details: dict = {}

    pages       = max(1, min(int(pages or 20), MAX_PAGES_PER_DOMAIN))
    max_reviews = max(1, min(int(max_reviews or MAX_REVIEWS), MAX_REVIEWS))

    # 8 Amazon locales — ordered by review volume
    domains = [
        target_domain,
        "amazon.co.uk",
        "amazon.ca",
        "amazon.in",
        "amazon.com.au",
        "amazon.de",
        "amazon.fr",
        "amazon.co.jp",
    ]
    # Deduplicate in case target_domain is already in list
    seen_domains = []
    for d in domains:
        if d not in seen_domains:
            seen_domains.append(d)
    domains = seen_domains

    def add_review(text, date_str):
        if not text or not text.strip() or text in seen_texts:
            return
        parsed_date = None
        if date_str:
            try:
                d = dateparser.parse(str(date_str))
                if d:
                    parsed_date = d.isoformat()
            except Exception:
                pass
        all_reviews.append({"text": text.strip(), "date": parsed_date or ""})
        seen_texts.add(text)

    # ── Step 1: product metadata ──────────────────────────────────────────────
    try:
        meta_res = GoogleSearch({
            "engine":        "amazon_product",
            "amazon_domain": target_domain,
            "asin":          asin,
            "api_key":       SERP_API_KEY,
        }).get_dict()

        if "product_results" in meta_res:
            prod = meta_res["product_results"]
            product_details = {
                "title":         prod.get("title",     f"Amazon Product ({asin})"),
                "thumbnail":     prod.get("thumbnail", ""),
                "rating":        prod.get("rating",    0),
                "reviews_total": prod.get("reviews",   0),
            }

        got = _extract_reviews_from_result(meta_res, add_review)
        print(f"[DEBUG] amazon_product metadata → {got} reviews")

    except Exception as e:
        print(f"[WARN] Product metadata failed: {e}")

    # ── Step 2: amazon_reviews engine — paginate all domains ─────────────────
    amazon_engine_worked = False

    for domain in domains:
        if len(all_reviews) >= max_reviews:
            break

        consecutive_empty = 0
        for page in range(1, pages + 1):
            if len(all_reviews) >= max_reviews:
                print(f"[INFO] Reached {max_reviews} review cap.")
                break

            try:
                res = GoogleSearch({
                    "engine":        "amazon_reviews",
                    "amazon_domain": domain,
                    "asin":          asin,
                    "page":          page,
                    "api_key":       SERP_API_KEY,
                }).get_dict()
            except Exception as e:
                print(f"[WARN] amazon_reviews domain={domain} page={page}: {e}")
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                continue

            # Debug info on first page of primary domain
            if page == 1 and domain == target_domain:
                err = (res.get("error") or
                       res.get("search_information", {}).get("query_displayed"))
                if err:
                    print(f"[DEBUG] amazon_reviews info: {err}")

            before = len(all_reviews)
            _extract_reviews_from_result(res, add_review)
            added  = len(all_reviews) - before

            if added == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    print(f"[INFO] No more reviews on domain={domain} at page={page}")
                    break
            else:
                consecutive_empty      = 0
                amazon_engine_worked   = True
                print(f"[INFO] domain={domain} page={page}: +{added}  total={len(all_reviews)}")

    # ── Step 3: fallback — paginate amazon_product ───────────────────────────
    if not amazon_engine_worked:
        print("[INFO] amazon_reviews returned 0 — trying amazon_product fallback")
        for domain in domains:
            if len(all_reviews) >= max_reviews:
                break
            for page in range(1, pages + 1):
                if len(all_reviews) >= max_reviews:
                    break
                try:
                    res    = GoogleSearch({
                        "engine":        "amazon_product",
                        "amazon_domain": domain,
                        "asin":          asin,
                        "page":          page,
                        "api_key":       SERP_API_KEY,
                    }).get_dict()
                    before = len(all_reviews)
                    _extract_reviews_from_result(res, add_review)
                    added  = len(all_reviews) - before
                    print(f"[DEBUG] fallback domain={domain} page={page}: +{added}")
                    if added == 0:
                        break
                except Exception as e:
                    print(f"[WARN] Fallback failed domain={domain} page={page}: {e}")
                    break

    total_fetched = len(all_reviews)
    print(f"[INFO] Fetching complete — {total_fetched} reviews for ASIN={asin}")
    return all_reviews[:max_reviews], product_details

# =============================================================================
# URL / ASIN HELPERS
# =============================================================================
def extract_domain(url_or_asin: str) -> str:
    url_or_asin = url_or_asin.strip()
    m = re.search(r"amazon\.([a-z\.]+)/", url_or_asin)
    return f"amazon.{m.group(1)}" if m else "amazon.com"


def extract_asin(url_or_asin: str) -> str:
    url_or_asin = url_or_asin.strip()
    if re.fullmatch(r"[A-Z0-9]{10}", url_or_asin):
        return url_or_asin
    m = re.search(r"([A-Z0-9]{10})", url_or_asin)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract ASIN from: {url_or_asin}")

# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================
def analyze_product(asin: str, pages: int = 20) -> tuple:
    """
    End-to-end pipeline:
      fetch all reviews  →  batch predict  →  summarise  →  write CSV

    Returns (summary: dict, df_results: pd.DataFrame)
    """
    domain  = extract_domain(asin)
    asin_id = extract_asin(asin)

    reviews, product_details = fetch_reviews_multi_page(
        asin_id, domain, pages=pages, max_reviews=MAX_REVIEWS
    )

    if not reviews:
        return {
            "total_reviews":   0,
            "fake_percent":    0.0,
            "genuine_percent": 0.0,
            "avg_confidence":  0.0,
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

        # LIME explanation only for first 5 (it is slow)
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
    print(f"[INFO] Done — {total} reviews analysed, {fake_pct:.1f}% flagged as fake.")

    summary = {
        "total_reviews":         int(total),
        "fake_percent":          float(round(fake_pct, 2)),
        "genuine_percent":       float(round(100 - fake_pct, 2)) if total else 0.0,
        "avg_confidence":        float(round(avg_conf, 2)),
        "avg_fake_probability":  float(round(avg_fake, 2)),
        "product_title":         product_details.get("title",         f"Amazon Product ({asin_id})"),
        "product_image":         product_details.get("thumbnail",     ""),
        "product_rating":        product_details.get("rating",        0),
        "product_reviews_total": product_details.get("reviews_total", 0),
    }

    return summary, df_results