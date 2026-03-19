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
from transformers import RobertaForSequenceClassification, AutoTokenizer, pipeline
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
SERP_API_KEY  = os.getenv("SERP_API_KEY", "262ad8b51b449946485141e9ee2521a8d0120bd6b0ba609c667ed3a3d56d0495")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENUINE_THRESHOLD = 60.0   # minimum P(Genuine) % to call a review Genuine

# ── Locate latest RoBERTa checkpoint ─────────────────────────────────────────
_CHECKPOINT_BASE = "./roberta_checkpoints"
_checkpoint_dirs = glob.glob(os.path.join(_CHECKPOINT_BASE, "checkpoint-*"))
if _checkpoint_dirs:
    MODEL_PATH = sorted(_checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))[-1]
else:
    MODEL_PATH = _CHECKPOINT_BASE
print(f"[INFO] Using RoBERTa checkpoint: {MODEL_PATH}")

# ── Locate ensemble models ────────────────────────────────────────────────────
# The models are stored directly in the `model` root folder.
ENSEMBLE_PATH = "./model"

# =============================================================================
# LOAD MODELS
# =============================================================================
MODELS_LOADED = False
roberta_model = None
tokenizer     = None
rf = lr_base = xgb = meta_learner = tfidf = None

try:
    # ── RoBERTa ──────────────────────────────────────────────────────────────
    roberta_model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    roberta_model.to(DEVICE)
    roberta_model.eval()

    # AutoTokenizer with use_fast=False avoids the sentencepiece / tiktoken error
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    # ── Ensemble ML models ───────────────────────────────────────────────────
    rf           = joblib.load(os.path.join(ENSEMBLE_PATH, "random_forest.pkl"))
    lr_base      = joblib.load(os.path.join(ENSEMBLE_PATH, "logistic_regression.pkl"))
    xgb          = joblib.load(os.path.join(ENSEMBLE_PATH, "xgboost.pkl"))
    meta_learner = joblib.load(os.path.join(ENSEMBLE_PATH, "meta_learner.pkl"))
    tfidf        = joblib.load(os.path.join(ENSEMBLE_PATH, "tfidf.pkl"))

    MODELS_LOADED = True
    print("[INFO] All models loaded successfully.")

except Exception as e:
    print(f"[WARN] Models not fully loaded. Did you run the export cell in the notebook?\n       {e}")

# =============================================================================
# LIME EXPLAINER
# =============================================================================
explainer = LimeTextExplainer(class_names=["Fake", "Genuine"])

# =============================================================================
# TEXT CLEANING
# =============================================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",              " ", text)   # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[^\w\s]",              " ", text)   # special chars
    text = re.sub(r"\s+",                  " ", text).strip()
    return text

# =============================================================================
# HELPER — class-aware P(Genuine)
# =============================================================================
def _get_genuine_proba(model, X):
    """
    Extracts P(Genuine) from a sklearn model safely,
    regardless of internal class ordering.
    """
    col = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(X)[:, col]

# =============================================================================
# CORE PREDICTION — returns (N, 2) array: [[P(Fake), P(Genuine)], ...]
# =============================================================================
def get_prediction_probs(texts):
    """
    Full stacking ensemble prediction for a list of raw text strings.
    Returns numpy array of shape (N, 2): columns are [P(Fake), P(Genuine)].
    Falls back to [[0.5, 0.5]] per sample if models are not loaded.
    """
    if not MODELS_LOADED:
        return np.array([[0.5, 0.5]] * len(texts))

    # ── Text cleaning + feature engineering ──────────────────────────────────
    cleaned_texts = [clean_text(t) for t in texts]
    lengths       = np.array([[len(t.split())] for t in cleaned_texts])

    X_tfidf   = tfidf.transform(cleaned_texts)
    X_len_sp  = scipy.sparse.csr_matrix(lengths)
    X_combined = scipy.sparse.hstack([X_tfidf, X_len_sp])

    # ── ML base models ────────────────────────────────────────────────────────
    p_rf  = _get_genuine_proba(rf,      X_combined)
    p_lr  = _get_genuine_proba(lr_base, X_combined)
    p_xgb = _get_genuine_proba(xgb,     X_combined)

    # ── RoBERTa (batched) ────────────────────────────────────────────────────
    batch_size    = 32
    all_p_roberta = []
    for i in range(0, len(texts), batch_size):
        batch = cleaned_texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = roberta_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_p_roberta.extend(probs[:, 1])   # P(Genuine) per sample
    p_roberta = np.array(all_p_roberta)

    # ── Meta-learner ──────────────────────────────────────────────────────────
    meta_features = np.column_stack([p_rf, p_lr, p_xgb, p_roberta])
    final_probs   = meta_learner.predict_proba(meta_features)  # (N, 2)
    return final_probs

# =============================================================================
# SINGLE REVIEW PREDICTION
# =============================================================================
def predict_review(text: str):
    """
    Predict a single review.
    Returns: (label, confidence_pct, fake_probability_pct)
    """
    if not MODELS_LOADED:
        return "Unknown", 0.0, 0.0

    probs            = get_prediction_probs([text])[0]
    fake_probability = probs[0] * 100
    genuine_prob     = probs[1] * 100

    if genuine_prob >= GENUINE_THRESHOLD:
        return "Genuine", round(genuine_prob, 2), round(fake_probability, 2)
    else:
        return "Fake", round(fake_probability, 2), round(fake_probability, 2)

# =============================================================================
# BATCH REVIEW PREDICTION
# =============================================================================
def predict_reviews_batch(texts):
    """
    Predict a list of reviews in one pass.
    Returns: (labels, confidences, fake_probabilities) — all lists of floats.
    """
    if not texts or not MODELS_LOADED:
        return [], [], []

    probs            = get_prediction_probs(texts)
    all_labels       = []
    all_confidences  = []
    all_fake_probs   = []

    for p in probs:
        fake_prob = p[0] * 100
        gen_prob  = p[1] * 100

        if gen_prob >= GENUINE_THRESHOLD:
            all_labels.append("Genuine")
            all_confidences.append(round(gen_prob, 2))
        else:
            all_labels.append("Fake")
            all_confidences.append(round(fake_prob, 2))

        all_fake_probs.append(round(fake_prob, 2))

    return all_labels, all_confidences, all_fake_probs

# =============================================================================
# LIME EXPLANATION
# =============================================================================
def explain_review(text: str, label: str):
    """
    Generate LIME word-level explanation for a single review.
    Returns list of {"word": str, "weight": float}.
    """
    if not MODELS_LOADED:
        return []
    try:
        exp        = explainer.explain_instance(
            text,
            get_prediction_probs,
            num_features=5,
            num_samples=100
        )
        target_idx = 0 if label == "Fake" else 1
        return [
            {"word": w, "weight": float(wt)}
            for w, wt in exp.as_list(label=target_idx)
        ]
    except Exception as e:
        print(f"[WARN] LIME explanation error: {e}")
        return []

# =============================================================================
# REVIEW FETCHING — multi-page, multi-domain
# =============================================================================
def _extract_reviews_from_result(results: dict, add_review_fn):
    """
    Tries every known SerpAPI field structure to extract reviews.
    Returns the count of reviews extracted.
    """
    extracted = 0

    # 1. Primary: 'reviews' array (amazon_reviews engine)
    for review in results.get("reviews", []):
        text = (
            review.get("body") or review.get("content") or
            review.get("text") or review.get("snippet") or ""
        )
        add_review_fn(text, review.get("date") or review.get("date_string"))
        if text:
            extracted += 1

    # 2. Nested reviews_information block (amazon_product engine)
    reviews_info = results.get("reviews_information", {})
    for key in ("authors_reviews", "top_reviews", "other_countries_reviews"):
        for review in reviews_info.get(key, []):
            text = (
                review.get("text") or review.get("body") or
                review.get("content") or ""
            )
            add_review_fn(text, review.get("date"))
            if text:
                extracted += 1

    # 3. Insight snippets from summary (amazon_product engine)
    summary_info = reviews_info.get("summary", {})
    for insight in summary_info.get("insights", []):
        for example in insight.get("examples", []):
            snippet = example.get("snippet", "")
            add_review_fn(snippet, "")
            if snippet:
                extracted += 1

    return extracted


def fetch_reviews_multi_page(asin, target_domain="amazon.com", pages=5, max_reviews=500):
    """
    Fetches up to max_reviews reviews for an Amazon product (ASIN).

    Strategy:
      1. amazon_product engine  → product metadata + any embedded reviews.
      2. amazon_reviews engine  → bulk reviews across multiple domains / pages.
      3. Fallback               → paginate amazon_product if step 2 returns nothing.
    """
    all_reviews     = []
    seen_texts      = set()
    product_details = {}
    pages           = max(1, min(int(pages or 1), 20))
    max_reviews     = max(1, min(int(max_reviews or 500), 1000))

    domains = [
        target_domain,
        "amazon.co.uk",
        "amazon.ca",
        "amazon.in",
        "amazon.com.au"
    ]

    def add_review(text, date_str):
        if not text or not text.strip():
            return
        if text in seen_texts:
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

    # ── Step 1: Product metadata + embedded reviews ───────────────────────────
    try:
        meta_params = {
            "engine":         "amazon_product",
            "amazon_domain":  target_domain,
            "asin":           asin,
            "api_key":        SERP_API_KEY
        }
        meta_results = GoogleSearch(meta_params).get_dict()
        print(f"[DEBUG] amazon_product keys: {list(meta_results.keys())}")

        if "product_results" in meta_results:
            prod = meta_results["product_results"]
            product_details = {
                "title":          prod.get("title", f"Amazon Product ({asin})"),
                "thumbnail":      prod.get("thumbnail", ""),
                "rating":         prod.get("rating", 0),
                "reviews_total":  prod.get("reviews", 0)
            }

        got = _extract_reviews_from_result(meta_results, add_review)
        print(f"[DEBUG] Reviews from amazon_product: {got}")

    except Exception as e:
        print(f"[WARN] Product metadata fetch failed for {asin}: {e}")

    # ── Step 2: Bulk reviews via amazon_reviews engine ────────────────────────
    amazon_reviews_engine_worked = False

    for domain in domains:
        if len(all_reviews) >= max_reviews:
            break

        consecutive_empty = 0
        for page in range(1, pages + 1):
            if len(all_reviews) >= max_reviews:
                break
            try:
                review_params = {
                    "engine":         "amazon_reviews",
                    "amazon_domain":  domain,
                    "asin":           asin,
                    "page":           page,
                    "api_key":        SERP_API_KEY
                }
                results = GoogleSearch(review_params).get_dict()
            except Exception as e:
                print(f"[WARN] amazon_reviews failed (domain={domain}, page={page}): {e}")
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                continue

            if page == 1 and domain == target_domain:
                print(f"[DEBUG] amazon_reviews keys: {list(results.keys())}")
                error_info = results.get("error") or results.get(
                    "search_information", {}
                ).get("query_displayed")
                if error_info:
                    print(f"[DEBUG] amazon_reviews error/info: {error_info}")

            before = len(all_reviews)
            _extract_reviews_from_result(results, add_review)
            added = len(all_reviews) - before

            if added == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
            else:
                consecutive_empty = 0
                amazon_reviews_engine_worked = True

    # ── Step 3: Fallback — paginate amazon_product ───────────────────────────
    if not amazon_reviews_engine_worked:
        print("[INFO] amazon_reviews returned 0 — falling back to amazon_product pagination")
        for domain in domains:
            if len(all_reviews) >= max_reviews:
                break
            for page in range(1, pages + 1):
                if len(all_reviews) >= max_reviews:
                    break
                try:
                    params = {
                        "engine":         "amazon_product",
                        "amazon_domain":  domain,
                        "asin":           asin,
                        "page":           page,
                        "api_key":        SERP_API_KEY
                    }
                    results = GoogleSearch(params).get_dict()
                    before  = len(all_reviews)
                    _extract_reviews_from_result(results, add_review)
                    added   = len(all_reviews) - before
                    print(f"[DEBUG] amazon_product fallback page={page} domain={domain}: +{added}")
                    if added == 0:
                        break
                except Exception as e:
                    print(f"[WARN] amazon_product fallback failed (domain={domain}, page={page}): {e}")
                    break

    print(f"[INFO] Total reviews fetched: {len(all_reviews)} for ASIN={asin}")
    return all_reviews[:max_reviews], product_details

# =============================================================================
# URL / ASIN HELPERS
# =============================================================================
def extract_domain(url_or_asin: str) -> str:
    url_or_asin = url_or_asin.strip()
    match = re.search(r"amazon\.([a-z\.]+)/", url_or_asin)
    if match:
        return f"amazon.{match.group(1)}"
    return "amazon.com"


def extract_asin(url_or_asin: str) -> str:
    """
    Extracts ASIN from a full Amazon URL or returns it directly
    if already in ASIN format (10 uppercase alphanumeric chars).
    """
    url_or_asin = url_or_asin.strip()

    if re.fullmatch(r"[A-Z0-9]{10}", url_or_asin):
        return url_or_asin

    match = re.search(r"([A-Z0-9]{10})", url_or_asin)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract a valid ASIN from: {url_or_asin}")

# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================
def analyze_product(asin: str, pages: int = 5) -> tuple:
    """
    Full pipeline: fetch → predict → summarise → export CSV.

    Returns:
        summary   (dict)         — aggregate statistics + product info
        df_results (pd.DataFrame) — per-review predictions
    """
    domain   = extract_domain(asin)
    asin_id  = extract_asin(asin)
    reviews, product_details = fetch_reviews_multi_page(
        asin_id, domain, pages=pages, max_reviews=500
    )

    if not reviews:
        return {
            "total_reviews":   0,
            "fake_percent":    0.0,
            "genuine_percent": 0.0,
            "avg_confidence":  0.0
        }, pd.DataFrame()

    texts  = [rev["text"] for rev in reviews]
    labels, confidences, fake_probabilities = predict_reviews_batch(texts)

    results_data         = []
    fake_count           = 0
    total_confidence     = 0.0
    total_fake_prob      = 0.0

    for i, rev_dict in enumerate(reviews):
        label            = labels[i]
        confidence       = confidences[i]
        fake_probability = fake_probabilities[i]

        if label == "Fake":
            fake_count += 1

        total_confidence += confidence
        total_fake_prob  += fake_probability

        # LIME only on first 3 reviews to keep latency acceptable
        explanation = []
        if i < 3:
            explanation = explain_review(rev_dict["text"], label)

        results_data.append({
            "review":           rev_dict["text"],
            "date":             rev_dict["date"],
            "prediction":       label,
            "confidence":       round(confidence, 2),
            "fake_probability": round(fake_probability, 2),
            "explanation":      explanation
        })

    total_reviews    = len(reviews)
    fake_percent     = (fake_count / total_reviews * 100) if total_reviews else 0.0
    avg_confidence   = total_confidence / total_reviews if total_reviews else 0.0
    avg_fake_prob    = total_fake_prob  / total_reviews if total_reviews else 0.0

    df_results = pd.DataFrame(results_data)
    df_results.to_csv("analysis_results.csv", index=False)
    print(f"[INFO] Analysis complete. {total_reviews} reviews processed.")

    summary = {
        "total_reviews":         int(total_reviews),
        "fake_percent":          float(round(fake_percent,   2)),
        "genuine_percent":       float(round(100 - fake_percent, 2)) if total_reviews else 0.0,
        "avg_confidence":        float(round(avg_confidence, 2)),
        "avg_fake_probability":  float(round(avg_fake_prob,  2)),
        "product_title":         product_details.get("title",         f"Amazon Product ({asin_id})"),
        "product_image":         product_details.get("thumbnail",     ""),
        "product_rating":        product_details.get("rating",        0),
        "product_reviews_total": product_details.get("reviews_total", 0)
    }

    return summary, df_results
    