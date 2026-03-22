from __future__ import annotations

import os
import re
import random
import time
import json
import html as htmllib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode

import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse

try:
    from dateparser import parse as parse_date
except Exception:  # pragma: no cover - optional dependency
    parse_date = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from serpapi import GoogleSearch
except Exception:  # pragma: no cover - optional dependency
    GoogleSearch = None

try:
    from Reviewer_timeline_detector import detect_reviewer_anomalies
except Exception:
    detect_reviewer_anomalies = None

try:
    from Review_ring_detector import detect_review_rings as detect_review_rings_module
except Exception:
    detect_review_rings_module = None

try:
    from Text_rating_mismatch import detect_text_rating_mismatch
except Exception:
    detect_text_rating_mismatch = None


if load_dotenv:
    load_dotenv()


SERP_API_KEY = (
    os.getenv("SERP_API_KEY")
    or os.getenv("262ad8b51b449946485141e9ee2521a8d0120bd6b0ba609c667ed3a3d56d0495")
    or ""
).strip()
SCRAPINGDOG_KEY = ""
RAPIDAPI_KEY = ""
GENUINE_THRESHOLD = 0.60
MAX_REVIEWS = 10000
BATCH_SIZE = 256
ANALYSIS_RESULTS_PATH = "analysis_results.csv"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = int(default)
    return max(minimum, min(v, maximum))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }


@dataclass
class ModelBundle:
    tfidf: Any = None
    rf: Any = None
    lr_base: Any = None
    xgb: Any = None
    meta_learner: Any = None
    loaded: bool = False


MODELS = ModelBundle()


def _load_models() -> None:
    model_dir = "model"
    paths = {
        "tfidf": os.path.join(model_dir, "tfidf.pkl"),
        "rf": os.path.join(model_dir, "random_forest.pkl"),
        "lr_base": os.path.join(model_dir, "logistic_regression.pkl"),
        "xgb": os.path.join(model_dir, "xgboost.pkl"),
        "meta_learner": os.path.join(model_dir, "meta_learner.pkl"),
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            continue
        try:
            model_obj = joblib.load(path)
            if hasattr(model_obj, "n_jobs"):
                try:
                    model_obj.n_jobs = 1
                except Exception:
                    pass
            setattr(MODELS, key, model_obj)
        except Exception as exc:
            # Optional models (notably xgb) may fail if dependency not installed.
            print(f"[WARN] Failed loading {key}: {exc}")
    MODELS.loaded = all([MODELS.tfidf is not None, MODELS.rf is not None, MODELS.lr_base is not None])
    if not MODELS.loaded:
        print("[WARN] Base models missing; using heuristic fallback.")


_load_models()


def clean_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_genuine_proba(model: Any, X: Any, genuine_label: int = 1) -> np.ndarray:
    if model is None:
        return np.full((X.shape[0],), 0.5, dtype=float)
    try:
        probs = model.predict_proba(X)
    except PermissionError:
        if hasattr(model, "n_jobs"):
            try:
                model.n_jobs = 1
                probs = model.predict_proba(X)
            except Exception:
                return np.full((X.shape[0],), 0.5, dtype=float)
        else:
            return np.full((X.shape[0],), 0.5, dtype=float)
    except Exception:
        return np.full((X.shape[0],), 0.5, dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None:
        return np.full((X.shape[0],), 0.5, dtype=float)
    idx = int(np.where(classes == genuine_label)[0][0]) if genuine_label in classes else 0
    return probs[:, idx]


def _heuristic_fake_probability(texts: List[str]) -> np.ndarray:
    scam_terms = {
        "best", "must buy", "value for money", "awesome", "amazing", "super", "perfect",
        "worst", "waste", "fake", "fraud", "useless", "not worth",
    }
    out = []
    for text in texts:
        t = clean_text(text)
        score = 0.50
        if len(t.split()) <= 3:
            score += 0.20
        matches = sum(1 for term in scam_terms if term in t)
        score += min(matches * 0.04, 0.24)
        repetitive = len(set(t.split())) / max(len(t.split()), 1)
        if repetitive < 0.45:
            score += 0.12
        out.append(float(min(max(score, 0.02), 0.98)))
    return np.asarray(out, dtype=float)


def get_prediction_probs(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 2), dtype=float)

    if not MODELS.loaded:
        fake_p = _heuristic_fake_probability(texts)
        genuine_p = 1.0 - fake_p
        return np.column_stack([fake_p, genuine_p])

    cleaned = [clean_text(t) for t in texts]
    lengths = np.array([len(t.split()) for t in cleaned], dtype=np.float32).reshape(-1, 1)
    X_tfidf = MODELS.tfidf.transform(cleaned)
    X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(lengths)], format="csr")

    p_rf = _get_genuine_proba(MODELS.rf, X_combined)
    p_lr = _get_genuine_proba(MODELS.lr_base, X_combined)

    if MODELS.xgb is not None:
        p_xgb = _get_genuine_proba(MODELS.xgb, X_combined)
    else:
        p_xgb = (p_rf + p_lr) / 2.0

    # Fourth feature slot (historically RoBERTa) is approximated when unavailable.
    p_bert = (p_rf + p_lr + p_xgb) / 3.0
    feature_matrix = np.column_stack([p_rf, p_lr, p_xgb, p_bert])

    if MODELS.meta_learner is not None:
        try:
            p_genuine = _get_genuine_proba(MODELS.meta_learner, feature_matrix)
        except Exception:
            p_genuine = feature_matrix.mean(axis=1)
    else:
        p_genuine = feature_matrix.mean(axis=1)

    p_genuine = np.clip(p_genuine, 0.0, 1.0)
    p_fake = 1.0 - p_genuine
    return np.column_stack([p_fake, p_genuine])


def predict_review(text: str) -> Tuple[str, float, float]:
    probs = get_prediction_probs([text])
    if probs.size == 0:
        return "Unknown", 0.0, 0.0
    fp, gp = probs[0]
    label = "Genuine" if gp >= GENUINE_THRESHOLD else "Fake"
    confidence = round(float(max(fp, gp) * 100.0), 2)
    fake_probability = round(float(fp * 100.0), 2)
    return label, confidence, fake_probability


def predict_reviews_batch(texts: List[str]) -> Tuple[List[str], List[float], List[float]]:
    labels: List[str] = []
    confs: List[float] = []
    fake_probs: List[float] = []
    if not texts:
        return labels, confs, fake_probs

    all_chunks = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        chunk_probs = get_prediction_probs(chunk)
        all_chunks.append(chunk_probs)
        print(f"[INFO]   Predicted {min(i + BATCH_SIZE, len(texts))}/{len(texts)} ...")
    probs = np.vstack(all_chunks) if all_chunks else np.zeros((0, 2))

    for fp, gp in probs:
        label = "Genuine" if gp >= GENUINE_THRESHOLD else "Fake"
        labels.append(label)
        confs.append(round(float(max(fp, gp) * 100.0), 2))
        fake_probs.append(round(float(fp * 100.0), 2))
    return labels, confs, fake_probs


def explain_review(text: str, label: str) -> List[Dict[str, Any]]:
    # Lightweight explanation using LR contribution instead of LIME.
    if MODELS.tfidf is None or MODELS.lr_base is None:
        return []
    try:
        cleaned = clean_text(text)
        vec = MODELS.tfidf.transform([cleaned])
        if vec.nnz == 0:
            return []
        coef = getattr(MODELS.lr_base, "coef_", None)
        if coef is None or coef.shape[0] == 0:
            return []
        class_idx = 0 if label == "Fake" else 1
        weights = coef[class_idx] if class_idx < coef.shape[0] else coef[0]
        feature_idx = vec.indices
        contributions = vec.data * weights[feature_idx]
        terms = MODELS.tfidf.get_feature_names_out()
        pairs = [
            {"word": str(terms[idx]), "weight": float(w)}
            for idx, w in zip(feature_idx, contributions)
            if np.isfinite(w)
        ]
        pairs.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return pairs[:5]
    except Exception:
        return []


def extract_domain(url_or_asin: str) -> str:
    raw = str(url_or_asin or "").strip()
    m = re.search(r"https?://(?:www\.)?([^/]+)", raw, flags=re.I)
    if m:
        return m.group(1).lower()
    return "amazon.com"


def extract_asin(url_or_asin: str) -> str:
    raw = str(url_or_asin or "").strip()
    if re.fullmatch(r"[A-Z0-9]{10}", raw, flags=re.I):
        return raw.upper()
    for pattern in [r"/dp/([A-Z0-9]{10})", r"/gp/product/([A-Z0-9]{10})", r"([A-Z0-9]{10})"]:
        m = re.search(pattern, raw, flags=re.I)
        if m:
            return m.group(1).upper()
    raise ValueError(f"Could not extract ASIN from: {url_or_asin}")


def is_flipkart_url(url: str) -> bool:
    return "flipkart.com" in str(url).lower()


def is_amazon_url(url: str) -> bool:
    u = str(url).lower()
    return ("amazon." in u) or bool(re.fullmatch(r"[A-Z0-9]{10}", str(url).strip(), flags=re.I))


def _extract_reviews_from_result(results: Dict[str, Any], add_review_fn) -> int:
    extracted = 0
    reviews_info = results.get("reviews_information") or {}
    for key in ("authors_reviews", "top_reviews", "other_countries_reviews"):
        for review in reviews_info.get(key, []) or []:
            text = review.get("text") or review.get("body") or review.get("content") or ""
            date = review.get("date", "")
            extracted += add_review_fn(text, date)
    for insight in results.get("insights", []) or []:
        for example in insight.get("examples", []) or []:
            snippet = example.get("snippet", "")
            extracted += add_review_fn(snippet, "")
    return extracted


def _normalize_review_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_identity(value: Any) -> str:
    raw = _normalize_review_text(value)
    if raw.lower() in {"", "unknown", "none", "nan", "null", "-", "--", "n/a"}:
        return ""
    return raw


def _review_dedupe_key(review: Dict[str, Any]) -> str:
    review_id = _normalize_identity(review.get("review_id"))
    if review_id:
        return f"id::{review_id.lower()}"

    text = _normalize_review_text(review.get("text"))
    user_name = _normalize_identity(review.get("user_name"))
    date = _normalize_identity(review.get("date"))
    star_rating = _safe_int(review.get("star_rating", 0), 0)

    if user_name and date:
        return f"ud::{user_name.lower()}::{date.lower()}::{star_rating}::{text.lower()[:220]}"
    if user_name:
        return f"u::{user_name.lower()}::{star_rating}::{text.lower()[:260]}"
    if date:
        return f"d::{date.lower()}::{star_rating}::{text.lower()[:260]}"
    return f"t::{star_rating}::{text.lower()}"


def _merge_unique_reviews(base: List[Dict[str, Any]], extra: List[Dict[str, Any]], max_reviews: int) -> List[Dict[str, Any]]:
    if not extra:
        return base
    seen = {_review_dedupe_key(r) for r in base if _normalize_review_text(r.get("text"))}
    out = list(base)
    for row in extra:
        text = _normalize_review_text(row.get("text"))
        if not text:
            continue
        key = _review_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        row["text"] = text
        out.append(row)
        if len(out) >= max_reviews:
            break
    return out[:max_reviews]


def _fetch_metadata_direct(asin: str, domain: str) -> Dict[str, Any]:
    """
    Best-effort product metadata extraction directly from Amazon HTML.
    """
    product = {"title": f"Amazon Product ({asin})", "thumbnail": "", "rating": "", "reviews_total": 0}
    url = f"https://www.{domain}/dp/{asin}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=20)
        if resp.status_code != 200:
            return product
        html_text = resp.text or ""

        title_text = ""
        rating_text = ""
        reviews_text = ""
        image_url = ""

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_text, "html.parser")
            title_el = soup.select_one("#productTitle")
            if title_el:
                title_text = title_el.get_text(" ", strip=True)
            rating_el = soup.select_one("span[data-hook='rating-out-of-text']") or soup.select_one("#acrPopover")
            if rating_el:
                rating_text = rating_el.get_text(" ", strip=True)
            reviews_el = soup.select_one("#acrCustomerReviewText")
            if reviews_el:
                reviews_text = reviews_el.get_text(" ", strip=True)
            image_el = soup.select_one("#landingImage") or soup.select_one("meta[property='og:image']")
            if image_el:
                image_url = image_el.get("src") or image_el.get("content") or ""

        if not title_text:
            m = re.search(r'id=["\']productTitle["\'][^>]*>\s*(.*?)\s*<', html_text, flags=re.I | re.S)
            if m:
                title_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1))).strip()
        if not rating_text:
            m = re.search(r'([\d\.]+)\s*out of\s*5', html_text, flags=re.I)
            if m:
                rating_text = m.group(1)
        if not reviews_text:
            m = re.search(r'id=["\']acrCustomerReviewText["\'][^>]*>\s*([^<]+)<', html_text, flags=re.I | re.S)
            if m:
                reviews_text = m.group(1)
        if not image_url:
            m = re.search(r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html_text, flags=re.I)
            if m:
                image_url = m.group(1)

        rating_val = ""
        m_rating = re.search(r"([\d\.]+)", str(rating_text))
        if m_rating:
            rating_val = m_rating.group(1)

        reviews_total = 0
        m_reviews = re.search(r"([\d,]+)", str(reviews_text))
        if m_reviews:
            try:
                reviews_total = int(m_reviews.group(1).replace(",", ""))
            except Exception:
                reviews_total = 0

        if title_text:
            product["title"] = title_text
        if image_url:
            product["thumbnail"] = image_url
        if rating_val:
            product["rating"] = rating_val
        if reviews_total > 0:
            product["reviews_total"] = reviews_total
        return product
    except Exception as exc:
        print(f"[WARN] Direct metadata parse failed: {exc}")
        return product


def _fetch_metadata(asin: str, domain: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    product = _fetch_metadata_direct(asin, domain)
    fallback_reviews: List[Dict[str, Any]] = []
    if not SERP_API_KEY or SERP_API_KEY == "YOUR_SERP_API_KEY_HERE" or GoogleSearch is None:
        return product, fallback_reviews
    try:
        params = {
            "engine": "amazon_product",
            "amazon_domain": domain,
            "asin": asin,
            "api_key": SERP_API_KEY,
        }
        result = GoogleSearch(params).get_dict()
        prod = result.get("product_results", {}) or {}
        product["title"] = prod.get("title") or product["title"]
        product["thumbnail"] = prod.get("thumbnail") or product["thumbnail"]
        product["rating"] = prod.get("rating") or product["rating"]
        product["reviews_total"] = prod.get("reviews") or product["reviews_total"]

        seen = set()

        def add_review(text: str, date: str = "") -> int:
            txt = str(text or "").strip()
            if not txt or txt in seen:
                return 0
            seen.add(txt)
            fallback_reviews.append(
                {
                    "text": txt,
                    "date": str(date or ""),
                    "user_name": "unknown",
                    "star_rating": 3,
                    "rating_source": "default",
                }
            )
            return 1

        _extract_reviews_from_result(result, add_review)
    except Exception as exc:
        print(f"[WARN] Metadata fetch failed: {exc}")
    return product, fallback_reviews


def _scrape_amazon_serpapi_reviews(
    asin: str,
    domain: str,
    max_reviews: int,
    max_pages: int,
) -> List[Dict[str, Any]]:
    """
    Fetch paginated Amazon review data from SerpAPI (when configured).
    """
    if not SERP_API_KEY or SERP_API_KEY == "YOUR_SERP_API_KEY_HERE" or GoogleSearch is None:
        return []

    reviews: List[Dict[str, Any]] = []
    seen = set()

    for sort_by in ("recent", "helpful"):
        page = 1
        next_page_token = None
        empty_pages = 0

        while len(reviews) < max_reviews and page <= max_pages:
            try:
                params = {
                    "engine": "amazon_reviews",
                    "amazon_domain": domain,
                    "api_key": SERP_API_KEY,
                    "asin": asin,
                    # Some SerpAPI variants use product_id; keeping both improves compatibility.
                    "product_id": asin,
                    "sort_by": sort_by,
                    "filter_by_star": "all_stars",
                    "language": "en_IN" if domain.endswith(".in") else "en_US",
                }
                if next_page_token:
                    params["next_page_token"] = next_page_token
                else:
                    params["page"] = page

                result = GoogleSearch(params).get_dict()
                error_msg = result.get("error")
                if error_msg:
                    print(f"[WARN] SerpAPI reviews error sort={sort_by} page={page}: {error_msg}")
                    break

                raw_reviews = result.get("reviews")
                if not raw_reviews:
                    reviews_results = result.get("reviews_results")
                    if isinstance(reviews_results, list):
                        raw_reviews = reviews_results
                    elif isinstance(reviews_results, dict):
                        raw_reviews = (
                            reviews_results.get("reviews")
                            or reviews_results.get("results")
                            or reviews_results.get("items")
                        )
                if not raw_reviews:
                    raw_reviews = result.get("top_reviews")
                if not isinstance(raw_reviews, list):
                    raw_reviews = []

                added = 0
                for rev in raw_reviews:
                    if not isinstance(rev, dict):
                        continue
                    text = _normalize_review_text(rev.get("body") or rev.get("text") or rev.get("content") or "")
                    if not text:
                        continue

                    rating_val = rev.get("rating", 3)
                    rating_present = rating_val is not None
                    try:
                        star_rating = int(round(float(rating_val)))
                    except Exception:
                        star_rating = 3

                    review_obj = {
                        "text": text,
                        "date": str(rev.get("date", "")),
                        "star_rating": max(1, min(5, star_rating)),
                        "user_name": str(rev.get("author", "unknown") or "unknown"),
                        "review_id": str(
                            rev.get("review_id")
                            or rev.get("id")
                            or rev.get("position")
                            or ""
                        ),
                        "rating_source": "parsed" if rating_present else "default",
                    }
                    dedupe_key = _review_dedupe_key(review_obj)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    reviews.append(review_obj)
                    added += 1
                    if len(reviews) >= max_reviews:
                        break

                print(f"[INFO] SerpAPI {sort_by} page={page}: +{added} -> total={len(reviews)}")
                if added == 0:
                    empty_pages += 1
                else:
                    empty_pages = 0
                if empty_pages >= 2:
                    break

                pag = result.get("pagination") or result.get("serpapi_pagination") or {}
                next_page_token = pag.get("next_page_token") or result.get("next_page_token")
                page += 1
            except Exception as exc:
                print(f"[WARN] SerpAPI reviews fetch error sort={sort_by} page={page}: {exc}")
                break

    print(f"[INFO] SerpAPI reviews total: {len(reviews)}")
    return reviews[:max_reviews]


def _parse_with_bs4(html_text: str) -> List[Dict[str, Any]]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    parsed: List[Dict[str, Any]] = []
    review_blocks = soup.select(
        "div[data-hook='review'], "
        "li[data-hook='review'], "
        "div.a-section.review.aok-relative, "
        "div[data-cel-widget^='customer_review'], "
        "div[data-hook='review-collapsed']"
    )
    for block in review_blocks:
        body_el = (
            block.select_one("span[data-hook='review-body'] span")
            or block.select_one("span[data-hook='review-body']")
            or block.select_one("div[data-hook='review-collapsed'] span")
            or block.select_one("span.review-text-content span")
            or block.select_one("span.review-text-content")
            or block.select_one(".review-text-content span")
            or block.select_one(".review-text")
        )
        text = _normalize_review_text(body_el.get_text(" ", strip=True) if body_el else "")
        if text.lower() in {"read more", "read less", "report"}:
            text = ""
        if not text:
            continue
        date_el = block.select_one("span[data-hook='review-date']") or block.select_one("span.review-date")
        date = date_el.get_text(strip=True) if date_el else ""
        star_el = block.select_one("i[data-hook='review-star-rating'] span") or block.select_one(
            "i[data-hook='cmps-review-star-rating'] span"
        )
        if not star_el:
            star_el = block.select_one("i.review-rating span")
        if not star_el:
            star_el = block.select_one("span.a-icon-alt")
        star = 3
        if star_el:
            m = re.search(r"([\d\.]+)", star_el.get_text(" ", strip=True))
            if m:
                try:
                    star = int(round(float(m.group(1))))
                except Exception:
                    star = 3
        user_el = block.select_one("span.a-profile-name") or block.select_one("a[data-hook='review-author']")
        user_name = _normalize_review_text(user_el.get_text(strip=True) if user_el else "unknown")
        review_id = (
            block.get("id")
            or block.get("data-review-id")
            or ""
        )
        parsed.append(
            {
                "text": text,
                "date": date,
                "star_rating": star,
                "user_name": user_name,
                "review_id": str(review_id),
                "rating_source": "parsed" if star_el else "default",
            }
        )
    return parsed


def _parse_with_regex(html_text: str) -> List[Dict[str, Any]]:
    # Fallback parser when BeautifulSoup is unavailable.
    parsed: List[Dict[str, Any]] = []
    for match in re.finditer(
        r'<span[^>]+data-hook=["\']review-body["\'][^>]*>(.*?)</span>',
        html_text,
        flags=re.I | re.S,
    ):
        text = re.sub(r"<[^>]+>", " ", match.group(1))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parsed.append(
                {
                    "text": text,
                    "date": "",
                    "star_rating": 3,
                    "user_name": "unknown",
                    "review_id": "",
                    "rating_source": "default",
                }
            )
    return parsed


def _collect_json_reviews(node: Any, out: List[Dict[str, Any]]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_json_reviews(item, out)
        return
    if not isinstance(node, dict):
        return

    review_type = str(node.get("@type", "")).lower()
    body = node.get("reviewBody") or node.get("description") or node.get("text")
    if review_type == "review" or body:
        text = _normalize_review_text(body)
        if text:
            author = node.get("author")
            if isinstance(author, dict):
                author = author.get("name")
            star_val = 3
            rating_data = node.get("reviewRating")
            if isinstance(rating_data, dict):
                rating_raw = rating_data.get("ratingValue") or rating_data.get("rating")
            else:
                rating_raw = node.get("ratingValue") or node.get("rating")
            try:
                star_val = int(round(float(rating_raw)))
            except Exception:
                star_val = 3
            out.append(
                {
                    "text": text,
                    "date": str(node.get("datePublished", "") or ""),
                    "star_rating": max(1, min(5, star_val)),
                    "user_name": _normalize_review_text(author or "unknown"),
                    "review_id": str(node.get("@id") or node.get("identifier") or ""),
                    "rating_source": "parsed",
                }
            )

    for key in ("review", "reviews", "itemListElement", "mainEntity"):
        if key in node:
            _collect_json_reviews(node.get(key), out)


def _parse_with_json_ld(html_text: str) -> List[Dict[str, Any]]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    parsed: List[Dict[str, Any]] = []
    for script in soup.select("script[type='application/ld+json']"):
        payload = script.string or script.get_text(" ", strip=False)
        if not payload:
            continue
        payload = payload.strip()
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except Exception:
            # Sometimes JSON-LD is malformed on marketplace pages.
            continue
        _collect_json_reviews(data, parsed)
    return parsed


def _is_amazon_bot_page(html_text: str) -> bool:
    t = str(html_text or "").lower()
    markers = [
        "robot check",
        "captcha",
        "enter the characters you see below",
        "sorry, we just need to make sure you're not a robot",
        "type the characters you see in this image",
        "to discuss automated access to amazon data please contact",
        "automated access to amazon data",
        "api-services-support@amazon.com",
    ]
    return any(m in t for m in markers)


def _parse_with_json_snippets(html_text: str) -> List[Dict[str, Any]]:
    """
    Last-resort extraction from embedded JSON-like payloads.
    """
    parsed: List[Dict[str, Any]] = []
    seen = set()
    for match in re.finditer(r'"reviewText"\s*:\s*"((?:\\.|[^"\\])*)"', html_text, flags=re.I | re.S):
        raw = match.group(1)
        try:
            text = bytes(raw, "utf-8").decode("unicode_escape", errors="ignore")
        except Exception:
            text = raw
        text = htmllib.unescape(re.sub(r"\s+", " ", text)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parsed.append(
            {
                "text": text,
                "date": "",
                "star_rating": 3,
                "user_name": "unknown",
                "review_id": "",
                "rating_source": "default",
            }
        )
    return parsed


def _discover_review_bases(session: requests.Session, asin: str, domain: str) -> List[str]:
    bases: List[str] = []
    product_url = f"https://www.{domain}/dp/{asin}"
    try:
        resp = session.get(product_url, headers=_headers(), timeout=20)
        html_text = resp.text or ""
        if BeautifulSoup is not None and html_text:
            soup = BeautifulSoup(html_text, "html.parser")
            for sel in ("a[data-hook='see-all-reviews-link-foot']", "a[data-hook='see-all-reviews-link']"):
                anchor = soup.select_one(sel)
                if not anchor:
                    continue
                href = anchor.get("href")
                if not href:
                    continue
                bases.append(urljoin(f"https://www.{domain}", href))
        if html_text:
            pattern = rf"/product-reviews/{re.escape(asin)}[^\"'\s>]*"
            for m in re.finditer(pattern, html_text, flags=re.I):
                bases.append(urljoin(f"https://www.{domain}", m.group(0)))
    except Exception:
        pass

    bases.extend(
        [
            f"https://www.{domain}/product-reviews/{asin}",
            f"https://www.{domain}/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt",
            f"https://www.{domain}/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm",
        ]
    )

    out: List[str] = []
    seen = set()
    for base in bases:
        parsed = urlparse(str(base))
        path = parsed.path or f"/product-reviews/{asin}"
        canonical = f"https://www.{domain}{path}"
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _build_review_page_url(base: str, page: int, sort_by: str, filter_by_star: str) -> str:
    parsed = urlparse(base)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(
        {
            "ie": query.get("ie", "UTF8"),
            "reviewerType": "all_reviews",
            "sortBy": sort_by,
            "pageNumber": str(page),
            "filterByStar": filter_by_star,
        }
    )
    return parsed._replace(query=urlencode(query)).geturl()


def _scrape_amazon_direct(asin: str, domain: str, max_reviews: int, max_pages: int) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []
    seen = set()
    session = requests.Session()
    review_bases = _discover_review_bases(session, asin, domain)
    strategy_list = [
        {"name": "helpful", "sort_by": "helpful", "filter_by_star": "all_stars"},
        {"name": "recent", "sort_by": "recent", "filter_by_star": "all_stars"},
        {"name": "critical", "sort_by": "recent", "filter_by_star": "critical"},
        {"name": "positive", "sort_by": "recent", "filter_by_star": "positive"},
        {"name": "one_star", "sort_by": "recent", "filter_by_star": "one_star"},
        {"name": "five_star", "sort_by": "recent", "filter_by_star": "five_star"},
    ]

    for strategy in strategy_list:
        page = 1
        consecutive_empty = 0
        consecutive_errors = 0
        while len(reviews) < max_reviews and page <= max_pages:
            try:
                resp = None
                status_code = None
                for base in review_bases:
                    candidate = _build_review_page_url(
                        base=base,
                        page=page,
                        sort_by=strategy["sort_by"],
                        filter_by_star=strategy["filter_by_star"],
                    )
                    r = session.get(candidate, headers=_headers(), timeout=20)
                    status_code = r.status_code
                    if r.status_code == 200:
                        resp = r
                        break

                if resp is None:
                    if status_code == 503:
                        print(f"[WARN] Amazon 503 - backing off sort={strategy['name']} page={page}")
                        time.sleep(5)
                        page += 1
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            break
                        continue
                    print(f"[WARN] HTTP {status_code} sort={strategy['name']} page={page}")
                    page += 1
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        break
                    continue

                if _is_amazon_bot_page(resp.text):
                    print(f"[WARN] Amazon bot challenge detected sort={strategy['name']} page={page}")
                    consecutive_errors += 1
                    time.sleep(2)
                    if consecutive_errors >= 2:
                        break
                    page += 1
                    continue

                consecutive_errors = 0
                parsed = _parse_with_bs4(resp.text)
                if not parsed:
                    parsed = _parse_with_json_ld(resp.text)
                if not parsed:
                    parsed = _parse_with_regex(resp.text)
                if not parsed:
                    parsed = _parse_with_json_snippets(resp.text)

                added = 0
                for row in parsed:
                    text = _normalize_review_text(row.get("text"))
                    if not text:
                        continue
                    review_obj = {
                        "text": text,
                        "date": str(row.get("date", "")),
                        "star_rating": int(row.get("star_rating", 3) or 3),
                        "user_name": str(row.get("user_name", "unknown") or "unknown"),
                        "review_id": str(row.get("review_id", "") or ""),
                        "rating_source": str(row.get("rating_source", "default")),
                    }
                    dedupe_key = _review_dedupe_key(review_obj)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    reviews.append(review_obj)
                    added += 1
                    if len(reviews) >= max_reviews:
                        break

                if added == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0

                if consecutive_empty >= 2:
                    print(f"[INFO] No more reviews sort={strategy['name']} at page={page}")
                    break

                print(f"[INFO] Direct sort={strategy['name']}: +{added} -> total={len(reviews)}")
                time.sleep(random.uniform(0.2, 0.6))
                page += 1
            except Exception as exc:
                print(f"[WARN] Direct scrape error sort={strategy['name']}: {exc}")
                page += 1
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                continue

    print(f"[INFO] Direct scrape total: {len(reviews)} reviews for ASIN={asin}")
    return reviews[:max_reviews]


def _scrape_amazon_scrapingdog(asin: str, domain: str, max_reviews: int, max_pages: int) -> List[Dict[str, Any]]:
    if not SCRAPINGDOG_KEY or SCRAPINGDOG_KEY == "YOUR_SCRAPINGDOG_KEY_HERE":
        return []
    reviews: List[Dict[str, Any]] = []
    seen = set()
    country = domain.replace("amazon.", "")

    for sort in ("helpful", "recent"):
        page = 1
        consecutive_empty = 0
        while len(reviews) < max_reviews and page <= max_pages:
            try:
                resp = requests.get(
                    "https://api.scrapingdog.com/amazon/reviews",
                    params={
                        "api_key": SCRAPINGDOG_KEY,
                        "asin": asin,
                        "country": country,
                        "page": page,
                        "sort_by": sort,
                    },
                    timeout=30,
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                raw_reviews = data.get("reviews", []) or []
                added = 0
                for rev in raw_reviews:
                    text = _normalize_review_text(rev.get("body") or rev.get("text") or rev.get("review_body") or "")
                    if not text:
                        continue
                    rating_present = rev.get("rating", None) is not None
                    review_obj = {
                        "text": text,
                        "date": str(rev.get("date", "")),
                        "star_rating": int(float(rev.get("rating", 3) or 3)),
                        "user_name": str(rev.get("author", "unknown") or "unknown"),
                        "review_id": str(rev.get("review_id") or rev.get("id") or ""),
                        "rating_source": "parsed" if rating_present else "default",
                    }
                    dedupe_key = _review_dedupe_key(review_obj)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    reviews.append(review_obj)
                    added += 1
                    if len(reviews) >= max_reviews:
                        break
                print(f"[INFO] ScrapingDog sort={sort} page={page}: +{added} -> total={len(reviews)}")
                if added == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                if consecutive_empty >= 2:
                    break
                if not data.get("next_page", False):
                    break
                page += 1
            except Exception as exc:
                print(f"[WARN] ScrapingDog error: {exc}")
                break
    print(f"[INFO] ScrapingDog total: {len(reviews)} reviews")
    return reviews[:max_reviews]


def _scrape_flipkart(product_url: str, max_reviews: int, max_pages: int) -> List[Dict[str, Any]]:
    if not RAPIDAPI_KEY or RAPIDAPI_KEY == "YOUR_RAPIDAPI_KEY_HERE":
        return []
    reviews: List[Dict[str, Any]] = []
    seen = set()
    page = 1
    while len(reviews) < max_reviews and page <= max_pages:
        try:
            resp = requests.get(
                "https://flipkart-reviews.p.rapidapi.com/reviews",
                params={"url": product_url, "page": page},
                headers={
                    "X-RapidAPI-Key": RAPIDAPI_KEY,
                    "X-RapidAPI-Host": "flipkart-reviews.p.rapidapi.com",
                },
                timeout=30,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            raw_reviews = data.get("reviews") or data.get("data") or []
            added = 0
            for rev in raw_reviews:
                text = _normalize_review_text(
                    rev.get("review") or rev.get("text") or rev.get("body") or rev.get("content") or ""
                )
                if not text:
                    continue
                rating_present = rev.get("rating", None) is not None
                review_obj = {
                    "text": text,
                    "date": str(rev.get("date", "")),
                    "star_rating": int(float(rev.get("rating", 3) or 3)),
                    "user_name": str(rev.get("user", "unknown") or "unknown"),
                    "review_id": str(rev.get("review_id") or rev.get("id") or ""),
                    "rating_source": "parsed" if rating_present else "default",
                }
                dedupe_key = _review_dedupe_key(review_obj)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                reviews.append(review_obj)
                added += 1
                if len(reviews) >= max_reviews:
                    break
            print(f"[INFO] Flipkart API page={page}: +{added} -> total={len(reviews)}")
            if not data.get("next_page", False) or added == 0:
                break
            page += 1
        except Exception as exc:
            print(f"[WARN] Flipkart API error page={page}: {exc}")
            break
    print(f"[INFO] Flipkart total: {len(reviews)} reviews")
    return reviews[:max_reviews]


def _load_cached_reviews(max_reviews: int) -> List[Dict[str, Any]]:
    if not os.path.exists(ANALYSIS_RESULTS_PATH):
        return []
    try:
        df = pd.read_csv(ANALYSIS_RESULTS_PATH).fillna("")
        out = []
        for _, row in df.iterrows():
            text = str(row.get("review", "")).strip()
            if not text:
                continue
            raw_star = row.get("star_rating", 3)
            try:
                star_rating = int(round(float(raw_star)))
            except Exception:
                star_rating = 3
            star_rating = max(1, min(5, star_rating))
            rating_source = str(row.get("rating_source", "default") or "default").strip().lower()
            if rating_source not in {"parsed", "default"}:
                rating_source = "default"
            out.append(
                {
                    "text": text,
                    "date": str(row.get("date", "")),
                    "star_rating": star_rating,
                    "user_name": str(row.get("user_name", "unknown") or "unknown"),
                    "review_id": str(row.get("review_id", "") or ""),
                    "rating_source": rating_source,
                }
            )
            if len(out) >= max_reviews:
                break
        return out
    except Exception:
        return []


def fetch_reviews_multi_page(
    url_or_asin: str,
    target_domain: str = "amazon.com",
    pages: int = 3,
    max_reviews: int = MAX_REVIEWS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch reviews from Amazon or Flipkart with graceful fallback.
    """
    max_reviews = _clamp_int(max_reviews, 1, MAX_REVIEWS, MAX_REVIEWS)
    max_pages = _clamp_int(pages, 1, 1000, 200)
    raw = str(url_or_asin or "").strip()

    if is_flipkart_url(raw):
        print("[INFO] Detected Flipkart - using RapidAPI")
        reviews = _scrape_flipkart(raw, max_reviews=max_reviews, max_pages=max_pages)
        return reviews, {"title": "Flipkart Product", "thumbnail": "", "rating": "", "reviews_total": len(reviews)}

    if not is_amazon_url(raw):
        return [], {"title": "Unknown Product", "thumbnail": "", "rating": "", "reviews_total": 0}

    asin = extract_asin(raw)
    domain = extract_domain(raw) if "amazon." in raw.lower() else target_domain
    print(f"[INFO] Fetching product metadata for ASIN={asin} ...")
    product_details, fallback_reviews = _fetch_metadata(asin, domain)
    available_reviews = _safe_int(product_details.get("reviews_total"), 0)
    target_reviews = max_reviews
    if available_reviews > 0:
        # Scan as much as possible, bounded by global safety cap.
        target_reviews = min(max_reviews, available_reviews)
    print(f"[INFO] Product : {product_details.get('title', '?')[:70]}")
    print(f"[INFO] Ratings : {product_details.get('rating', '')} ({product_details.get('reviews_total', 0)} reviews)")
    print(f"[INFO] Target scan volume: {target_reviews} reviews")

    reviews: List[Dict[str, Any]] = []
    if SCRAPINGDOG_KEY and SCRAPINGDOG_KEY != "YOUR_SCRAPINGDOG_KEY_HERE":
        print("[INFO] Using ScrapingDog API (premium scraping) ...")
        reviews = _scrape_amazon_scrapingdog(
            asin,
            domain,
            max_reviews=target_reviews,
            max_pages=max_pages,
        )
        if not reviews:
            print("[INFO] ScrapingDog returned 0 - falling back to direct scraping ...")
    else:
        print("[INFO] SCRAPINGDOG_KEY not set - using direct scraping (free) ...")

    if len(reviews) < target_reviews:
        serp_reviews = _scrape_amazon_serpapi_reviews(
            asin,
            domain,
            max_reviews=target_reviews - len(reviews),
            max_pages=max_pages,
        )
        reviews = _merge_unique_reviews(reviews, serp_reviews, target_reviews)

    if len(reviews) < target_reviews:
        direct = _scrape_amazon_direct(
            asin,
            domain,
            max_reviews=target_reviews - len(reviews),
            max_pages=max_pages,
        )
        reviews = _merge_unique_reviews(reviews, direct, target_reviews)

    if len(reviews) < target_reviews and fallback_reviews:
        print("[WARN] Adding metadata snippet reviews to improve coverage ...")
        reviews = _merge_unique_reviews(reviews, fallback_reviews, target_reviews)

    if not reviews:
        local_cache = _load_cached_reviews(max_reviews=target_reviews)
        if local_cache:
            print(
                f"[WARN] Live fetch unavailable. Using local cache ({len(local_cache)} reviews). "
                "Confidence may be reduced for low sample size."
            )
            reviews = local_cache

    if not reviews:
        print("[WARN] Could not fetch any reviews at all.")
        print("       Tip: set SERP_API_KEY or SCRAPINGDOG_KEY in .env for large pulls.")

    return reviews[:target_reviews], product_details


def _parse_timestamp(date_text: str) -> pd.Timestamp:
    if not str(date_text or "").strip():
        return pd.Timestamp.utcnow().tz_localize(None)
    if parse_date is not None:
        try:
            parsed = parse_date(str(date_text))
            if parsed is not None:
                return pd.Timestamp(parsed).tz_localize(None)
        except Exception:
            pass
    try:
        ts = pd.to_datetime(str(date_text), errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts)
    except Exception:
        pass
    return pd.Timestamp.utcnow().tz_localize(None)


def detect_reviewer_behavior_anomalies(df_reviews: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        user_name: {
          velocity_flag: bool,
          dormant_burst: bool,
          rating_pattern_flag: bool,
          behavior_score: float [0,1],
        }
      }
    """
    if detect_reviewer_anomalies is None or df_reviews.empty:
        return {}

    required = {"user_id", "product_id", "timestamp", "rating"}
    normalized = df_reviews.copy()
    if not required.issubset(normalized.columns):
        # Backward-compatible schema mapping for callers that provide UI fields.
        user_col = "user_id" if "user_id" in normalized.columns else ("user_name" if "user_name" in normalized.columns else None)
        product_col = "product_id" if "product_id" in normalized.columns else ("asin" if "asin" in normalized.columns else None)
        ts_col = "timestamp" if "timestamp" in normalized.columns else ("date" if "date" in normalized.columns else None)
        rating_col = "rating" if "rating" in normalized.columns else ("star_rating" if "star_rating" in normalized.columns else None)
        if user_col is None or ts_col is None:
            return {}
        normalized = pd.DataFrame(
            {
                "user_id": normalized[user_col].fillna("unknown").astype(str),
                "product_id": (
                    normalized[product_col].fillna("unknown_product").astype(str)
                    if product_col is not None
                    else pd.Series(["unknown_product"] * len(normalized))
                ),
                "timestamp": normalized[ts_col],
                "rating": (
                    pd.to_numeric(normalized[rating_col], errors="coerce").fillna(3).astype(int)
                    if rating_col is not None
                    else pd.Series([3] * len(normalized))
                ),
            }
        )
    else:
        normalized = normalized[["user_id", "product_id", "timestamp", "rating"]].copy()

    try:
        res = detect_reviewer_anomalies(normalized)
    except Exception as exc:
        print(f"[WARN] Behavior detector failed: {exc}")
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    if res is None or res.empty:
        return out

    for _, row in res.iterrows():
        user = str(row.get("user_id", "unknown"))
        flags = row.get("flags", []) or []
        flag_types = {str(f.get("flag_type", "")) for f in flags if isinstance(f, dict)}
        out[user] = {
            "velocity_flag": "HIGH_VELOCITY" in flag_types,
            "dormant_burst": "DORMANT_BURST" in flag_types,
            "rating_pattern_flag": "RATING_EXTREMISM" in flag_types,
            "behavior_score": round(float(row.get("risk_score", 0)) / 100.0, 3),
        }
    return out


def detect_sentiment_rating_mismatch_for_ui(texts: List[str], star_ratings: List[float]) -> List[Dict[str, Any]]:
    if detect_text_rating_mismatch is None:
        return [{"sentiment_score": 0.0, "normalized_rating": 0.0, "discrepancy": 0.0, "is_mismatch": False} for _ in texts]

    df = pd.DataFrame(
        {
            "review_id": list(range(len(texts))),
            "text": texts,
            "rating": [float(r if pd.notna(r) else 3) for r in star_ratings],
        }
    )
    try:
        out = detect_text_rating_mismatch(df)
        result = []
        for _, row in out.iterrows():
            result.append(
                {
                    "sentiment_score": float(row.get("sentiment_compound", 0.0)),
                    "normalized_rating": float(row.get("normalized_rating", 0.0)),
                    "discrepancy": float(row.get("discrepancy", 0.0)),
                    "is_mismatch": bool(row.get("is_mismatch", False)),
                    "mismatch_type": str(row.get("mismatch_type", "NORMAL")),
                }
            )
        return result
    except Exception as exc:
        print(f"[WARN] Mismatch detector failed: {exc}")
        return [{"sentiment_score": 0.0, "normalized_rating": 0.0, "discrepancy": 0.0, "is_mismatch": False} for _ in texts]


def detect_sentiment_rating_mismatch(texts: List[str], star_ratings: List[float]) -> List[Dict[str, Any]]:
    """
    Backward-compatible alias used by older tests/integrations.
    """
    return detect_sentiment_rating_mismatch_for_ui(texts, star_ratings)


def detect_review_rings_for_ui(df_graph: pd.DataFrame, current_asin: str) -> Dict[str, Any]:
    norm_users = pd.Series(dtype=str)
    known_user_count = 0
    if "user_id" in df_graph.columns:
        norm_users = df_graph["user_id"].fillna("").astype(str).str.strip()
        known_users = norm_users[~norm_users.str.lower().isin({"", "unknown", "nan", "none"})]
        known_user_count = int(known_users.nunique())

    result = {
        "suspicious_clusters": [],
        "graph_summary": {
            "total_users": known_user_count,
            "total_products": int(df_graph["product_id"].nunique()) if "product_id" in df_graph.columns else 0,
            "total_edges": 0,
        },
        "overall_graph_score": 0.0,
    }
    if df_graph.empty or detect_review_rings_module is None:
        return result

    if known_user_count < 3:
        result["note"] = "Insufficient unique reviewer identities for reliable network analysis."
        return result

    try:
        clusters = detect_review_rings_module(df_graph[["user_id", "product_id", "timestamp", "rating"]].copy())
        if clusters:
            transformed = []
            for c in clusters:
                transformed.append(
                    {
                        "users": c.get("users", []),
                        "shared_products": c.get("shared_products", []),
                        "cluster_size": int(c.get("cluster_size", 0)),
                        "cluster_score": round(float(c.get("risk_score", 0)) / 100.0, 3),
                        "detection_method": "module",
                    }
                )
            result["suspicious_clusters"] = transformed
            max_score = max((c["cluster_score"] for c in transformed), default=0.0)
            result["overall_graph_score"] = round(float(max_score), 3)
            result["graph_summary"]["total_edges"] = int(
                sum(int(c.get("cluster_size", 0)) for c in transformed)
            )
        if result["graph_summary"]["total_products"] <= 1:
            result["note"] = "Only one product analysed - graph analysis needs multi-product data for meaningful ring detection."
        return result
    except Exception as exc:
        print(f"[WARN] Review ring detector failed: {exc}")
        result["note"] = f"Graph detection fallback used for ASIN={current_asin}"
        return result


def detect_review_rings(df_reviews: pd.DataFrame, current_asin: str) -> Dict[str, Any]:
    """
    Backward-compatible wrapper that accepts legacy column names.
    """
    if df_reviews is None or df_reviews.empty:
        return {
            "suspicious_clusters": [],
            "graph_summary": {"total_users": 0, "total_products": 0, "total_edges": 0},
            "overall_graph_score": 0.0,
        }

    user_col = "user_id" if "user_id" in df_reviews.columns else ("user_name" if "user_name" in df_reviews.columns else None)
    product_col = "product_id" if "product_id" in df_reviews.columns else ("asin" if "asin" in df_reviews.columns else None)
    ts_col = "timestamp" if "timestamp" in df_reviews.columns else ("date" if "date" in df_reviews.columns else None)
    rating_col = "rating" if "rating" in df_reviews.columns else ("star_rating" if "star_rating" in df_reviews.columns else None)
    if user_col is None:
        return {
            "suspicious_clusters": [],
            "graph_summary": {"total_users": 0, "total_products": 0, "total_edges": 0},
            "overall_graph_score": 0.0,
            "note": "Missing user identifiers for graph analysis.",
        }

    transformed = pd.DataFrame(
        {
            "user_id": df_reviews[user_col].fillna("unknown").astype(str),
            "product_id": (
                df_reviews[product_col].fillna(str(current_asin)).astype(str)
                if product_col is not None
                else pd.Series([str(current_asin)] * len(df_reviews))
            ),
            "timestamp": (
                df_reviews[ts_col].apply(_parse_timestamp)
                if ts_col is not None
                else pd.Series([pd.Timestamp.utcnow().tz_localize(None)] * len(df_reviews))
            ),
            "rating": (
                pd.to_numeric(df_reviews[rating_col], errors="coerce").fillna(3).astype(int)
                if rating_col is not None
                else pd.Series([3] * len(df_reviews))
            ),
        }
    )
    return detect_review_rings_for_ui(transformed, current_asin=current_asin)


def compute_advanced_fraud_score(
    ml_score: float,
    behavior_score: float,
    mismatch_score: float,
    graph_score: float,
) -> float:
    """
    Weighted combination of all detection signals in [0,1].
    """
    ml_score = min(max(float(ml_score), 0.0), 1.0)
    behavior_score = min(max(float(behavior_score), 0.0), 1.0)
    mismatch_score = min(max(float(mismatch_score), 0.0), 1.0)
    graph_score = min(max(float(graph_score), 0.0), 1.0)
    return (0.50 * ml_score) + (0.20 * behavior_score) + (0.20 * mismatch_score) + (0.10 * graph_score)


def analyze_product(asin: str, pages: int = 50, max_reviews: int = MAX_REVIEWS) -> Tuple[Dict[str, Any], pd.DataFrame]:
    target_reviews = _clamp_int(max_reviews, 1, MAX_REVIEWS, MAX_REVIEWS)
    reviews, product_details = fetch_reviews_multi_page(asin, pages=pages, max_reviews=target_reviews)
    if not reviews:
        empty_summary = {
            "total_reviews": 0,
            "fake_percent": 0.0,
            "genuine_percent": 0.0,
            "avg_confidence": 0.0,
            "avg_fake_probability": 0.0,
            "product_title": product_details.get("title", "Unknown Product"),
            "product_image": product_details.get("thumbnail", ""),
            "product_rating": product_details.get("rating", ""),
            "product_reviews_total": product_details.get("reviews_total", 0),
            "requested_review_limit": target_reviews,
            "advanced_analysis": {
                "behavior_anomalies": {},
                "sentiment_mismatches": [],
                "review_rings": {
                    "suspicious_clusters": [],
                    "graph_summary": {"total_users": 0, "total_products": 0, "total_edges": 0},
                    "overall_graph_score": 0.0,
                },
                "combined_fraud_score": 0.0,
            },
        }
        return empty_summary, pd.DataFrame()

    texts = [str(r.get("text", "")).strip() for r in reviews if str(r.get("text", "")).strip()]
    reviews = [r for r in reviews if str(r.get("text", "")).strip()]
    if not texts:
        return {
            "total_reviews": 0,
            "fake_percent": 0.0,
            "genuine_percent": 0.0,
            "avg_confidence": 0.0,
            "avg_fake_probability": 0.0,
            "product_title": product_details.get("title", "Unknown Product"),
            "product_image": product_details.get("thumbnail", ""),
            "product_rating": product_details.get("rating", ""),
            "product_reviews_total": product_details.get("reviews_total", 0),
            "requested_review_limit": target_reviews,
            "advanced_analysis": {},
        }, pd.DataFrame()

    print(f"[INFO] Running inference on {len(texts)} reviews ...")
    labels, confidences, fake_probs = predict_reviews_batch(texts)

    results_data: List[Dict[str, Any]] = []
    fake_count = 0
    total_confidence = 0.0
    total_fake_prob = 0.0

    for i, review in enumerate(reviews):
        label = labels[i]
        confidence = confidences[i]
        fake_probability = fake_probs[i]
        explanation = explain_review(review["text"], label)
        if label == "Fake":
            fake_count += 1
        total_confidence += confidence
        total_fake_prob += fake_probability
        results_data.append(
            {
                "review": review["text"],
                "user_name": review.get("user_name", "unknown") or "unknown",
                "review_id": review.get("review_id", "") or "",
                "date": review.get("date", ""),
                "prediction": label,
                "confidence": round(confidence, 2),
                "fake_probability": round(fake_probability, 2),
                "explanation": explanation,
                "star_rating": int(review.get("star_rating", 3) or 3),
                "rating_source": str(review.get("rating_source", "default")),
            }
        )

    total = len(results_data)
    fake_pct = round((fake_count / total) * 100.0, 1) if total else 0.0
    avg_conf = round(total_confidence / total, 2) if total else 0.0
    avg_fake = round(total_fake_prob / total, 2) if total else 0.0

    df_results = pd.DataFrame(results_data)
    try:
        # Keep star/rating_source in cache so fallback runs can still power advanced insights.
        df_results.to_csv(ANALYSIS_RESULTS_PATH, index=False)
    except Exception as exc:
        print(f"[WARN] Could not write {ANALYSIS_RESULTS_PATH}: {exc}")

    advanced_analysis: Dict[str, Any] = {
        "behavior_anomalies": {},
        "sentiment_mismatches": [],
        "review_rings": {
            "suspicious_clusters": [],
            "graph_summary": {
                "total_users": int(df_results["user_name"].nunique()) if "user_name" in df_results else 0,
                "total_products": 1,
                "total_edges": total,
            },
            "overall_graph_score": 0.0,
        },
        "combined_fraud_score": round(avg_fake, 1),
        "data_quality": {},
    }

    try:
        product_id = extract_asin(asin) if is_amazon_url(asin) else str(asin)
        user_series = df_results["user_name"].fillna("unknown").astype(str).str.strip()
        known_user_series = user_series[~user_series.str.lower().isin({"unknown", "", "none", "nan"})]
        user_identity_coverage = float(len(known_user_series) / max(len(user_series), 1))
        rating_source_series = df_results["rating_source"].fillna("default").astype(str).str.lower()
        rating_coverage = float((rating_source_series == "parsed").mean()) if len(rating_source_series) else 0.0
        sample_size_warning = total < 20
        sentiment_reliable = bool(rating_coverage >= 0.6 and total >= 10)
        network_reliable = bool(user_identity_coverage >= 0.6 and known_user_series.nunique() >= 3 and total >= 10)

        graph_df = pd.DataFrame(
            {
                "user_id": user_series,
                "product_id": [product_id] * total,
                "timestamp": [_parse_timestamp(d) for d in df_results["date"].tolist()],
                "rating": pd.to_numeric(df_results.get("star_rating", 3), errors="coerce").fillna(3).astype(int),
            }
        )

        behavior_results = detect_reviewer_behavior_anomalies(graph_df)
        advanced_analysis["behavior_anomalies"] = behavior_results

        star_ratings = pd.to_numeric(df_results.get("star_rating", 3), errors="coerce").fillna(3).tolist()
        mismatch_results = (
            detect_sentiment_rating_mismatch_for_ui(df_results["review"].tolist(), star_ratings)
            if sentiment_reliable
            else []
        )
        advanced_analysis["sentiment_mismatches"] = mismatch_results

        ring_results = detect_review_rings_for_ui(graph_df, current_asin=product_id)
        if not network_reliable and not ring_results.get("note"):
            ring_results["note"] = "Insufficient reviewer identity quality for reliable network analysis."
        advanced_analysis["review_rings"] = ring_results

        advanced_analysis["data_quality"] = {
            "sample_size_warning": sample_size_warning,
            "sample_size": int(total),
            "rating_coverage": round(rating_coverage, 3),
            "user_identity_coverage": round(user_identity_coverage, 3),
            "sentiment_reliable": sentiment_reliable,
            "network_reliable": network_reliable,
        }

        avg_behavior = (
            float(sum(v.get("behavior_score", 0.0) for v in behavior_results.values()) / max(len(behavior_results), 1))
            if behavior_results
            else 0.0
        )
        avg_mismatch = (
            float(sum(1 for m in mismatch_results if m.get("is_mismatch")) / max(len(mismatch_results), 1))
            if mismatch_results
            else 0.0
        )
        graph_score = float(ring_results.get("overall_graph_score", 0.0)) if network_reliable else 0.0
        ml_score = avg_fake / 100.0
        combined = compute_advanced_fraud_score(
            ml_score=ml_score,
            behavior_score=avg_behavior,
            mismatch_score=avg_mismatch,
            graph_score=graph_score,
        )
        advanced_analysis["combined_fraud_score"] = round(combined * 100.0, 1)
    except Exception as exc:
        print(f"[WARN] Advanced analysis partial failure: {exc}")
        advanced_analysis["error"] = str(exc)

    product_total = _safe_int(product_details.get("reviews_total", 0), 0)
    coverage_ratio = float(total / product_total) if product_total > 0 else None
    coverage_warning = None
    if coverage_ratio is not None and coverage_ratio < 0.25:
        coverage_warning = (
            f"Fetched {total} textual reviews out of ~{product_total} ratings. "
            "This can happen when many ratings have no text or sources are rate-limited."
        )

    summary = {
        "total_reviews": int(total),
        "fake_percent": float(fake_pct),
        "genuine_percent": float(round(100.0 - fake_pct, 1)),
        "avg_confidence": float(avg_conf),
        "avg_fake_probability": float(avg_fake),
        "product_title": product_details.get("title", "Unknown Product"),
        "product_image": product_details.get("thumbnail", ""),
        "product_rating": product_details.get("rating", ""),
        "product_reviews_total": product_total,
        "review_coverage_ratio": round(coverage_ratio, 4) if coverage_ratio is not None else None,
        "coverage_warning": coverage_warning,
        "requested_review_limit": target_reviews,
        "advanced_analysis": advanced_analysis,
    }
    print(f"[INFO] Done - {total} reviews, {fake_pct:.1f}% fake.")
    return summary, df_results.drop(columns=["star_rating", "rating_source"], errors="ignore")


if __name__ == "__main__":
    sample_input = "B0C7M9PX4N"
    summary_obj, df_obj = analyze_product(sample_input, pages=5)
    print(pd.Series(summary_obj))
    print(df_obj.head(5).to_string(index=False))
