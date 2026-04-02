from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

# Scraper-only runs do not need ML model loading from analyzer.py.
os.environ.setdefault("ANALYZER_SKIP_MODEL_LOAD", "1")

import analyzer

PROJECT_DEFAULTS: Dict[str, Any] = {
    # Edit these once for your project and run: python scrape_reviews_dataset.py
    "target": "https://www.amazon.in/dp/B08N5WRWNW",
    "output": "scraped_training_reviews.csv",
    "debug_json": "scrape_debug_report.json",
    "pages": 50,
    "max_reviews": 10000,
    "min_required_reviews": 50,
    "target_domain": "amazon.in",
    "min_words": 1,
    "allow_cache": True,
}

RAPIDAPI_AMAZON_HOST = (
    os.getenv("RAPIDAPI_AMAZON_HOST")
    or "real-time-amazon-data.p.rapidapi.com"
).strip()
RAPIDAPI_AMAZON_ENDPOINT = (
    os.getenv("RAPIDAPI_AMAZON_ENDPOINT")
    or "https://real-time-amazon-data.p.rapidapi.com/top-product-reviews"
).strip()
RAPIDAPI_AMAZON_REVIEWS_ENDPOINT = (
    os.getenv("RAPIDAPI_AMAZON_REVIEWS_ENDPOINT")
    or "https://real-time-amazon-data.p.rapidapi.com/product-reviews"
).strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_text_for_training(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_user_id(value: Any) -> str:
    user = _normalize_text(value)
    if user.lower() in {"", "unknown", "none", "nan", "null", "-", "--", "n/a"}:
        return "unknown"
    return user


def _safe_rating(value: Any, default: int = 3) -> int:
    try:
        rating = int(round(float(value)))
    except Exception:
        rating = int(default)
    return max(1, min(5, rating))


def _to_timestamp_iso(date_text: Any) -> str:
    raw = _normalize_text(date_text)
    if not raw:
        return ""
    try:
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
        if pd.isna(ts):
            return ""
        return pd.Timestamp(ts).tz_convert(None).isoformat()
    except Exception:
        return ""


def _tag_reviews(reviews: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for review in reviews or []:
        if not isinstance(review, dict):
            continue
        row = dict(review)
        row["_source"] = source_name
        out.append(row)
    return out


def _record_source(
    diagnostics: Dict[str, Any],
    source: str,
    started: float,
    status: str,
    count: int = 0,
    error: str = "",
    extra: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "source": source,
        "status": status,
        "count": int(count),
        "duration_sec": round(float(time.time() - started), 3),
        "error": str(error or ""),
    }
    if extra:
        payload.update(extra)
    diagnostics["sources"].append(payload)


def _run_with_captured_stdout(func) -> Tuple[Any, str]:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        result = func()
    logs = capture.getvalue()
    if logs:
        # Replay captured logs so CLI behavior remains transparent.
        print(logs, end="", flush=True)
    return result, logs


def _get_serp_api_key() -> str:
    getter = getattr(analyzer, "_get_serp_api_key", None)
    if callable(getter):
        return str(getter() or "").strip()
    return (os.getenv("SERP_API_KEY") or getattr(analyzer, "SERP_API_KEY", "") or "").strip()


def _get_rapidapi_key() -> str:
    getter = getattr(analyzer, "_get_rapidapi_key", None)
    if callable(getter):
        return str(getter() or "").strip()
    return (os.getenv("RAPIDAPI_KEY") or getattr(analyzer, "RAPIDAPI_KEY", "") or "").strip()


def _get_scrapingdog_key() -> str:
    getter = getattr(analyzer, "_get_scrapingdog_key", None)
    if callable(getter):
        return str(getter() or "").strip()
    return (
        os.getenv("SCRAPINGDOG_KEY")
        or os.getenv("SCRAPINGDOG_API_KEY")
        or getattr(analyzer, "SCRAPINGDOG_KEY", "")
        or ""
    ).strip()


def _probe_serpapi_amazon_reviews(asin: str, domain: str) -> Dict[str, Any]:
    serp_api_key = _get_serp_api_key()
    if (
        not serp_api_key
        or serp_api_key == "YOUR_SERP_API_KEY_HERE"
        or analyzer.GoogleSearch is None
    ):
        return {
            "enabled": False,
            "ok": False,
            "error": "SERP_API_KEY missing or SerpAPI SDK unavailable.",
            "hint": "Set SERP_API_KEY to enable SerpAPI scraping.",
        }

    params = {
        "engine": "amazon_reviews",
        "amazon_domain": domain,
        "api_key": serp_api_key,
        "asin": asin,
        "product_id": asin,
        "page": 1,
    }
    try:
        result = analyzer.GoogleSearch(params).get_dict()
        error_msg = str(result.get("error") or "").strip()
        if error_msg:
            hint = ""
            if "Unsupported `amazon_reviews` search engine" in error_msg:
                hint = (
                    "Your SerpAPI account/region does not support engine=amazon_reviews. "
                    "Use direct scraping/ScrapingDog or switch to a supported SerpAPI plan."
                )
            return {"enabled": True, "ok": False, "error": error_msg, "hint": hint}
        return {"enabled": True, "ok": True, "error": "", "hint": ""}
    except Exception as exc:
        return {
            "enabled": True,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "SerpAPI request failed before parsing response.",
        }


def _domain_to_country(domain: str) -> str:
    d = str(domain or "").lower().strip()
    if d.endswith(".in"):
        return "IN"
    if d.endswith(".co.uk"):
        return "UK"
    if d.endswith(".ca"):
        return "CA"
    if d.endswith(".de"):
        return "DE"
    if d.endswith(".ae"):
        return "AE"
    return "US"


def _collect_review_dicts(node: Any, out: List[Dict[str, Any]]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_review_dicts(item, out)
        return
    if not isinstance(node, dict):
        return

    text_keys = ("review_comment", "review_text", "review", "text", "content", "body", "title")
    if any(k in node for k in text_keys):
        out.append(node)

    for value in node.values():
        if isinstance(value, (dict, list)):
            _collect_review_dicts(value, out)


def _scrape_rapidapi_amazon_top_reviews(
    asin: str,
    domain: str,
    max_reviews: int,
) -> Tuple[List[Dict[str, Any]], str]:
    rapid_key = _get_rapidapi_key()
    if not rapid_key:
        return [], "RAPIDAPI_KEY missing."

    country_candidates = [_domain_to_country(domain)]
    if "US" not in country_candidates:
        country_candidates.append("US")
    logs: List[str] = []
    collected: List[Dict[str, Any]] = []
    seen = set()

    try:
        for country in country_candidates:
            resp = requests.get(
                RAPIDAPI_AMAZON_ENDPOINT,
                params={"asin": asin, "country": country},
                headers={
                    "Content-Type": "application/json",
                    "x-rapidapi-host": RAPIDAPI_AMAZON_HOST,
                    "x-rapidapi-key": rapid_key,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logs.append(f"[WARN] RapidAPI amazon top status={resp.status_code} country={country}")
                continue

            payload = resp.json()
            candidates: List[Dict[str, Any]] = []
            _collect_review_dicts(payload, candidates)
            if not candidates:
                logs.append(f"[INFO] RapidAPI amazon top had no review-like objects for country={country}")
                continue

            for item in candidates:
                title = _normalize_text(item.get("review_title") or item.get("title") or "")
                body = _normalize_text(
                    item.get("review_comment")
                    or item.get("review_text")
                    or item.get("review")
                    or item.get("text")
                    or item.get("content")
                    or item.get("body")
                    or ""
                )
                text = _normalize_text(f"{title}. {body}" if title and body else (title or body))
                if not text:
                    continue
                row = {
                    "text": text,
                    "date": _normalize_text(item.get("review_date") or item.get("date") or ""),
                    "star_rating": _safe_rating(
                        item.get("review_star_rating")
                        or item.get("rating")
                        or item.get("rating_value")
                        or item.get("stars")
                        or 3,
                        default=3,
                    ),
                    "user_name": _clean_user_id(
                        item.get("review_author")
                        or item.get("author")
                        or item.get("user_name")
                        or "unknown"
                    ),
                    "review_id": _normalize_text(
                        item.get("review_id")
                        or item.get("id")
                        or item.get("reviewer_id")
                        or ""
                    ),
                    "rating_source": "parsed",
                }
                key = analyzer._review_dedupe_key(row)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(row)
                if len(collected) >= max_reviews:
                    break

            if collected:
                logs.append(f"[INFO] RapidAPI amazon top country={country} total: {len(collected)}")
                break
        logs.append(f"[INFO] RapidAPI amazon top reviews total: {len(collected)}")
        return collected[:max_reviews], "\n".join(logs)
    except Exception as exc:
        logs.append(f"[WARN] RapidAPI amazon error: {type(exc).__name__}: {exc}")
        logs.append(traceback.format_exc())
        return [], "\n".join(logs)


def _scrape_rapidapi_amazon_reviews_paginated(
    asin: str,
    domain: str,
    max_reviews: int,
    max_pages: int,
) -> Tuple[List[Dict[str, Any]], str]:
    rapid_key = _get_rapidapi_key()
    if not rapid_key:
        return [], "RAPIDAPI_KEY missing."

    country_candidates = [_domain_to_country(domain)]
    if "US" not in country_candidates:
        country_candidates.append("US")
    logs: List[str] = []
    collected: List[Dict[str, Any]] = []
    seen = set()

    max_pages = max(1, min(int(max_pages), 100))
    for country in country_candidates:
        empty_pages = 0
        country_added = 0
        for page in range(1, max_pages + 1):
            if len(collected) >= max_reviews:
                break
            try:
                resp = requests.get(
                    RAPIDAPI_AMAZON_REVIEWS_ENDPOINT,
                    params={
                        "asin": asin,
                        "country": country,
                        "page": page,
                        "sort_by": "TOP_REVIEWS",
                        "star_rating": "ALL",
                        "verified_purchases_only": "false",
                        "images_or_videos_only": "false",
                        "current_format_only": "false",
                    },
                    headers={
                        "Content-Type": "application/json",
                        "x-rapidapi-host": RAPIDAPI_AMAZON_HOST,
                        "x-rapidapi-key": rapid_key,
                    },
                    timeout=30,
                )
                if resp.status_code != 200:
                    logs.append(
                        f"[WARN] RapidAPI product-reviews status={resp.status_code} country={country} page={page}"
                    )
                    if resp.status_code in {401, 403, 404, 429}:
                        break
                    continue

                payload = resp.json()
                data = payload.get("data", payload)
                raw_reviews = []
                if isinstance(data, dict):
                    raw_reviews = data.get("reviews") or data.get("results") or data.get("items") or []
                if not isinstance(raw_reviews, list):
                    raw_reviews = []

                added = 0
                for item in raw_reviews:
                    if not isinstance(item, dict):
                        continue
                    title = _normalize_text(item.get("review_title") or item.get("title") or "")
                    body = _normalize_text(
                        item.get("review_comment")
                        or item.get("review_text")
                        or item.get("review")
                        or item.get("text")
                        or item.get("content")
                        or item.get("body")
                        or ""
                    )
                    text = _normalize_text(f"{title}. {body}" if title and body else (title or body))
                    if not text:
                        continue
                    row = {
                        "text": text,
                        "date": _normalize_text(item.get("review_date") or item.get("date") or ""),
                        "star_rating": _safe_rating(
                            item.get("review_star_rating")
                            or item.get("rating")
                            or item.get("rating_value")
                            or item.get("stars")
                            or 3,
                            default=3,
                        ),
                        "user_name": _clean_user_id(
                            item.get("review_author")
                            or item.get("author")
                            or item.get("user_name")
                            or "unknown"
                        ),
                        "review_id": _normalize_text(
                            item.get("review_id")
                            or item.get("id")
                            or item.get("reviewer_id")
                            or ""
                        ),
                        "rating_source": "parsed",
                    }
                    key = analyzer._review_dedupe_key(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.append(row)
                    added += 1
                    country_added += 1
                    if len(collected) >= max_reviews:
                        break

                logs.append(
                    f"[INFO] RapidAPI product-reviews country={country} page={page}: +{added} -> total={len(collected)}"
                )
                if added == 0:
                    empty_pages += 1
                else:
                    empty_pages = 0
                if empty_pages >= 2:
                    break
            except Exception as exc:
                logs.append(
                    f"[WARN] RapidAPI product-reviews error country={country} page={page}: {type(exc).__name__}: {exc}"
                )
                break
        if country_added > 0:
            break

    logs.append(f"[INFO] RapidAPI product-reviews total: {len(collected)}")
    return collected[:max_reviews], "\n".join(logs)


def _scrape_serpapi_amazon_product_reviews(
    asin: str,
    domain: str,
    max_reviews: int,
) -> Tuple[List[Dict[str, Any]], str]:
    serp_api_key = _get_serp_api_key()
    if (
        not serp_api_key
        or serp_api_key == "YOUR_SERP_API_KEY_HERE"
        or analyzer.GoogleSearch is None
    ):
        return [], "SERP_API_KEY missing or SerpAPI SDK unavailable."

    logs: List[str] = []
    collected: List[Dict[str, Any]] = []
    seen = set()

    try:
        params = {
            "engine": "amazon_product",
            "amazon_domain": domain,
            "asin": asin,
            "api_key": serp_api_key,
        }
        result = analyzer.GoogleSearch(params).get_dict()
        err = str(result.get("error") or "").strip()
        if err:
            logs.append(f"[WARN] SerpAPI amazon_product error: {err}")
            return [], "\n".join(logs)

        def add_review(text: Any, date: Any = "", author: Any = "unknown", rating: Any = 3, review_id: Any = "") -> int:
            review_text = _normalize_text(text)
            if not review_text:
                return 0
            row = {
                "text": review_text,
                "date": _normalize_text(date),
                "star_rating": _safe_rating(rating, default=3),
                "user_name": _clean_user_id(author),
                "review_id": _normalize_text(review_id),
                "rating_source": "parsed",
            }
            key = analyzer._review_dedupe_key(row)
            if key in seen:
                return 0
            seen.add(key)
            collected.append(row)
            return 1

        # Reuse existing extractor logic from analyzer for common SerpAPI fields.
        analyzer._extract_reviews_from_result(
            result,
            lambda t, d="": add_review(text=t, date=d, author="unknown", rating=3),
        )

        for bucket_key in ("reviews", "top_reviews"):
            raw_bucket = result.get(bucket_key)
            if isinstance(raw_bucket, list):
                for rev in raw_bucket:
                    if isinstance(rev, dict):
                        add_review(
                            text=rev.get("body") or rev.get("text") or rev.get("content"),
                            date=rev.get("date", ""),
                            author=rev.get("author", "unknown"),
                            rating=rev.get("rating", 3),
                            review_id=rev.get("review_id") or rev.get("id") or "",
                        )

        reviews_results = result.get("reviews_results")
        if isinstance(reviews_results, dict):
            reviews_results = (
                reviews_results.get("reviews")
                or reviews_results.get("results")
                or reviews_results.get("items")
                or []
            )
        if isinstance(reviews_results, list):
            for rev in reviews_results:
                if isinstance(rev, dict):
                    add_review(
                        text=rev.get("body") or rev.get("text") or rev.get("content"),
                        date=rev.get("date", ""),
                        author=rev.get("author", "unknown"),
                        rating=rev.get("rating", 3),
                        review_id=rev.get("review_id") or rev.get("id") or "",
                    )
                elif isinstance(rev, str):
                    add_review(text=rev)

        logs.append(f"[INFO] SerpAPI amazon_product fallback total: {len(collected)}")
        return collected[:max_reviews], "\n".join(logs)
    except Exception as exc:
        logs.append(f"[WARN] SerpAPI amazon_product fallback error: {type(exc).__name__}: {exc}")
        logs.append(traceback.format_exc())
        return [], "\n".join(logs)


def _load_local_cache_flexible(max_reviews: int) -> List[Dict[str, Any]]:
    def load_csv(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path, on_bad_lines="skip").fillna("")
        except Exception:
            try:
                return pd.read_csv(path, engine="python", on_bad_lines="skip").fillna("")
            except Exception:
                return pd.DataFrame()

    def pick_col(df: pd.DataFrame, *tokens: str) -> str:
        normalized = {col: re.sub(r"\s+", "", str(col).lower()) for col in df.columns}
        for token in tokens:
            token_norm = re.sub(r"\s+", "", token.lower())
            for col, norm in normalized.items():
                if norm == token_norm or norm.startswith(token_norm):
                    return col
        return ""

    def extract_rows(df: pd.DataFrame, limit: int, source_tag: str) -> List[Dict[str, Any]]:
        if df.empty or limit <= 0:
            return []
        review_col = pick_col(df, "text_raw", "review", "text", "text_", "content", "body")
        user_col = pick_col(df, "user_name", "user", "user_id")
        date_col = pick_col(df, "date_raw", "date", "timestamp")
        review_id_col = pick_col(df, "review_id", "id")
        star_col = pick_col(df, "star_rating", "rating")
        rating_source_col = pick_col(df, "rating_source")

        rows: List[Dict[str, Any]] = []
        safe_source_tag = re.sub(r"[^a-z0-9]+", "_", source_tag.lower()).strip("_") or "cache"
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            text = _normalize_text(row.get(review_col, "")) if review_col else ""
            if not text:
                continue
            user_id = _clean_user_id(row.get(user_col, "unknown")) if user_col else "unknown"
            if user_id == "unknown":
                user_id = f"{safe_source_tag}_user_{idx}"
            review_id = _normalize_text(row.get(review_id_col, "")) if review_id_col else ""
            if not review_id:
                review_id = f"{safe_source_tag}_review_{idx}"
            rows.append(
                {
                    "text": text,
                    "date": _normalize_text(row.get(date_col, "")) if date_col else "",
                    "star_rating": _safe_rating(row.get(star_col, 3), default=3) if star_col else 3,
                    "user_name": user_id,
                    "review_id": review_id,
                    "rating_source": _normalize_text(row.get(rating_source_col, "default")) if rating_source_col else "default",
                }
            )
            if len(rows) >= limit:
                break
        return rows

    candidates = [
        analyzer.ANALYSIS_RESULTS_PATH,
        "scraped_training_reviews.csv",
        "fake reviews dataset.csv",
    ]
    out: List[Dict[str, Any]] = []
    seen = set()
    for path in candidates:
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        needed = max_reviews - len(out)
        if needed <= 0:
            break
        extracted = extract_rows(df, needed * 3, source_tag=os.path.basename(path))
        for row in extracted:
            key = analyzer._review_dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= max_reviews:
                break
    return out[:max_reviews]


def collect_reviews_everywhere(
    target: str,
    pages: int,
    max_reviews: int,
    target_domain: str,
    allow_cache: bool,
    min_required_reviews: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "target": target,
        "started_at": _now_iso(),
        "sources": [],
    }

    product_details: Dict[str, Any] = {
        "title": "Unknown Product",
        "thumbnail": "",
        "rating": "",
        "reviews_total": 0,
    }
    collected: List[Dict[str, Any]] = []
    raw = str(target or "").strip()

    if analyzer.is_flipkart_url(raw):
        diagnostics["platform"] = "flipkart"
        start = time.time()
        try:
            flipkart_reviews = analyzer._scrape_flipkart(raw, max_reviews=max_reviews, max_pages=pages)
            tagged = _tag_reviews(flipkart_reviews, "flipkart_api")
            collected = analyzer._merge_unique_reviews(collected, tagged, max_reviews)
            _record_source(diagnostics, "flipkart_api", start, "ok", count=len(tagged))
            product_details["title"] = "Flipkart Product"
            product_details["reviews_total"] = len(collected)
        except Exception as exc:
            _record_source(
                diagnostics,
                "flipkart_api",
                start,
                "error",
                error=f"{type(exc).__name__}: {exc}",
                extra={"traceback": traceback.format_exc()},
            )
        return collected, product_details, diagnostics

    if not analyzer.is_amazon_url(raw):
        diagnostics["platform"] = "unknown"
        diagnostics["ended_at"] = _now_iso()
        return collected, product_details, diagnostics

    diagnostics["platform"] = "amazon"
    asin = analyzer.extract_asin(raw)
    domain = analyzer.extract_domain(raw) if "amazon." in raw.lower() else target_domain
    diagnostics["asin"] = asin
    product_details["title"] = f"Amazon Product ({asin})"

    # For ASIN-only runs, probe multiple Amazon domains and choose the one with best metadata coverage.
    domain_candidates: List[str] = [str(domain).strip().lower()]
    if "amazon." not in raw.lower():
        for d in ("amazon.in", "amazon.com"):
            if d not in domain_candidates:
                domain_candidates.append(d)
    diagnostics["domain_candidates"] = domain_candidates

    fallback_reviews: List[Dict[str, Any]] = []
    best_domain = domain_candidates[0]
    best_product = product_details
    best_fallback: List[Dict[str, Any]] = []
    best_total = -1

    for candidate_domain in domain_candidates:
        metadata_start = time.time()
        try:
            (candidate_product, candidate_fallback), metadata_logs = _run_with_captured_stdout(
                lambda d=candidate_domain: analyzer._fetch_metadata(asin, d)
            )
            candidate_total = int(analyzer._safe_int(candidate_product.get("reviews_total", 0), 0))
            _record_source(
                diagnostics,
                f"metadata@{candidate_domain}",
                metadata_start,
                "warning" if "[WARN]" in metadata_logs else "ok",
                count=len(candidate_fallback),
                extra={
                    "title": candidate_product.get("title", ""),
                    "reviews_total": candidate_total,
                    "logs": metadata_logs[-6000:],
                },
            )

            if candidate_total > best_total:
                best_total = candidate_total
                best_domain = candidate_domain
                best_product = candidate_product
                best_fallback = candidate_fallback
            elif best_total <= 0 and candidate_fallback and not best_fallback:
                # If no domain has review count metadata, prefer one that still gives snippets.
                best_domain = candidate_domain
                best_product = candidate_product
                best_fallback = candidate_fallback
        except Exception as exc:
            _record_source(
                diagnostics,
                f"metadata@{candidate_domain}",
                metadata_start,
                "error",
                error=f"{type(exc).__name__}: {exc}",
                extra={"traceback": traceback.format_exc()},
            )

    domain = best_domain
    product_details = best_product
    fallback_reviews = best_fallback
    diagnostics["domain"] = domain
    if len(domain_candidates) > 1:
        _log(f"[INFO] Selected domain for ASIN {asin}: {domain}")

    target_reviews = max_reviews
    available_reviews = analyzer._safe_int(product_details.get("reviews_total", 0), 0)
    if available_reviews > 0:
        target_reviews = min(max_reviews, available_reviews)
    min_required_reviews = max(0, int(min_required_reviews or 0))
    if min_required_reviews > 0:
        target_reviews = max(target_reviews, min_required_reviews)
        target_reviews = min(target_reviews, max_reviews)
    diagnostics["target_reviews"] = int(target_reviews)
    _log(f"[INFO] Target scan volume: {target_reviews} reviews")

    scrapingdog_key = _get_scrapingdog_key()
    if scrapingdog_key and scrapingdog_key != "YOUR_SCRAPINGDOG_KEY_HERE":
        start = time.time()
        try:
            remaining = max(target_reviews - len(collected), 0)
            sd_reviews, sd_logs = _run_with_captured_stdout(
                lambda: analyzer._scrape_amazon_scrapingdog(
                    asin, domain, max_reviews=remaining, max_pages=pages
                )
            )
            tagged = _tag_reviews(sd_reviews, "scrapingdog")
            collected = analyzer._merge_unique_reviews(collected, tagged, target_reviews)
            status = "warning" if ("[WARN]" in sd_logs or (len(tagged) == 0 and sd_logs)) else "ok"
            _record_source(
                diagnostics,
                "scrapingdog",
                start,
                status,
                count=len(tagged),
                extra={"logs": sd_logs[-6000:]},
            )
        except Exception as exc:
            _record_source(
                diagnostics,
                "scrapingdog",
                start,
                "error",
                error=f"{type(exc).__name__}: {exc}",
                extra={"traceback": traceback.format_exc()},
            )
    else:
        _record_source(
            diagnostics,
            "scrapingdog",
            time.time(),
            "skipped",
            error="SCRAPINGDOG_KEY not set.",
        )

    serp_probe_start = time.time()
    probe = _probe_serpapi_amazon_reviews(asin, domain)
    probe_error = str(probe.get("error", "")).strip()
    probe_status = "ok" if probe.get("ok") else ("skipped" if not probe.get("enabled") else "error")
    if probe_status == "error" and "Unsupported `amazon_reviews` search engine" in probe_error:
        # Account-level limitation, not a pipeline crash. SerpAPI product fallback still runs.
        probe_status = "warning"
    _record_source(
        diagnostics,
        "serpapi_probe",
        serp_probe_start,
        probe_status,
        error=probe_error,
        extra={"hint": str(probe.get("hint", "")), "enabled": bool(probe.get("enabled", False))},
    )

    if probe.get("ok"):
        serp_start = time.time()
        try:
            remaining = max(target_reviews - len(collected), 0)
            serp_reviews, serp_logs = _run_with_captured_stdout(
                lambda: analyzer._scrape_amazon_serpapi_reviews(
                    asin, domain, max_reviews=remaining, max_pages=pages
                )
            )
            tagged = _tag_reviews(serp_reviews, "serpapi")
            collected = analyzer._merge_unique_reviews(collected, tagged, target_reviews)
            status = "warning" if ("[WARN]" in serp_logs or (len(tagged) == 0 and serp_logs)) else "ok"
            _record_source(
                diagnostics,
                "serpapi",
                serp_start,
                status,
                count=len(tagged),
                extra={"logs": serp_logs[-6000:]},
            )
        except Exception as exc:
            _record_source(
                diagnostics,
                "serpapi",
                serp_start,
                "error",
                error=f"{type(exc).__name__}: {exc}",
                extra={"traceback": traceback.format_exc()},
            )
    else:
        _record_source(
            diagnostics,
            "serpapi",
            time.time(),
            "skipped",
            error=str(probe.get("error", "")),
            extra={"hint": str(probe.get("hint", ""))},
        )

    # Secondary SerpAPI fallback that does not require engine=amazon_reviews.
    serp_product_start = time.time()
    try:
        remaining = max(target_reviews - len(collected), 0)
        product_reviews, product_logs = _scrape_serpapi_amazon_product_reviews(
            asin=asin,
            domain=domain,
            max_reviews=remaining,
        )
        tagged = _tag_reviews(product_reviews, "serpapi_amazon_product")
        collected = analyzer._merge_unique_reviews(collected, tagged, target_reviews)
        status = "warning" if ("[WARN]" in product_logs or (len(tagged) == 0 and product_logs)) else "ok"
        _record_source(
            diagnostics,
            "serpapi_amazon_product",
            serp_product_start,
            status,
            count=len(tagged),
            extra={"logs": product_logs[-6000:]},
        )
    except Exception as exc:
        _record_source(
            diagnostics,
            "serpapi_amazon_product",
            serp_product_start,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc()},
        )

    rapidapi_reviews_start = time.time()
    try:
        remaining = max(target_reviews - len(collected), 0)
        rapid_reviews, rapid_logs = _scrape_rapidapi_amazon_reviews_paginated(
            asin=asin,
            domain=domain,
            max_reviews=remaining,
            max_pages=pages,
        )
        tagged = _tag_reviews(rapid_reviews, "rapidapi_amazon_reviews")
        collected = analyzer._merge_unique_reviews(collected, tagged, target_reviews)
        status = "warning" if ("[WARN]" in rapid_logs or (len(tagged) == 0 and rapid_logs)) else "ok"
        _record_source(
            diagnostics,
            "rapidapi_amazon_reviews",
            rapidapi_reviews_start,
            status,
            count=len(tagged),
            extra={"logs": rapid_logs[-6000:]},
        )
    except Exception as exc:
        _record_source(
            diagnostics,
            "rapidapi_amazon_reviews",
            rapidapi_reviews_start,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc()},
        )

    rapidapi_top_start = time.time()
    try:
        remaining = max(target_reviews - len(collected), 0)
        rapid_top_reviews, rapid_top_logs = _scrape_rapidapi_amazon_top_reviews(
            asin=asin,
            domain=domain,
            max_reviews=remaining,
        )
        tagged = _tag_reviews(rapid_top_reviews, "rapidapi_amazon_top")
        collected = analyzer._merge_unique_reviews(collected, tagged, target_reviews)
        status = "warning" if ("[WARN]" in rapid_top_logs or (len(tagged) == 0 and rapid_top_logs)) else "ok"
        _record_source(
            diagnostics,
            "rapidapi_amazon_top",
            rapidapi_top_start,
            status,
            count=len(tagged),
            extra={"logs": rapid_top_logs[-6000:]},
        )
    except Exception as exc:
        _record_source(
            diagnostics,
            "rapidapi_amazon_top",
            rapidapi_top_start,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc()},
        )

    direct_start = time.time()
    try:
        remaining = max(target_reviews - len(collected), 0)
        direct_reviews, direct_logs = _run_with_captured_stdout(
            lambda: analyzer._scrape_amazon_direct(
                asin, domain, max_reviews=remaining, max_pages=pages
            )
        )
        tagged = _tag_reviews(direct_reviews, "direct")
        collected = analyzer._merge_unique_reviews(collected, tagged, target_reviews)
        status = "warning" if ("[WARN]" in direct_logs or (len(tagged) == 0 and direct_logs)) else "ok"
        _record_source(
            diagnostics,
            "direct",
            direct_start,
            status,
            count=len(tagged),
            extra={"logs": direct_logs[-6000:]},
        )
    except Exception as exc:
        _record_source(
            diagnostics,
            "direct",
            direct_start,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc()},
        )

    fallback_start = time.time()
    try:
        tagged_fallback = _tag_reviews(fallback_reviews, "metadata_snippet")
        collected = analyzer._merge_unique_reviews(collected, tagged_fallback, target_reviews)
        _record_source(diagnostics, "metadata_snippet", fallback_start, "ok", count=len(tagged_fallback))
    except Exception as exc:
        _record_source(
            diagnostics,
            "metadata_snippet",
            fallback_start,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc()},
        )

    if allow_cache and len(collected) < target_reviews:
        cache_start = time.time()
        try:
            needed = max(target_reviews - len(collected), 0)
            cache_reviews = _load_local_cache_flexible(max_reviews=needed)
            if not cache_reviews:
                cache_reviews = analyzer._load_cached_reviews(max_reviews=needed)
            tagged = _tag_reviews(cache_reviews, "local_cache")
            collected = analyzer._merge_unique_reviews(collected, tagged, target_reviews)
            status = "ok" if len(tagged) > 0 else "warning"
            _record_source(diagnostics, "local_cache", cache_start, status, count=len(tagged))
        except Exception as exc:
            _record_source(
                diagnostics,
                "local_cache",
                cache_start,
                "error",
                error=f"{type(exc).__name__}: {exc}",
                extra={"traceback": traceback.format_exc()},
            )

    diagnostics["ended_at"] = _now_iso()
    diagnostics["total_collected"] = int(len(collected))
    live_source_names = {
        "direct",
        "scrapingdog",
        "serpapi",
        "rapidapi_amazon_reviews",
        "rapidapi_amazon_top",
        "flipkart_api",
    }
    fallback_source_names = {
        "metadata_snippet",
        "local_cache",
        "serpapi_amazon_product",
    }
    live_count = int(
        sum(
            int(s.get("count", 0) or 0)
            for s in diagnostics.get("sources", [])
            if str(s.get("source", "")) in live_source_names
        )
    )
    cache_count = int(
        sum(
            int(s.get("count", 0) or 0)
            for s in diagnostics.get("sources", [])
            if str(s.get("source", "")) in fallback_source_names
        )
    )
    diagnostics["live_collected"] = live_count
    diagnostics["fallback_collected"] = cache_count
    return collected[:target_reviews], product_details, diagnostics


def build_training_dataframe(
    reviews: List[Dict[str, Any]],
    product_id: str,
    platform: str,
    min_words: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dropped_empty = 0
    dropped_short = 0

    for i, review in enumerate(reviews):
        try:
            raw_text = _normalize_text(review.get("text", ""))
            if not raw_text:
                dropped_empty += 1
                continue

            clean = _clean_text_for_training(raw_text)
            if not clean:
                dropped_empty += 1
                continue
            if len(clean.split()) < min_words:
                dropped_short += 1
                continue

            rating = _safe_rating(review.get("star_rating", review.get("rating", 3)), default=3)
            user_id = _clean_user_id(review.get("user_name", review.get("user_id", "unknown")))
            review_id = _normalize_text(review.get("review_id", ""))
            if not review_id:
                review_id = f"{product_id}_{i + 1}"
            date_raw = _normalize_text(review.get("date", ""))
            ts_iso = _to_timestamp_iso(date_raw)

            rows.append(
                {
                    "category": f"{platform}_{product_id}",
                    "rating": float(rating),
                    "label": "",
                    "text_": clean,
                    "text_raw": raw_text,
                    "review_id": review_id,
                    "user_id": user_id,
                    "date_raw": date_raw,
                    "timestamp": ts_iso,
                    "product_id": product_id,
                    "platform": platform,
                    "source": _normalize_text(review.get("_source", "unknown")) or "unknown",
                    "rating_source": _normalize_text(review.get("rating_source", "default")) or "default",
                }
            )
        except Exception:
            # Keep dataset export resilient even with malformed rows.
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["text_", "user_id", "rating"], keep="first").reset_index(drop=True)
    _log(
        f"[INFO] Build dataset: kept={len(df)} | dropped_empty={dropped_empty} | dropped_short={dropped_short}"
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape reviews from multiple sources and export cleaned training CSV."
    )
    # Positional mode (no flags): target pages max_reviews output debug_json
    parser.add_argument("target", nargs="?", help="Amazon URL/ASIN or Flipkart URL")
    parser.add_argument("pages_pos", nargs="?", type=int, help="Max pages per source")
    parser.add_argument("max_reviews_pos", nargs="?", type=int, help="Max reviews to collect")
    parser.add_argument("output_pos", nargs="?", help="Output CSV path")
    parser.add_argument("debug_json_pos", nargs="?", help="Debug report JSON path")

    # Flag mode remains supported for compatibility.
    parser.add_argument("--target", dest="target_flag", default=None, help="Amazon URL/ASIN or Flipkart URL")
    parser.add_argument("--output", dest="output_flag", default=None, help="Output CSV path")
    parser.add_argument("--debug-json", dest="debug_json_flag", default=None, help="Debug report JSON path")
    parser.add_argument("--pages", dest="pages_flag", type=int, default=None, help="Max pages per source")
    parser.add_argument("--max-reviews", dest="max_reviews_flag", type=int, default=None, help="Max reviews to collect")
    parser.add_argument("--min-required-reviews", dest="min_required_reviews_flag", type=int, default=None, help="Ensure at least this many reviews via live+cache backfill")
    parser.add_argument("--target-domain", dest="target_domain_flag", default=None, help="Domain for ASIN input")
    parser.add_argument("--min-words", dest="min_words_flag", type=int, default=None, help="Minimum words in cleaned text")
    parser.add_argument("--allow-cache", dest="allow_cache_flag", action="store_true", help="Allow fallback to local cached CSV when live scraping fails")
    parser.add_argument("--no-allow-cache", dest="allow_cache_flag", action="store_false", help="Disable local cache fallback")
    parser.set_defaults(allow_cache_flag=None)
    args = parser.parse_args()

    target = (
        args.target_flag
        or args.target
        or PROJECT_DEFAULTS["target"]
    )
    pages = (
        args.pages_flag
        if args.pages_flag is not None
        else (args.pages_pos if args.pages_pos is not None else PROJECT_DEFAULTS["pages"])
    )
    max_reviews = (
        args.max_reviews_flag
        if args.max_reviews_flag is not None
        else (args.max_reviews_pos if args.max_reviews_pos is not None else PROJECT_DEFAULTS["max_reviews"])
    )
    min_required_reviews = (
        args.min_required_reviews_flag
        if args.min_required_reviews_flag is not None
        else PROJECT_DEFAULTS["min_required_reviews"]
    )
    output = args.output_flag or args.output_pos or PROJECT_DEFAULTS["output"]
    debug_json = args.debug_json_flag or args.debug_json_pos or PROJECT_DEFAULTS["debug_json"]
    target_domain = args.target_domain_flag or PROJECT_DEFAULTS["target_domain"]
    min_words = args.min_words_flag if args.min_words_flag is not None else PROJECT_DEFAULTS["min_words"]
    allow_cache = (
        args.allow_cache_flag if args.allow_cache_flag is not None else bool(PROJECT_DEFAULTS["allow_cache"])
    )
    if int(max_reviews) < int(min_required_reviews):
        _log(
            f"[INFO] Increasing max_reviews from {max_reviews} to {min_required_reviews} "
            f"to satisfy minimum required reviews."
        )
        max_reviews = int(min_required_reviews)

    _log(f"[INFO] Started scraping at {datetime.now().isoformat()}")
    _log(f"[INFO] Target: {target}")
    _log(f"[INFO] Output CSV: {output}")
    _log(f"[INFO] Debug JSON: {debug_json}")
    _log(
        f"[INFO] Pages={pages} | MaxReviews={max_reviews} | MinRequired={min_required_reviews} "
        f"| MinWords={min_words} | AllowCache={allow_cache}"
    )

    try:
        reviews, product, diagnostics = collect_reviews_everywhere(
            target=target,
            pages=max(1, int(pages)),
            max_reviews=max(1, int(max_reviews)),
            target_domain=str(target_domain or "amazon.com"),
            allow_cache=bool(allow_cache),
            min_required_reviews=max(0, int(min_required_reviews)),
        )

        if analyzer.is_amazon_url(target):
            product_id = analyzer.extract_asin(target)
            platform = "amazon"
        elif analyzer.is_flipkart_url(target):
            product_id = "flipkart_product"
            platform = "flipkart"
        else:
            product_id = "unknown_product"
            platform = "unknown"

        df = build_training_dataframe(
            reviews=reviews,
            product_id=product_id,
            platform=platform,
            min_words=max(1, int(min_words)),
        )

        output_dir = os.path.dirname(os.path.abspath(output))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        existing_count = 0
        merged_df = df.copy()
        if os.path.exists(output):
            try:
                existing_df = pd.read_csv(output, on_bad_lines="skip").fillna("")
                existing_count = int(len(existing_df))
                merged_df = pd.concat([existing_df, df], ignore_index=True)
                if not merged_df.empty and {"text_", "user_id", "rating"}.issubset(set(merged_df.columns)):
                    merged_df = merged_df.drop_duplicates(subset=["text_", "user_id", "rating"], keep="first").reset_index(drop=True)
            except Exception:
                merged_df = df.copy()

        merged_df.to_csv(output, index=False, encoding="utf-8")
        added_count = int(len(merged_df) - existing_count)
        _log(
            f"[INFO] Saved {len(merged_df)} total rows to {output} "
            f"(existing={existing_count}, added={max(0, added_count)})"
        )
        _log(
            f"[INFO] Source counts: live={diagnostics.get('live_collected', 0)}, "
            f"fallback={diagnostics.get('fallback_collected', 0)}, "
            f"total_collected={diagnostics.get('total_collected', 0)}"
        )
        if int(diagnostics.get("live_collected", 0)) == 0:
            _log(
                "[WARN] Live scraping returned 0 rows. Output currently comes from fallback/cache only."
            )
            _log(
                "[WARN] Typical causes: RapidAPI access denied/rate-limited (403/429), Amazon bot challenge, or missing paid scraper key."
            )

        report = {
            "run_at": _now_iso(),
            "target": target,
            "output_csv": os.path.abspath(output),
            "product": product,
            "rows_written": int(len(merged_df)),
            "diagnostics": diagnostics,
        }

        report_dir = os.path.dirname(os.path.abspath(debug_json))
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        with open(debug_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        _log(f"[INFO] Saved debug report to {debug_json}")

        blocking_errors = [s for s in diagnostics.get("sources", []) if s.get("status") == "error"]
        source_warnings = [s for s in diagnostics.get("sources", []) if s.get("status") == "warning"]
        if blocking_errors or source_warnings:
            _log("[WARN] Some sources reported issues. Check debug JSON for exact errors and captured logs.")
            for e in blocking_errors:
                _log(f"[WARN] {e.get('source')} [error]: {e.get('error')}")
            for w in source_warnings:
                err = str(w.get("error", "")).strip()
                if err:
                    _log(f"[WARN] {w.get('source')} [warning]: {err}")
                else:
                    _log(f"[WARN] {w.get('source')} [warning]: see captured logs in debug JSON.")
                logs = str(w.get("logs", "")).strip()
                if logs:
                    first_line = logs.splitlines()[0][:260]
                    _log(f"[WARN] {w.get('source')} detail: {first_line}")
        else:
            _log("[INFO] Completed without source-level errors.")
        return 0
    except Exception as exc:
        _log(f"[ERROR] Scraping pipeline crashed: {type(exc).__name__}: {exc}")
        _log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
