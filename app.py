from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import math
import re
import hashlib
from analyzer import analyze_product
import json
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime, timezone, timedelta
import pandas as pd

from product_intelligence import (
    calculate_discrepancy_score,
    detect_coordinated_bot_networks,
)

app = Flask(__name__)
app.secret_key = "262ad8b51b449946485141e9ee2521a8d0120bd6b0ba609c667ed3a3d56d0495"

FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY", "AIzaSyCFZukdrr5sehQ2NRvtrzRnIm0qog2AiDQ")

@app.context_processor
def inject_firebase_key():
    return dict(firebase_api_key=FIREBASE_WEB_API_KEY)

UPLOAD_FOLDER      = "static/uploads/avatars"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LOCAL_TRUST_HISTORY_FILE = "trust_history.csv"
LOCAL_SCAN_HISTORY_FILE = "scan_history_local.jsonl"
TRAINING_DATA_CANDIDATES = ["fake reviews dataset.csv", "scraped_training_reviews.csv"]
_HOME_METRICS_CACHE = {"key": None, "payload": None}
_GLOBAL_CATEGORY_CACHE = {"key": None, "rows": [], "data_source": "none"}
DEFAULT_AVATAR_URL = (
    "https://lh3.googleusercontent.com/aida-public/AB6AXuBdi-P-pYGbwKCGyQeR-ewKR2hOIv7_NdiX301B0Kb913TQmlFjJNwWMICwFvIluwmozxWn7Jg-hf7OIIQeLXhWj-h8aadKRhnliFfLEht3B6ECeskiKiHi7LNZgvaOvuyCY-BS_A8hwypI_WFdSKIAb8Qh95TilDHaRdM6VQSRGtqUYDXHBfo0bu8559XF0d-E5JJ_qddJcXkuVrJc3hW_GFI2i0ZDF8sXDAilqj3LOXVHpMXGs0AsFiL_d5OsPgoGtJOVjZ5EGHMS"
)


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _format_int(value: int) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _count_csv_rows_fast(path: str) -> int:
    """
    Counts CSV rows without loading into pandas.
    Returns number of data rows (header excluded).
    """
    try:
        with open(path, "rb") as fh:
            line_count = sum(1 for _ in fh)
        return max(0, line_count - 1)
    except Exception:
        return 0


def _build_home_metrics() -> dict:
    cache_key_parts = []
    for p in [LOCAL_TRUST_HISTORY_FILE, *TRAINING_DATA_CANDIDATES]:
        try:
            cache_key_parts.append((p, os.path.getmtime(p)))
        except OSError:
            cache_key_parts.append((p, None))
    cache_key = tuple(cache_key_parts)
    if _HOME_METRICS_CACHE["key"] == cache_key and isinstance(_HOME_METRICS_CACHE["payload"], dict):
        return _HOME_METRICS_CACHE["payload"]

    metrics = {
        "recent_avg_trust": 0.0,
        "recent_avg_fraud": 0.0,
        "model_accuracy": 0.0,
        "total_scans": 0,
        "unique_products": 0,
        "training_reviews": 0,
        "last_scan_label": "No scans yet",
        "recent_avg_trust_text": "0.0%",
        "recent_avg_fraud_text": "0.0%",
        "model_accuracy_text": "0.0%",
        "total_scans_text": "0",
        "unique_products_text": "0",
        "training_reviews_text": "0",
    }

    try:
        if os.path.exists(LOCAL_TRUST_HISTORY_FILE):
            df = pd.read_csv(LOCAL_TRUST_HISTORY_FILE)
            if not df.empty:
                df["trust_score"] = pd.to_numeric(df.get("trust_score"), errors="coerce")
                df["fraud_score"] = pd.to_numeric(df.get("fraud_score"), errors="coerce")
                df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)
                clean = df.dropna(subset=["trust_score", "fraud_score", "timestamp"]).copy()
                if not clean.empty:
                    clean = clean.sort_values("timestamp", ascending=False)
                    recent = clean.head(50)
                    metrics["recent_avg_trust"] = round(float(recent["trust_score"].mean()), 1)
                    metrics["recent_avg_fraud"] = round(float(recent["fraud_score"].mean()), 1)
                    metrics["total_scans"] = int(len(clean))
                    metrics["unique_products"] = int(clean["product_id"].astype(str).nunique())
                    latest_ts = clean["timestamp"].max()
                    if pd.notna(latest_ts):
                        metrics["last_scan_label"] = latest_ts.strftime("%d %b %Y, %I:%M %p UTC")
    except Exception as exc:
        print(f"[WARN] Home metrics history parse failed: {exc}")

    try:
        training_counts = []
        for path in TRAINING_DATA_CANDIDATES:
            if os.path.exists(path):
                training_counts.append(_count_csv_rows_fast(path))
        metrics["training_reviews"] = int(max(training_counts)) if training_counts else 0
    except Exception as exc:
        print(f"[WARN] Home metrics dataset count failed: {exc}")

    env_model_accuracy = _safe_float(os.getenv("MODEL_ACCURACY", "92.0"), 92.0)
    env_model_accuracy = max(0.0, min(100.0, float(env_model_accuracy)))
    metrics["model_accuracy"] = round(env_model_accuracy, 1)

    metrics["recent_avg_trust_text"] = f"{metrics['recent_avg_trust']:.1f}%"
    metrics["recent_avg_fraud_text"] = f"{metrics['recent_avg_fraud']:.1f}%"
    metrics["model_accuracy_text"] = f"{metrics['model_accuracy']:.1f}%"
    metrics["total_scans_text"] = _format_int(metrics["total_scans"])
    metrics["unique_products_text"] = _format_int(metrics["unique_products"])
    metrics["training_reviews_text"] = _format_int(metrics["training_reviews"])

    _HOME_METRICS_CACHE["key"] = cache_key
    _HOME_METRICS_CACHE["payload"] = metrics
    return metrics


def _empty_leaderboard_payload(page: int = 1, page_size: int = 12) -> dict:
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "none",
        "rows": [],
        "top3": [],
        "platforms": [],
        "total_filtered": 0,
        "page": page,
        "page_size": page_size,
        "total_pages": 1,
        "rank_offset": 0,
        "summary": {
            "products": 0,
            "scans": 0,
            "high_risk": 0,
            "clean": 0,
        },
    }


def _clean_events_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp"), errors="coerce", utc=True)
    out["platform"] = out.get("platform", "").astype(str).str.strip().str.lower()
    out["product_id"] = out.get("product_id", "").astype(str).str.strip()
    out["trust_score"] = pd.to_numeric(out.get("trust_score"), errors="coerce")
    out["fraud_score"] = pd.to_numeric(out.get("fraud_score"), errors="coerce")
    out = out.dropna(subset=["timestamp", "platform", "product_id", "trust_score", "fraud_score"]).copy()
    if out.empty:
        return out
    out["product_id"] = out.apply(
        lambda r: _normalize_product_id(r.get("platform", ""), r.get("product_id", "")),
        axis=1,
    )
    return out


def _load_local_leaderboard_events_df() -> pd.DataFrame:
    if not os.path.exists(LOCAL_TRUST_HISTORY_FILE):
        return pd.DataFrame(columns=["timestamp", "platform", "product_id", "trust_score", "fraud_score"])
    try:
        df = pd.read_csv(LOCAL_TRUST_HISTORY_FILE)
        return _clean_events_df(df)
    except Exception as exc:
        print(f"[WARN] Leaderboard local history read failed: {exc}")
        return pd.DataFrame(columns=["timestamp", "platform", "product_id", "trust_score", "fraud_score"])


def _load_firestore_leaderboard_events_df(limit_docs: int = 4000) -> pd.DataFrame:
    cols = ["timestamp", "platform", "product_id", "trust_score", "fraud_score"]
    if db is None:
        return pd.DataFrame(columns=cols)

    docs = None
    try:
        docs = (
            db.collection("scans")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(int(limit_docs))
            .stream()
        )
    except Exception:
        try:
            docs = db.collection("scans").stream()
        except Exception as exc:
            print(f"[WARN] Firestore leaderboard read failed: {exc}")
            return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for i, doc in enumerate(docs):
        if i >= int(limit_docs):
            break
        data = doc.to_dict() or {}
        asin_or_id = str(data.get("asin") or data.get("product_id") or "").strip()
        if not asin_or_id:
            continue

        platform = str(data.get("platform") or _infer_platform(asin_or_id)).strip().lower()
        product_id = _normalize_product_id(platform, asin_or_id)

        summary_obj = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        chart_obj = data.get("chart_data") if isinstance(data.get("chart_data"), dict) else {}
        advanced_obj = (
            summary_obj.get("advanced_analysis", {})
            if isinstance(summary_obj.get("advanced_analysis"), dict)
            else {}
        )

        fraud_val = _safe_float(chart_obj.get("fake_score"), None)
        if fraud_val is None:
            fraud_val = _safe_float(advanced_obj.get("combined_fraud_score"), None)
        if fraud_val is None:
            fraud_val = _safe_float(summary_obj.get("avg_fake_probability"), None)
        if fraud_val is None:
            continue
        fraud_val = round(max(0.0, min(100.0, float(fraud_val))), 3)
        trust_val = round(100.0 - fraud_val, 3)

        ts = data.get("timestamp")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        rows.append(
            {
                "timestamp": ts,
                "platform": platform,
                "product_id": product_id,
                "trust_score": trust_val,
                "fraud_score": fraud_val,
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)
    return _clean_events_df(pd.DataFrame(rows, columns=cols))


def _load_leaderboard_events_df() -> tuple[pd.DataFrame, str]:
    local_df = _load_local_leaderboard_events_df()
    if not local_df.empty:
        return local_df, "local_csv"
    firestore_df = _load_firestore_leaderboard_events_df()
    if not firestore_df.empty:
        return firestore_df, "firestore_scans"
    return pd.DataFrame(columns=["timestamp", "platform", "product_id", "trust_score", "fraud_score"]), "none"


def _aggregate_leaderboard_rows(events_df: pd.DataFrame, max_products: int = 300) -> list[dict]:
    if events_df.empty:
        return []

    now = pd.Timestamp.now(tz="UTC")
    rows: list[dict] = []
    for (platform, product_id), group in events_df.groupby(["platform", "product_id"], sort=False):
        g = group.sort_values("timestamp")
        recent = g.tail(20)
        trend = recent["trust_score"].tail(12).round(1).tolist()
        scans_count = int(len(g))
        trust_score = round(float(recent["trust_score"].mean()), 1)
        fraud_score = round(float(recent["fraud_score"].mean()), 1)
        volatility = float(recent["trust_score"].std(ddof=0)) if len(recent) > 1 else 0.0

        recent_7 = int((g["timestamp"] >= (now - pd.Timedelta(days=7))).sum())
        prev_21 = int(
            ((g["timestamp"] >= (now - pd.Timedelta(days=28))) & (g["timestamp"] < (now - pd.Timedelta(days=7)))).sum()
        )
        burst_ratio = float((recent_7 + 1) / ((prev_21 / 3.0) + 1))

        reasons: list[str] = []
        if fraud_score >= 55:
            reasons.append("High Fraud Score")
        if volatility >= 12:
            reasons.append("Volatile Trend")
        if burst_ratio >= 2.0:
            reasons.append("Burst Activity")
        if scans_count < 3:
            reasons.append("Low Evidence")
        if not reasons:
            reasons.append("Consistent Pattern" if fraud_score <= 20 else "Mixed Signals")

        if trust_score >= 85:
            tier = "Clean"
        elif trust_score >= 60:
            tier = "Watchlist"
        else:
            tier = "High Risk"

        if scans_count >= 8:
            confidence = "High"
        elif scans_count >= 3:
            confidence = "Medium"
        else:
            confidence = "Low"

        latest_ts = g["timestamp"].max()
        last_scan_iso = latest_ts.isoformat() if pd.notna(latest_ts) else ""
        last_scan_epoch = int(latest_ts.timestamp()) if pd.notna(latest_ts) else 0
        rows.append(
            {
                "platform": platform,
                "product_id": product_id,
                "trust_score": trust_score,
                "fraud_score": fraud_score,
                "scans_count": scans_count,
                "confidence": confidence,
                "tier": tier,
                "reasons": reasons[:3],
                "trend": trend,
                "last_scan_iso": last_scan_iso,
                "last_scan_label": latest_ts.strftime("%d %b %Y") if pd.notna(latest_ts) else "Unknown",
                "last_scan_epoch": last_scan_epoch,
            }
        )

    rows.sort(key=lambda x: (x["trust_score"], -x["fraud_score"], x["scans_count"]), reverse=True)
    return rows[: max(1, int(max_products))]


def _sort_leaderboard_rows(rows: list[dict], sort_key: str, tab: str) -> list[dict]:
    out = list(rows)
    if sort_key == "fraud_desc":
        out.sort(key=lambda r: float(r.get("fraud_score", 0.0)), reverse=True)
    elif sort_key == "scans_desc":
        out.sort(key=lambda r: int(r.get("scans_count", 0)), reverse=True)
    elif sort_key == "latest_desc":
        out.sort(key=lambda r: int(r.get("last_scan_epoch", 0)), reverse=True)
    else:
        out.sort(key=lambda r: float(r.get("trust_score", 0.0)), reverse=True)
        if tab == "risk":
            out.sort(key=lambda r: float(r.get("fraud_score", 0.0)), reverse=True)
    return out


def _row_matches_fraud_band(row: dict, fraud_band: str) -> bool:
    if fraud_band == "all":
        return True
    value = _safe_float(row.get("fraud_score"), None)
    if value is None:
        return False
    if fraud_band == "0-20":
        return 0 <= value < 20
    if fraud_band == "20-40":
        return 20 <= value < 40
    if fraud_band == "40-60":
        return 40 <= value < 60
    if fraud_band == "60-100":
        return value >= 60
    return True


def _label_to_genuine_flag(value) -> bool | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw in {"or", "original", "genuine", "real", "authentic", "human", "0", "true"}:
        return True
    if raw in {"cg", "computer_generated", "computer generated", "fake", "spam", "bot", "1", "false"}:
        return False
    if "fake" in raw or "generated" in raw or raw.startswith("cg"):
        return False
    if "orig" in raw or "genuine" in raw or "real" in raw:
        return True
    return None


def _resolve_global_dataset_path() -> str | None:
    for path in TRAINING_DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _build_global_category_rows(max_categories: int = 600) -> tuple[list[dict], str]:
    dataset_path = _resolve_global_dataset_path()
    if not dataset_path:
        return [], "none"

    try:
        file_key = (dataset_path, os.path.getmtime(dataset_path), os.path.getsize(dataset_path))
    except OSError:
        return [], "none"

    if _GLOBAL_CATEGORY_CACHE["key"] == file_key:
        cached_rows = _GLOBAL_CATEGORY_CACHE.get("rows", [])
        cached_source = str(_GLOBAL_CATEGORY_CACHE.get("data_source", "global_dataset"))
        return list(cached_rows)[: max(1, int(max_categories))], cached_source

    try:
        columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    except Exception as exc:
        print(f"[WARN] Failed reading dataset header for leaderboard: {exc}")
        return [], "none"

    category_col = next((c for c in ("category", "Category", "product_category", "productCategory") if c in columns), None)
    label_col = next((c for c in ("label", "Label", "class", "prediction") if c in columns), None)
    if not category_col or not label_col:
        print("[WARN] Global dataset missing category/label columns for leaderboard.")
        return [], "none"

    stats: dict[str, dict[str, float]] = {}
    try:
        for chunk in pd.read_csv(dataset_path, usecols=[category_col, label_col], chunksize=200_000):
            if chunk.empty:
                continue
            cat = (
                chunk[category_col]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
            )
            flags = chunk[label_col].map(_label_to_genuine_flag)
            frame = pd.DataFrame({"category": cat, "is_genuine": flags})
            frame = frame[frame["is_genuine"].notna()]
            if frame.empty:
                continue
            agg = frame.groupby("category")["is_genuine"].agg(total="size", genuine="sum")
            for category_name, row in agg.iterrows():
                name = str(category_name).strip() or "Unknown"
                rec = stats.setdefault(name, {"total": 0.0, "genuine": 0.0})
                rec["total"] += float(row.get("total", 0.0))
                rec["genuine"] += float(row.get("genuine", 0.0))
    except Exception as exc:
        print(f"[WARN] Global leaderboard aggregation failed: {exc}")
        return [], "none"

    if not stats:
        return [], "none"

    try:
        dataset_ts = datetime.fromtimestamp(os.path.getmtime(dataset_path), tz=timezone.utc)
    except Exception:
        dataset_ts = datetime.now(timezone.utc)
    dataset_epoch = int(dataset_ts.timestamp())
    dataset_label = dataset_ts.strftime("%d %b %Y")

    max_total = max(int(v.get("total", 0.0)) for v in stats.values()) if stats else 1
    max_total = max(1, max_total)

    rows: list[dict] = []
    for category_name, rec in stats.items():
        total = int(rec.get("total", 0.0))
        if total <= 0:
            continue
        genuine = int(round(rec.get("genuine", 0.0)))
        genuine = max(0, min(genuine, total))
        fake = total - genuine
        genuine_pct = (genuine / total) * 100.0
        sample_confidence = (total / max_total) * 100.0
        trust = round((0.85 * genuine_pct) + (0.15 * sample_confidence), 1)
        fraud = round(100.0 - trust, 1)

        if trust >= 85:
            tier = "Clean"
        elif trust >= 60:
            tier = "Watchlist"
        else:
            tier = "High Risk"

        if total >= 10_000:
            confidence = "High"
        elif total >= 2_000:
            confidence = "Medium"
        else:
            confidence = "Low"

        reasons: list[str] = []
        if fraud >= 60:
            reasons.append("High Fake Share")
        elif fraud >= 35:
            reasons.append("Moderate Fake Share")
        if total >= 25_000:
            reasons.append("Large Sample Size")
        elif total < 2_000:
            reasons.append("Low Sample Size")
        if sample_confidence >= 85:
            reasons.append("High Sample Confidence")
        if trust >= 85:
            reasons.append("Highly Reliable Category")
        if not reasons:
            reasons.append("Mixed Quality Signals")

        swing = min(7.0, max(1.5, fraud / 12.0))
        trend = [
            round(max(0.0, min(100.0, trust - (swing * 1.5))), 1),
            round(max(0.0, min(100.0, trust - swing)), 1),
            round(max(0.0, min(100.0, trust - (swing * 0.5))), 1),
            round(max(0.0, min(100.0, trust)), 1),
            round(max(0.0, min(100.0, trust + (swing * 0.35))), 1),
            round(max(0.0, min(100.0, trust + (swing * 0.7))), 1),
        ]

        rows.append(
            {
                "platform": "global",
                "product_id": category_name,
                "trust_score": trust,
                "fraud_score": fraud,
                "scans_count": total,
                "confidence": confidence,
                "tier": tier,
                "reasons": reasons[:3],
                "trend": trend,
                "last_scan_iso": dataset_ts.isoformat(),
                "last_scan_label": dataset_label,
                "last_scan_epoch": dataset_epoch,
                "genuine_count": genuine,
                "fake_count": fake,
                "genuine_share": round(genuine_pct, 1),
                "sample_confidence": round(sample_confidence, 1),
            }
        )

    rows.sort(key=lambda x: (x["trust_score"], x["scans_count"]), reverse=True)
    rows = rows[: max(1, int(max_categories))]

    _GLOBAL_CATEGORY_CACHE["key"] = file_key
    _GLOBAL_CATEGORY_CACHE["rows"] = rows
    _GLOBAL_CATEGORY_CACHE["data_source"] = "global_dataset"
    return list(rows), "global_dataset"


def _build_leaderboard_payload(
    max_products: int = 300,
    tab: str = "trusted",
    search: str = "",
    platform: str = "all",
    date_days: str = "all",
    min_scans: int = 1,
    fraud_band: str = "all",
    sort: str = "trust_desc",
    page: int = 1,
    page_size: int = 12,
) -> dict:
    page = max(1, _safe_int(page, 1))
    page_size = max(1, min(_safe_int(page_size, 12), 500))
    payload = _empty_leaderboard_payload(page=page, page_size=page_size)

    all_rows, source = _build_global_category_rows(max_categories=max_products)
    payload["data_source"] = source
    if not all_rows:
        return payload

    payload["platforms"] = sorted(
        {str(r.get("platform", "")).lower() for r in all_rows if str(r.get("platform", "")).strip()}
    )
    payload["summary"] = {
        "products": int(len(all_rows)),
        "scans": int(sum(int(r.get("scans_count", 0)) for r in all_rows)),
        "high_risk": int(sum(1 for r in all_rows if r.get("tier") == "High Risk")),
        "clean": int(sum(1 for r in all_rows if r.get("tier") == "Clean")),
    }

    q = str(search or "").strip().lower()
    normalized_platform = str(platform or "all").strip().lower()
    normalized_tab = "risk" if str(tab).strip().lower() in {"risk", "suspicious"} else "trusted"
    normalized_sort = str(sort or "trust_desc").strip().lower()
    normalized_fraud_band = str(fraud_band or "all").strip().lower()
    days_value = _safe_int(date_days, -1) if str(date_days).strip().lower() != "all" else -1
    min_scans = max(1, _safe_int(min_scans, 1))

    now_epoch = int(datetime.now(timezone.utc).timestamp())
    filtered = []
    for row in all_rows:
        pid = str(row.get("product_id", "")).lower()
        plat = str(row.get("platform", "")).lower()
        scans_count = _safe_int(row.get("scans_count", 0), 0)
        last_epoch = _safe_int(row.get("last_scan_epoch", 0), 0)

        if q and q not in pid:
            continue
        if normalized_platform != "all" and normalized_platform != plat:
            continue
        if scans_count < min_scans:
            continue
        if days_value > 0:
            if last_epoch <= 0:
                continue
            if (now_epoch - last_epoch) > (days_value * 86400):
                continue
        if not _row_matches_fraud_band(row, normalized_fraud_band):
            continue
        if normalized_tab == "trusted" and _safe_float(row.get("trust_score"), 0.0) < _safe_float(row.get("fraud_score"), 0.0):
            continue
        if normalized_tab == "risk" and _safe_float(row.get("fraud_score"), 0.0) < 20.0:
            continue
        filtered.append(row)

    filtered = _sort_leaderboard_rows(filtered, normalized_sort, normalized_tab)
    for idx, row in enumerate(filtered, start=1):
        row["rank"] = idx

    total_filtered = len(filtered)
    total_pages = max(1, math.ceil(total_filtered / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    page_rows = filtered[start : start + page_size]
    top3 = filtered[:3]

    payload.update(
        {
            "rows": page_rows,
            "top3": top3,
            "total_filtered": int(total_filtered),
            "page": int(page),
            "page_size": int(page_size),
            "total_pages": int(total_pages),
            "rank_offset": int(start),
        }
    )
    return payload


def _infer_platform(product_ref: str) -> str:
    ref = str(product_ref or "").lower()
    if "flipkart.com" in ref:
        return "flipkart"
    if "walmart.com" in ref:
        return "walmart"
    return "amazon"


def _extract_amazon_asin(product_ref: str) -> str:
    raw = str(product_ref or "").strip()
    if re.fullmatch(r"[A-Z0-9]{10}", raw, flags=re.I):
        return raw.upper()
    for pattern in (r"/dp/([A-Z0-9]{10})", r"/gp/product/([A-Z0-9]{10})", r"\b([A-Z0-9]{10})\b"):
        m = re.search(pattern, raw, flags=re.I)
        if m:
            return m.group(1).upper()
    return raw


def _normalize_product_id(platform: str | None, product_id: str) -> str:
    pid = str(product_id or "").strip()
    p = str(platform or "").strip().lower()
    if p == "amazon":
        return _extract_amazon_asin(pid)
    return pid


def _adjusted_rating_from_trust(authentic_rating: float | None, fraud_score: float) -> float | None:
    if authentic_rating is None:
        return None
    # Discount up to 2 stars when fraud risk reaches 100%.
    adjusted = authentic_rating - (max(0.0, min(100.0, fraud_score)) / 100.0) * 2.0
    return round(max(1.0, min(5.0, adjusted)), 1)


def _append_local_trust_history(platform: str, product_id: str, summary: dict) -> None:
    normalized_platform = str(platform or "unknown").strip().lower()
    normalized_product_id = _normalize_product_id(normalized_platform, product_id)

    advanced = summary.get("advanced_analysis", {}) if isinstance(summary, dict) else {}
    fraud = _safe_float(advanced.get("combined_fraud_score"), None)
    if fraud is None:
        fraud = _safe_float(summary.get("avg_fake_probability"), 0.0)
    fraud = round(max(0.0, min(100.0, fraud)), 3)
    trust = round(100.0 - fraud, 3)

    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "platform": normalized_platform,
                "product_id": normalized_product_id,
                "trust_score": trust,
                "fraud_score": fraud,
            }
        ]
    )
    header = not os.path.exists(LOCAL_TRUST_HISTORY_FILE)
    row.to_csv(LOCAL_TRUST_HISTORY_FILE, mode="a", index=False, header=header)


def _json_default(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    try:
        return obj.item()
    except Exception:
        return str(obj)


def _append_local_scan_history(asin: str, summary: dict, chart_data: dict, user_id: str) -> None:
    payload = {
        "asin": str(asin or "").strip(),
        "user_id": str(user_id or "anonymous"),
        "summary": json.loads(json.dumps(summary or {}, default=_json_default)),
        "chart_data": json.loads(json.dumps(chart_data or {}, default=_json_default)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(LOCAL_SCAN_HISTORY_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_local_scan_history(user_id: str | None = None, limit: int = 100) -> list[dict]:
    if not os.path.exists(LOCAL_SCAN_HISTORY_FILE):
        return []
    out: list[dict] = []
    try:
        with open(LOCAL_SCAN_HISTORY_FILE, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if user_id and str(item.get("user_id", "")).strip() not in {"", str(user_id)}:
                    continue
                out.append(item)
    except Exception as exc:
        print(f"[WARN] Local scan history read failed: {exc}")
        return []

    out.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
    return out[: max(1, int(limit))]


def _load_legacy_trust_history_as_scans(limit: int = 100) -> list[dict]:
    if not os.path.exists(LOCAL_TRUST_HISTORY_FILE):
        return []
    try:
        df = pd.read_csv(LOCAL_TRUST_HISTORY_FILE)
    except Exception:
        return []
    if df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)
    df["fraud_score"] = pd.to_numeric(df.get("fraud_score"), errors="coerce")
    df["trust_score"] = pd.to_numeric(df.get("trust_score"), errors="coerce")
    df["platform"] = df.get("platform", "amazon").astype(str)
    df["product_id"] = df.get("product_id", "").astype(str)
    df = df.dropna(subset=["timestamp", "fraud_score", "trust_score"])
    if df.empty:
        return []

    df = df.sort_values("timestamp", ascending=False).head(max(1, int(limit)))
    items: list[dict] = []
    for _, row in df.iterrows():
        asin = str(row.get("product_id", "")).strip()
        platform = str(row.get("platform", "amazon")).strip().lower()
        fraud = round(float(row.get("fraud_score", 0.0)), 1)
        trust = round(float(row.get("trust_score", 0.0)), 1)
        items.append(
            {
                "asin": asin,
                "timestamp": row["timestamp"].isoformat(),
                "summary": {
                    "product_title": f"{platform.title()} Product ({asin})",
                    "product_rating": "",
                    "total_reviews": 0,
                    "avg_fake_probability": fraud,
                },
                "chart_data": {
                    "fake": 0,
                    "genuine": 0,
                    "fake_score": fraud,
                    "genuine_score": trust,
                    "ml_fake_score": fraud,
                },
            }
        )
    return items


def _build_monthly_history(product_id: str, platform: str | None, months: int = 12):
    months = max(1, min(int(months), 24))
    month_labels = pd.date_range(
        end=pd.Timestamp.now(tz="UTC"),
        periods=months,
        freq="MS",
        tz="UTC",
    ).strftime("%Y-%m").tolist()
    template = pd.DataFrame({"month": month_labels})

    requested_platform = str(platform or "").strip().lower() or None
    pid = _normalize_product_id(requested_platform or _infer_platform(product_id), product_id)

    trust_df = pd.DataFrame(columns=["month", "trust_score", "fraud_score"])
    if os.path.exists(LOCAL_TRUST_HISTORY_FILE):
        try:
            df = pd.read_csv(LOCAL_TRUST_HISTORY_FILE)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                df = df.dropna(subset=["timestamp"])
                df["product_id"] = df["product_id"].astype(str)
                df["platform"] = df["platform"].astype(str)
                df["platform_norm"] = df["platform"].str.strip().str.lower()
                df["product_id_norm"] = df.apply(
                    lambda r: _normalize_product_id(r.get("platform_norm", ""), r.get("product_id", "")),
                    axis=1,
                )
                df["trust_score"] = pd.to_numeric(df["trust_score"], errors="coerce")
                df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce")
                df = df.dropna(subset=["trust_score", "fraud_score"])
                df = df[df["product_id_norm"] == pid]
                if requested_platform:
                    df = df[df["platform_norm"] == requested_platform]
                if not df.empty:
                    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
                    trust_df = (
                        df.groupby("month", as_index=False)[["trust_score", "fraud_score"]]
                        .mean()
                        .round(3)
                    )
        except Exception as exc:
            print(f"[WARN] Trust history monthly aggregation failed: {exc}")

    review_count_by_month: dict[str, int] = {m: 0 for m in month_labels}
    scan_events_by_month: dict[str, int] = {m: 0 for m in month_labels}
    if os.path.exists(LOCAL_SCAN_HISTORY_FILE):
        try:
            with open(LOCAL_SCAN_HISTORY_FILE, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    ts = pd.to_datetime(item.get("timestamp"), errors="coerce", utc=True)
                    if pd.isna(ts):
                        continue
                    month_key = pd.Timestamp(ts).strftime("%Y-%m")
                    if month_key not in review_count_by_month:
                        continue

                    summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
                    chart_data = item.get("chart_data") if isinstance(item.get("chart_data"), dict) else {}
                    item_ref = str(
                        item.get("asin")
                        or item.get("product_id")
                        or summary.get("asin")
                        or summary.get("product_id")
                        or ""
                    ).strip()
                    if not item_ref:
                        continue
                    item_platform = str(
                        item.get("platform")
                        or summary.get("platform")
                        or _infer_platform(item_ref)
                    ).strip().lower()
                    item_pid = _normalize_product_id(item_platform, item_ref)
                    if item_pid != pid:
                        continue
                    if requested_platform and item_platform != requested_platform:
                        continue

                    monthly_reviews = _safe_int(summary.get("total_reviews"), 0)
                    if monthly_reviews <= 0:
                        monthly_reviews = _safe_int(chart_data.get("fake"), 0) + _safe_int(chart_data.get("genuine"), 0)
                    monthly_reviews = max(0, int(monthly_reviews))

                    review_count_by_month[month_key] = review_count_by_month.get(month_key, 0) + monthly_reviews
                    scan_events_by_month[month_key] = scan_events_by_month.get(month_key, 0) + 1
        except Exception as exc:
            print(f"[WARN] Scan history monthly aggregation failed: {exc}")

    merged = template.merge(trust_df, on="month", how="left")
    merged["review_count"] = merged["month"].map(review_count_by_month).fillna(0).astype(int)
    merged["scan_events"] = merged["month"].map(scan_events_by_month).fillna(0).astype(int)
    records = merged.to_dict(orient="records")
    for item in records:
        trust_val = item.get("trust_score")
        fraud_val = item.get("fraud_score")
        item["trust_score"] = None if pd.isna(trust_val) else round(float(trust_val), 3)
        item["fraud_score"] = None if pd.isna(fraud_val) else round(float(fraud_val), 3)
        item["review_count"] = max(0, _safe_int(item.get("review_count"), 0))
        item["scan_events"] = max(0, _safe_int(item.get("scan_events"), 0))
    return records


def _compute_platform_trust(platform: str, product_ref: str, pages: int = 20, provided_score=None):
    """
    Resolves trust score for a single platform.
    For Walmart, pass `provided_score` unless you wire an external source.
    """
    if provided_score is not None:
        trust_score = round(max(0.0, min(100.0, float(provided_score))), 1)
        return {
            "platform": platform,
            "product_ref": product_ref,
            "trust_score": trust_score,
            "fraud_score": round(100.0 - trust_score, 1),
            "source": "provided",
            "available": True,
        }

    if platform == "walmart":
        return {
            "platform": platform,
            "product_ref": product_ref,
            "available": False,
            "error": "Walmart fetch not configured. Provide `platform_scores.walmart`.",
        }

    target_reviews = max(1, min(int(pages) * 10, 10000))
    summary, _ = analyze_product(product_ref, pages=pages, max_reviews=target_reviews)
    total_reviews = int(summary.get("total_reviews", 0))
    advanced = summary.get("advanced_analysis", {})
    fraud = _safe_float(advanced.get("combined_fraud_score"), None)
    if fraud is None:
        fraud = _safe_float(summary.get("avg_fake_probability"), 0.0)
    fraud = round(max(0.0, min(100.0, float(fraud))), 1)
    trust = round(100.0 - fraud, 1)
    rating = _safe_float(summary.get("product_rating"), None)
    adjusted = _adjusted_rating_from_trust(rating, fraud)
    return {
        "platform": platform,
        "product_ref": product_ref,
        "available": total_reviews > 0,
        "total_reviews": total_reviews,
        "trust_score": trust,
        "fraud_score": fraud,
        "authentic_rating": rating,
        "adjusted_rating": adjusted,
        "source": "analyze_product",
    }

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _to_utc_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            return parsed.to_pydatetime()
    except Exception:
        return None
    return None


def _extract_profile_scan_metrics(item: dict) -> tuple[float | None, datetime | None, str]:
    chart_data = item.get("chart_data") if isinstance(item, dict) else {}
    chart_data = chart_data if isinstance(chart_data, dict) else {}
    summary = item.get("summary") if isinstance(item, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    advanced = summary.get("advanced_analysis") if isinstance(summary.get("advanced_analysis"), dict) else {}

    fraud_score = _safe_float(chart_data.get("fake_score"), None)
    if fraud_score is None:
        fraud_score = _safe_float(advanced.get("combined_fraud_score"), None)
    if fraud_score is None:
        fraud_score = _safe_float(summary.get("avg_fake_probability"), None)

    trust_score = None
    if fraud_score is not None:
        trust_score = round(max(0.0, min(100.0, 100.0 - float(fraud_score))), 1)

    timestamp = _to_utc_datetime(item.get("timestamp"))
    product_id = str(
        item.get("asin")
        or item.get("product_id")
        or summary.get("asin")
        or summary.get("product_id")
        or ""
    ).strip()
    return trust_score, timestamp, product_id


def _build_profile_view_data(user_id: str) -> dict:
    user_id = str(user_id or "").strip()
    default_name = str(session.get("user_name") or "TrustLens User").strip() or "TrustLens User"
    display_name = default_name
    email = str(session.get("user_email") or "").strip()
    avatar_url = str(session.get("user_photo") or "").strip()

    if "@" in default_name and " " not in default_name:
        email = email or default_name

    if db is not None and user_id:
        try:
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict() or {}
                db_display_name = str(user_data.get("displayName") or "").strip()
                db_email = str(user_data.get("email") or "").strip()
                db_avatar = str(user_data.get("photoURL") or "").strip()
                if db_display_name:
                    display_name = db_display_name
                if db_email:
                    email = db_email
                if db_avatar:
                    avatar_url = db_avatar
        except Exception as exc:
            print(f"[WARN] Profile user lookup failed: {exc}")

    if user_id and (not email or not display_name or not avatar_url):
        try:
            auth_user = auth.get_user(user_id)
            auth_email = str(getattr(auth_user, "email", "") or "").strip()
            auth_name = str(getattr(auth_user, "display_name", "") or "").strip()
            auth_avatar = str(getattr(auth_user, "photo_url", "") or "").strip()
            if auth_email and not email:
                email = auth_email
            if auth_name and (not display_name or display_name == "TrustLens User"):
                display_name = auth_name
            if auth_avatar and not avatar_url:
                avatar_url = auth_avatar
        except Exception as exc:
            print(f"[WARN] Firebase auth user lookup failed: {exc}")

    if display_name:
        session["user_name"] = display_name
    if email:
        session["user_email"] = email
    if avatar_url:
        session["user_photo"] = avatar_url

    if not avatar_url:
        avatar_url = DEFAULT_AVATAR_URL
    if not email:
        email = "Email not available"

    history_items: list[dict] = []
    data_source = "none"

    if db is not None and user_id:
        try:
            docs = db.collection("scans").where("user_id", "==", user_id).limit(500).stream()
            firestore_rows: list[dict] = []
            for doc in docs:
                payload = doc.to_dict() or {}
                ts = payload.get("timestamp")
                if ts is not None and hasattr(ts, "isoformat"):
                    payload["timestamp"] = ts.isoformat()
                firestore_rows.append(payload)
            if firestore_rows:
                history_items = firestore_rows
                data_source = "firestore_scans"
        except Exception as exc:
            print(f"[WARN] Profile firestore scan read failed: {exc}")

    if not history_items:
        local_rows = _load_local_scan_history(user_id=user_id, limit=500)
        if local_rows:
            history_items = local_rows
            data_source = "local_scan_log"

    if not history_items:
        legacy_rows = _load_legacy_trust_history_as_scans(limit=500)
        if legacy_rows:
            history_items = legacy_rows
            data_source = "local_trust_csv"

    trust_scores: list[float] = []
    timestamps: list[datetime] = []
    products: set[str] = set()

    for item in history_items:
        trust_score, ts, product_id = _extract_profile_scan_metrics(item)
        if trust_score is not None:
            trust_scores.append(float(trust_score))
        if ts is not None:
            timestamps.append(ts)
        if product_id:
            products.add(product_id)

    total_scans = int(len(history_items))
    unique_products = int(len(products))
    avg_trust = round((sum(trust_scores) / len(trust_scores)), 1) if trust_scores else 0.0

    now_utc = datetime.now(timezone.utc)
    thirty_days_ago = now_utc - timedelta(days=30)
    sixty_days_ago = now_utc - timedelta(days=60)
    recent_30 = sum(1 for ts in timestamps if ts >= thirty_days_ago)
    previous_30 = sum(1 for ts in timestamps if sixty_days_ago <= ts < thirty_days_ago)

    if recent_30 == 0 and previous_30 == 0:
        monthly_change_text = "No 30d activity"
    elif previous_30 == 0:
        monthly_change_text = f"+{recent_30} new scans"
    else:
        delta_pct = ((recent_30 - previous_30) / previous_30) * 100.0
        monthly_change_text = f"{delta_pct:+.0f}% vs last 30d"

    last_scan_at = max(timestamps) if timestamps else None
    if last_scan_at is None:
        last_analysis_text = "No scans yet"
        activity_label = "No activity"
    else:
        last_analysis_text = last_scan_at.strftime("%d %b %Y, %I:%M %p UTC")
        gap = now_utc - last_scan_at
        if gap < timedelta(days=1):
            activity_label = "Active today"
        elif gap < timedelta(days=2):
            activity_label = "Active 1 day ago"
        else:
            activity_label = f"Active {gap.days} days ago"

    if avg_trust >= 85:
        trust_tier_text = "High confidence pattern"
    elif avg_trust >= 70:
        trust_tier_text = "Stable pattern"
    elif avg_trust >= 55:
        trust_tier_text = "Needs review"
    else:
        trust_tier_text = "High risk pattern"

    return {
        "user_id": user_id,
        "name": display_name,
        "email": email,
        "avatar_url": avatar_url,
        "total_scans": total_scans,
        "total_scans_text": _format_int(total_scans),
        "avg_trust": avg_trust,
        "avg_trust_text": f"{avg_trust:.1f}%",
        "monthly_change_text": monthly_change_text,
        "trust_tier_text": trust_tier_text,
        "last_analysis_text": last_analysis_text,
        "activity_label": activity_label,
        "unique_products": unique_products,
        "unique_products_text": _format_int(unique_products),
        "plan_text": "Community Plan - No billing configured",
        "data_source": data_source,
    }


@app.after_request
def add_cors_headers(response):
    # Allows browser extension requests to Flask APIs.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Firebase init
cred_path = "firebase-credentials.json"
db = None
USE_MOCK_AUTH = False

if os.path.exists(cred_path):
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized.")
    except Exception as e:
        print(f"Warning: Firebase init failed: {e}")
        USE_MOCK_AUTH = True
else:
    print(f"Warning: '{cred_path}' not found. Using MOCK_AUTH mode.")
    USE_MOCK_AUTH = True

# =============================================================================
# ROUTES
# =============================================================================
@app.route("/", methods=["GET"])
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", home_metrics=_build_home_metrics())

@app.route("/analyze", methods=["GET"])
def analyze():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("analyze.html")

@app.route("/login",      methods=["GET"])
def login():       return render_template("login.html")

@app.route("/register",   methods=["GET"])
def register():    return render_template("register.html")

@app.route("/history",    methods=["GET"])
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("scan.html")

@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("leaderboard.html", leaderboard_payload=_build_leaderboard_payload())


@app.route("/api/leaderboard", methods=["GET", "OPTIONS"])
def leaderboard_api():
    if request.method == "OPTIONS":
        return ("", 204)
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        payload = _build_leaderboard_payload(
            max_products=_safe_int(request.args.get("max_products"), 300),
            tab=request.args.get("tab", "trusted"),
            search=request.args.get("search", ""),
            platform=request.args.get("platform", "all"),
            date_days=request.args.get("date_days", "all"),
            min_scans=_safe_int(request.args.get("min_scans"), 1),
            fraud_band=request.args.get("fraud_band", "all"),
            sort=request.args.get("sort", "trust_desc"),
            page=_safe_int(request.args.get("page"), 1),
            page_size=_safe_int(request.args.get("page_size"), 12),
        )
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/profile",    methods=["GET"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user_id = str(session.get("user_id", "") or "").strip()
    return render_template("profile.html", profile_data=_build_profile_view_data(user_id))

@app.route("/logout")
def logout():
    session.pop("user_id",   None)
    session.pop("user_name", None)
    session.pop("user_email", None)
    session.pop("user_photo",None)
    return redirect(url_for("login"))


@app.route("/api/profile", methods=["GET", "POST"])
def api_profile():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = str(session.get("user_id", "") or "").strip()

    if request.method == "GET":
        return jsonify({"profile": _build_profile_view_data(user_id)})

    data = request.get_json(silent=True) or {}
    display_name = re.sub(r"\s+", " ", str(data.get("display_name") or "").strip())[:80]
    photo_url = str(data.get("photo_url") or "").strip()[:600]

    if display_name:
        session["user_name"] = display_name
    if photo_url:
        session["user_photo"] = photo_url

    if db is not None:
        update_payload = {"uid": user_id, "updatedAt": firestore.SERVER_TIMESTAMP}
        if display_name:
            update_payload["displayName"] = display_name
        if photo_url:
            update_payload["photoURL"] = photo_url
        try:
            db.collection("users").document(user_id).set(update_payload, merge=True)
        except Exception as exc:
            print(f"[WARN] Profile update firestore write failed: {exc}")

    return jsonify({"status": "success", "profile": _build_profile_view_data(user_id)})


@app.route("/api/profile/delete-request", methods=["POST"])
def api_profile_delete_request():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = str(session.get("user_id", "") or "").strip()
    if db is not None and user_id:
        try:
            db.collection("users").document(user_id).set(
                {
                    "deleteRequested": True,
                    "deleteRequestedAt": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
        except Exception as exc:
            print(f"[WARN] Delete request flag write failed: {exc}")

    session.pop("user_id", None)
    session.pop("user_name", None)
    session.pop("user_email", None)
    session.pop("user_photo", None)
    return jsonify({"status": "success", "redirect": url_for("login")})

# Auth
@app.route("/api/sessionLogin", methods=["POST"])
def session_login():
    try:
        data     = request.get_json(silent=True) or {}
        id_token = data.get("idToken")
        if not id_token:
            return jsonify({"error": "No token provided"}), 400

        name_hint = re.sub(r"\s+", " ", str(data.get("displayName") or data.get("name") or "").strip())[:80]
        email_hint = str(data.get("email") or "").strip().lower()[:160]
        photo_hint = str(data.get("photoURL") or data.get("photo_url") or "").strip()[:600]

        def _mock_uid() -> str:
            if email_hint:
                digest = hashlib.sha1(email_hint.encode("utf-8")).hexdigest()[:12]
                return f"mock_{digest}"
            return "mock_local_user"
        
        # --- Bypass for Local Development ---
        if USE_MOCK_AUTH and id_token == "local-dev-mock-token":
            uid = _mock_uid()
            session["user_id"]    = uid
            session["user_name"]  = name_hint or (email_hint.split("@")[0] if "@" in email_hint else "Developer")
            session["user_email"] = email_hint or "developer@local.mock"
            session["user_photo"] = photo_hint or None
            return jsonify({"status": "success"})
        # ------------------------------------

        try:
            decoded = auth.verify_id_token(id_token)
            uid     = decoded["uid"]
            session["user_id"]    = uid
            session["user_name"]  = decoded.get("name") or decoded.get("email", "User")
            session["user_email"] = decoded.get("email") or ""
            session["user_photo"] = decoded.get("picture")
            if db:
                try:
                    db.collection("users").document(uid).set({
                        "uid":         uid,
                        "email":       decoded.get("email"),
                        "displayName": session["user_name"],
                        "photoURL":    session["user_photo"],
                        "lastLogin":   firestore.SERVER_TIMESTAMP,
                    }, merge=True)
                except Exception as e:
                    print(f"Firestore write error: {e}")
            return jsonify({"status": "success"})
        except Exception as e:
            if USE_MOCK_AUTH:
                print(f"Bypassing real auth failure: {e}")
                uid = _mock_uid()
                session["user_id"]    = uid
                session["user_name"]  = name_hint or (email_hint.split("@")[0] if "@" in email_hint else "Developer (Mock)")
                session["user_email"] = email_hint or "developer@local.mock"
                session["user_photo"] = photo_hint or None

                if db:
                    try:
                        db.collection("users").document(uid).set({
                            "uid": uid,
                            "email": session["user_email"],
                            "displayName": session["user_name"],
                            "photoURL": session["user_photo"],
                            "lastLogin": firestore.SERVER_TIMESTAMP,
                            "mockAuth": True,
                        }, merge=True)
                    except Exception as write_err:
                        print(f"[WARN] Mock auth profile write failed: {write_err}")
                return jsonify({"status": "success"})
            
            print(f"Login error: {e}")
            return jsonify({"error": "Invalid token"}), 401
    except Exception as e:
        print(f"Outer login error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/uploadAvatar", methods=["POST"])
def upload_avatar():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if file and allowed_file(file.filename):
        ext      = file.filename.rsplit(".", 1)[1].lower()
        filename = secure_filename(f"avatar_{session['user_id']}_{uuid.uuid4().hex[:8]}.{ext}")
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        avatar_url = url_for("static", filename=f"uploads/avatars/{filename}")
        session["user_photo"] = avatar_url
        if db is not None:
            try:
                db.collection("users").document(str(session.get("user_id"))).set(
                    {
                        "uid": str(session.get("user_id")),
                        "photoURL": avatar_url,
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    },
                    merge=True,
                )
            except Exception as exc:
                print(f"[WARN] Avatar firestore write failed: {exc}")
        return jsonify(
            {
                "status": "success",
                "url": avatar_url,
                "profile": _build_profile_view_data(str(session.get("user_id", "") or "")),
            }
        )
    return jsonify({"error": "File type not allowed"}), 400

@app.route("/api/history", methods=["GET"])
def api_history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = str(session.get("user_id", "") or "").strip()

    if db is not None:
        try:
            docs = (
                db.collection("scans")
                .where("user_id", "==", user_id)
                .stream()
            )
            history_data = []
            for doc in docs:
                data = doc.to_dict() or {}
                if "timestamp" in data and data["timestamp"] and hasattr(data["timestamp"], "isoformat"):
                    data["timestamp"] = data["timestamp"].isoformat()
                history_data.append(data)
            history_data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            if history_data:
                return jsonify({"history": history_data[:100], "source": "firestore"})
        except Exception as e:
            print(f"[WARN] Firestore history read failed: {e}")

    local_history = _load_local_scan_history(user_id=user_id, limit=100)
    if local_history:
        return jsonify({"history": local_history, "source": "local_scan_log"})

    legacy = _load_legacy_trust_history_as_scans(limit=100)
    return jsonify({"history": legacy, "source": "local_trust_csv" if legacy else "none"})

# =============================================================================
# MAIN ANALYSIS
# =============================================================================
@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    try:
        # Accept both JSON requests (web app/extension) and form posts.
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = {}
        if not data:
            data = request.form.to_dict(flat=True)

        asin = str(data.get("asin") or "").strip()
        if not asin:
            return jsonify({"error": "Please enter a product URL or ASIN."}), 400

        review_limit_raw = data.get("review_limit")
        if review_limit_raw is None:
            # Backward compatibility: older UI may send `pages`; otherwise use full-scan default.
            legacy_pages_raw = data.get("pages")
            if legacy_pages_raw is None or str(legacy_pages_raw).strip() == "":
                review_limit_raw = 10000
            else:
                try:
                    legacy_pages = int(legacy_pages_raw)
                except (TypeError, ValueError):
                    legacy_pages = 1000
                review_limit_raw = legacy_pages * 10
        try:
            review_limit = int(review_limit_raw)
        except (TypeError, ValueError):
            review_limit = 10000
        review_limit = max(1, min(review_limit, 10000))
        pages = max(1, min(int(math.ceil(review_limit / 10.0)), 1000))

        summary, df_results = analyze_product(asin, pages=pages, max_reviews=review_limit)

        # ── Clear error if no reviews found ───────────────────────────────────
        if df_results.empty or summary.get("total_reviews", 0) == 0:
            ml_fake_score = float(summary.get("avg_fake_probability", 0.0))
            ml_fake_score = round(max(0.0, min(100.0, ml_fake_score)), 1)
            combined_score = _safe_float(
                summary.get("advanced_analysis", {}).get("combined_fraud_score"),
                ml_fake_score,
            )
            fake_score = round(max(0.0, min(100.0, float(combined_score))), 1)
            chart_data = {
                "fake": 0,
                "genuine": 0,
                "fake_score": fake_score,
                "genuine_score": round(100.0 - fake_score, 1),
                "ml_fake_score": ml_fake_score,
            }
            try:
                _append_local_trust_history(
                    platform=_infer_platform(asin),
                    product_id=str(asin).strip(),
                    summary=summary,
                )
                _append_local_scan_history(
                    asin=asin,
                    summary=summary,
                    chart_data=chart_data,
                    user_id=str(session.get("user_id", "anonymous")),
                )
            except Exception as e:
                print(f"Local history write error: {e}")
            return jsonify(
                {
                    "summary": summary,
                    "chart_data": chart_data,
                    "results": {
                        "review": [],
                        "prediction": [],
                        "confidence": [],
                        "fake_probability": [],
                        "date": [],
                        "user_name": [],
                        "explanation": [],
                    },
                    "asin": asin,
                    "advanced_analysis": summary.get("advanced_analysis", {}),
                    "warning": (
                        "No review text was fetched from live sources for this run. "
                        "Please verify network/API access for fuller coverage."
                    ),
                }
            ), 200

        # ── Build results ─────────────────────────────────────────────────────
        if "prediction" in df_results.columns:
            fake_count    = int((df_results["prediction"] == "Fake").sum())
            genuine_count = int((df_results["prediction"] == "Genuine").sum())
            results = {
                "review":           df_results["review"].tolist(),
                "prediction":       df_results["prediction"].tolist(),
                "confidence":       [round(float(c), 1) for c in df_results["confidence"].tolist()],
                "fake_probability": [round(float(c), 1) for c in df_results["fake_probability"].tolist()]
                                    if "fake_probability" in df_results.columns else [],
                "date":             df_results["date"].tolist()
                                    if "date" in df_results.columns else [""] * len(df_results),
                "user_name":        df_results["user_name"].tolist()
                                    if "user_name" in df_results.columns else ["unknown"] * len(df_results),
                "explanation":      df_results["explanation"].tolist()
                                    if "explanation" in df_results.columns
                                    else [[] for _ in range(len(df_results))],
            }
        else:
            fake_count = genuine_count = 0
            results    = {"review": [], "prediction": [], "confidence": [],
                          "fake_probability": [], "date": [], "user_name": [], "explanation": []}

        ml_fake_score = float(summary.get("avg_fake_probability", 0.0))
        ml_fake_score = round(max(0.0, min(100.0, ml_fake_score)), 1)
        combined_score = _safe_float(
            summary.get("advanced_analysis", {}).get("combined_fraud_score"),
            ml_fake_score,
        )
        fake_score = round(max(0.0, min(100.0, float(combined_score))), 1)
        genuine_score = round(100.0 - fake_score, 1)

        chart_data = {
            "fake":          fake_count,
            "genuine":       genuine_count,
            "fake_score":    fake_score,
            "genuine_score": genuine_score,
            "ml_fake_score": ml_fake_score,
        }

        # Local history store (used by extension trend chart API)
        try:
            _append_local_trust_history(
                platform=_infer_platform(asin),
                product_id=str(asin).strip(),
                summary=summary,
            )
            _append_local_scan_history(
                asin=asin,
                summary=summary,
                chart_data=chart_data,
                user_id=str(session.get("user_id", "anonymous")),
            )
        except Exception as e:
            print(f"Local history write error: {e}")

        # Firestore save
        if db is not None:
            try:
                db.collection("scans").document().set({
                    "asin":       asin,
                    "user_id":    session.get("user_id", "anonymous"),
                    "summary":    summary,
                    "chart_data": chart_data,
                    "timestamp":  firestore.SERVER_TIMESTAMP,
                })
            except Exception as e:
                print(f"Firestore write error: {e}")

        return jsonify({
            "summary":           summary,
            "chart_data":        chart_data,
            "results":           results,
            "asin":              asin,
            "advanced_analysis": summary.get("advanced_analysis", {}),
            "warning":           summary.get("coverage_warning"),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500


# =============================================================================
# PRODUCT INTELLIGENCE APIs
# =============================================================================
@app.route("/api/extension/trust-badge", methods=["GET", "OPTIONS"])
def extension_trust_badge():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        product_id = (request.args.get("product_id") or request.args.get("asin") or "").strip()
        product_url = (request.args.get("url") or "").strip()
        product_ref = product_id or product_url
        if not product_ref:
            return jsonify({"error": "Missing `product_id` or `url`."}), 400

        platform = (request.args.get("platform") or _infer_platform(product_ref)).strip().lower()
        pages = max(1, min(int(request.args.get("pages", 15)), 30))

        result = _compute_platform_trust(platform=platform, product_ref=product_ref, pages=pages)
        if not result.get("available", False):
            return jsonify(result), 200

        fraud = float(result.get("fraud_score", 0.0))
        trust = float(result.get("trust_score", 0.0))
        authentic_rating = result.get("authentic_rating")
        adjusted_rating = result.get("adjusted_rating")

        # Extension badge-oriented payload.
        payload = {
            "platform": platform,
            "product_id": product_ref,
            "trust_score": trust,
            "fraud_risk": fraud,
            "authentic_rating": authentic_rating,
            "adjusted_rating": adjusted_rating,
            "badge_text": (
                f"Authentic Rating: {authentic_rating:.1f} -> Adjusted: {adjusted_rating:.1f}"
                if isinstance(authentic_rating, (int, float)) and isinstance(adjusted_rating, (int, float))
                else f"Trust Score: {trust:.1f}"
            ),
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trust-history", methods=["GET", "OPTIONS"])
def trust_history():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        product_id = (request.args.get("product_id") or request.args.get("asin") or "").strip()
        if not product_id:
            return jsonify({"error": "Missing `product_id` or `asin` query param."}), 400
        platform = (request.args.get("platform") or "").strip().lower() or None
        months = max(1, min(int(request.args.get("months", 12)), 24))
        history = _build_monthly_history(product_id=product_id, platform=platform, months=months)
        return jsonify(
            {
                "product_id": product_id,
                "platform": platform,
                "months": months,
                "history": history,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cross-platform/compare", methods=["POST", "OPTIONS"])
def cross_platform_compare():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.json or {}
        pages = max(1, min(int(data.get("pages", 20)), 30))
        refs = data.get("product_refs") or {}
        supplied_scores = data.get("platform_scores") or {}

        # Convenience input shape.
        if not refs:
            refs = {
                "amazon": data.get("amazon"),
                "walmart": data.get("walmart"),
                "flipkart": data.get("flipkart"),
            }
        refs = {k: v for k, v in refs.items() if v}

        platforms = ("amazon", "walmart", "flipkart")
        results = {}
        trust_scores = {}

        for platform in platforms:
            provided = supplied_scores.get(platform)
            product_ref = refs.get(platform, "")

            if product_ref:
                item = _compute_platform_trust(
                    platform=platform,
                    product_ref=product_ref,
                    pages=pages,
                    provided_score=provided,
                )
            elif provided is not None:
                item = _compute_platform_trust(
                    platform=platform,
                    product_ref="",
                    pages=pages,
                    provided_score=provided,
                )
            else:
                item = {
                    "platform": platform,
                    "available": False,
                    "error": "No product reference or precomputed score provided.",
                }

            results[platform] = item
            if item.get("available", False):
                trust_scores[platform] = float(item["trust_score"])

        discrepancy = calculate_discrepancy_score(trust_scores)
        return jsonify(
            {
                "platform_results": results,
                "discrepancy": discrepancy,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reviewer-network", methods=["POST", "OPTIONS"])
def reviewer_network():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.json or {}
        reviews = data.get("reviews", [])
        if not isinstance(reviews, list) or not reviews:
            return jsonify({"error": "Provide `reviews` as a non-empty list."}), 400
        df = pd.DataFrame(reviews)
        result = detect_coordinated_bot_networks(
            df=df,
            user_col=data.get("user_col", "user_id"),
            product_col=data.get("product_col", "product_id"),
            timestamp_col=data.get("timestamp_col", "timestamp"),
            time_window=data.get("time_window", "48h"),
            min_shared_products=int(data.get("min_shared_products", 2)),
            min_co_review_events=int(data.get("min_co_review_events", 2)),
            min_nodes=int(data.get("min_nodes", 3)),
            min_density=float(data.get("min_density", 0.45)),
            min_avg_edge_weight=float(data.get("min_avg_edge_weight", 2.5)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
