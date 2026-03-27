from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import math
import re
from analyzer import analyze_product
import json
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime, timezone
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


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


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


def _build_monthly_history(product_id: str, platform: str | None, months: int = 12):
    months = max(1, min(int(months), 24))
    if not os.path.exists(LOCAL_TRUST_HISTORY_FILE):
        return []

    df = pd.read_csv(LOCAL_TRUST_HISTORY_FILE)
    if df.empty:
        return []

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

    requested_platform = str(platform or "").strip().lower() or None
    pid = _normalize_product_id(requested_platform or _infer_platform(product_id), product_id)
    df = df[df["product_id_norm"] == pid]
    if requested_platform:
        df = df[df["platform_norm"] == requested_platform]

    now = pd.Timestamp.now(tz="UTC").to_period("M")
    month_labels = [(now - i).strftime("%Y-%m") for i in range(months - 1, -1, -1)]
    template = pd.DataFrame({"month": month_labels})

    if df.empty:
        template["trust_score"] = None
        template["fraud_score"] = None
        return template.to_dict(orient="records")

    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    agg = (
        df.groupby("month", as_index=False)[["trust_score", "fraud_score"]]
        .mean()
        .round(3)
    )
    merged = template.merge(agg, on="month", how="left")
    merged["trust_score"] = merged["trust_score"].where(pd.notna(merged["trust_score"]), None)
    merged["fraud_score"] = merged["fraud_score"].where(pd.notna(merged["fraud_score"]), None)
    return merged.to_dict(orient="records")


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
    return render_template("home.html")

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
    return render_template("leaderboard.html")

@app.route("/profile",    methods=["GET"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("profile.html")

@app.route("/logout")
def logout():
    session.pop("user_id",   None)
    session.pop("user_name", None)
    session.pop("user_photo",None)
    return redirect(url_for("login"))

# Auth
@app.route("/api/sessionLogin", methods=["POST"])
def session_login():
    try:
        data     = request.json
        id_token = data.get("idToken")
        if not id_token:
            return jsonify({"error": "No token provided"}), 400
        
        # --- Bypass for Local Development ---
        if USE_MOCK_AUTH and id_token == "local-dev-mock-token":
            uid = "abc123mock"
            session["user_id"]    = uid
            session["user_name"]  = "Developer"
            session["user_photo"] = None
            return jsonify({"status": "success"})
        # ------------------------------------

        try:
            decoded = auth.verify_id_token(id_token)
            uid     = decoded["uid"]
            session["user_id"]    = uid
            session["user_name"]  = decoded.get("name") or decoded.get("email", "User")
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
                uid = "abc123mock"
                session["user_id"]    = uid
                session["user_name"]  = "Developer (Mock)"
                session["user_photo"] = None
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
        return jsonify({"status": "success",
                        "url": url_for("static", filename=f"uploads/avatars/{filename}")})
    return jsonify({"error": "File type not allowed"}), 400

@app.route("/api/history", methods=["GET"])
def api_history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if db is None:
        return jsonify({"error": "Database not configured"}), 500
    try:
        docs = (db.collection("scans")
                  .where("user_id", "==", session["user_id"])
                  .stream())
        history_data = []
        for doc in docs:
            data = doc.to_dict()
            if "timestamp" in data and data["timestamp"]:
                data["timestamp"] = data["timestamp"].isoformat()
            history_data.append(data)
        history_data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jsonify({"history": history_data[:50]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
