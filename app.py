from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
from analyzer import analyze_product
import json
from werkzeug.utils import secure_filename
import uuid

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

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Firebase init
cred_path = "firebase-credentials.json"
db = None
if os.path.exists(cred_path):
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized.")
    except Exception as e:
        print(f"Warning: Firebase init failed: {e}")
else:
    print(f"Warning: '{cred_path}' not found.")

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
        return jsonify({"error": "Invalid token"}), 401

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
        data = request.json
        asin = (data.get("asin") or "").strip()
        if not asin:
            return jsonify({"error": "Please enter a product URL or ASIN."}), 400

        pages = max(1, min(int(data.get("pages", 50)), 50))

        summary, df_results = analyze_product(asin, pages)

        # ── Clear error if no reviews found ───────────────────────────────────
        if df_results.empty or summary.get("total_reviews", 0) == 0:
            return jsonify({
                "error": (
                    "No reviews could be fetched for this product. "
                    "This can happen if: (1) the product has very few text reviews, "
                    "(2) Amazon is temporarily blocking requests. "
                    "Please try again in a few seconds or try a different product."
                )
            }), 400

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
                "explanation":      df_results["explanation"].tolist()
                                    if "explanation" in df_results.columns
                                    else [[] for _ in range(len(df_results))],
            }
        else:
            fake_count = genuine_count = 0
            results    = {"review": [], "prediction": [], "confidence": [],
                          "fake_probability": [], "date": [], "explanation": []}

        fake_score    = float(summary.get("avg_fake_probability", 0.0))
        fake_score    = round(max(0.0, min(100.0, fake_score)), 1)
        genuine_score = round(100.0 - fake_score, 1)

        chart_data = {
            "fake":          fake_count,
            "genuine":       genuine_count,
            "fake_score":    fake_score,
            "genuine_score": genuine_score,
        }

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
            "summary":    summary,
            "chart_data": chart_data,
            "results":    results,
            "asin":       asin,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)