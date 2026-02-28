from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from analyzer import analyze_product
import json

app = Flask(__name__)
app.secret_key = "super_secret_mockup_key"  # Needed for session

@app.route("/", methods=["GET"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form.get("email")
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    # Very basic mockup register logic: just log them in
    if request.method == "POST":
        session["user"] = request.form.get("email")
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    try:
        data = request.json
        asin = data.get("asin")
        if not asin:
            return jsonify({"error": "No ASIN or URL provided."}), 400

        pages = int(data.get("pages", 1))
        
        summary, df_results = analyze_product(asin, pages)
        
        if not df_results.empty and "prediction" in df_results.columns:
            fake_count = int((df_results["prediction"] == "Fake").sum())
            genuine_count = int((df_results["prediction"] == "Genuine").sum())
            
            # Export full results as parallel arrays for the Review Feed
            results = {
                "review":     df_results["review"].tolist(),
                "prediction": df_results["prediction"].tolist(),
                "confidence": [round(float(c), 1) for c in df_results["confidence"].tolist()]
            }
        else:
            fake_count = 0
            genuine_count = 0
            results = {"review": [], "prediction": [], "confidence": []}
            
        chart_data = {
            "fake": fake_count,
            "genuine": genuine_count,
        }
        
        return jsonify({
            "summary": summary,
            "chart_data": chart_data,
            "results": results,   # <-- new parallel arrays for the Intelligence Feed
            "asin": asin
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
