from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
from analyzer import analyze_product
import json
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.secret_key = "super_secret_mockup_key"  # Needed for session

# Configuration for Image Uploads
UPLOAD_FOLDER = 'static/uploads/avatars'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Firebase Admin
cred_path = "firebase-credentials.json"
db = None
if os.path.exists(cred_path):
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        # Initialize Firestore
        db = firestore.client()
        print("Firebase Admin and Firestore initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Firebase Admin: {e}")
else:
    print(f"Warning: '{cred_path}' not found. Firebase Auth and Database will not work until you provide it.")

@app.route("/", methods=["GET"])
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

@app.route("/register", methods=["GET"])
def register():
    return render_template("register.html")

@app.route("/api/sessionLogin", methods=["POST"])
def session_login():
    try:
        data = request.json
        id_token = data.get("idToken")
        if not id_token:
            return jsonify({"error": "No token provided"}), 400
            
        # Verify the ID token
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        
        # Set session
        session["user_id"] = uid
        
        name = decoded_token.get("name") or decoded_token.get("email", "User")
        picture = decoded_token.get("picture")
        
        session["user_name"] = name
        session["user_photo"] = picture
        
        # Save or Update User in Firestore
        if db is not None:
            try:
                user_ref = db.collection('users').document(uid)
                user_ref.set({
                    'uid': uid,
                    'email': decoded_token.get("email"),
                    'displayName': name,
                    'photoURL': picture,
                    'lastLogin': firestore.SERVER_TIMESTAMP
                }, merge=True)
            except Exception as e:
                print(f"Error saving user to Firestore: {e}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Firebase token verification failed: {e}")
        return jsonify({"error": "Invalid token"}), 401

@app.route('/api/uploadAvatar', methods=['POST'])
def upload_avatar():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        # Create a unique filename based on the user ID and a unique string to prevent caching issues
        filename = secure_filename(f"avatar_{session['user_id']}_{uuid.uuid4().hex[:8]}.{ext}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Construct the URL that the frontend can use to reach to this image
        # Since it's in the static folder, it's accessible via /static/...
        image_url = url_for('static', filename=f"uploads/avatars/{filename}")
        
        return jsonify({"status": "success", "url": image_url})
        
    return jsonify({"error": "File type not allowed"}), 400

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    session.pop("user_photo", None)
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
        
        # Save analysis to Firebase Firestore if connected
        if db is not None:
            try:
                # Add timestamp and user ID if logged in
                user_id = session.get("user_id", "anonymous")
                doc_ref = db.collection('scans').document()
                doc_ref.set({
                    'asin': asin,
                    'user_id': user_id,
                    'summary': summary,
                    'chart_data': chart_data,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                print(f"Saved analysis for {asin} to Firestore.")
            except Exception as e:
                print(f"Error saving to Firestore: {e}")

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
