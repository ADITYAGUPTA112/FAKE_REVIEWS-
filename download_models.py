"""
download_models.py
------------------
Run this ONCE on Render startup (via render.yaml build command) to download
large model files from Google Drive that can't be stored in GitHub (>100MB).

HOW TO USE:
1. Upload each model file to Google Drive
2. Right-click → "Get link" → set to "Anyone with the link"
3. Copy the file ID from the URL:
   https://drive.google.com/file/d/FILE_ID_HERE/view
4. Paste the FILE_ID below for each model

Install gdown: pip install gdown  (already in requirements)
"""

import os
import gdown

# ─── Configure your Google Drive file IDs here ────────────────────────────────
# Format: { "local/path/to/file.pkl": "GOOGLE_DRIVE_FILE_ID" }

MODEL_FILES = {
    "model/random_forest.pkl":      "PASTE_RANDOM_FOREST_FILE_ID_HERE",
    "model/logistic_regression.pkl": "PASTE_LR_FILE_ID_HERE",
    "model/xgboost.pkl":            "PASTE_XGBOOST_FILE_ID_HERE",
    "model/meta_learner.pkl":       "PASTE_META_LEARNER_FILE_ID_HERE",
    "model/tfidf.pkl":              "PASTE_TFIDF_FILE_ID_HERE",
}

# For RoBERTa directory (if used), add individual files similarly.
# "model/roberta/config.json":    "PASTE_FILE_ID_HERE",
# "model/roberta/model.safetensors": "PASTE_FILE_ID_HERE",

# ──────────────────────────────────────────────────────────────────────────────

def download_if_missing():
    os.makedirs("model", exist_ok=True)

    for local_path, file_id in MODEL_FILES.items():
        if file_id.startswith("PASTE_"):
            print(f"[SKIP] No file ID configured for {local_path}")
            continue

        if os.path.exists(local_path):
            print(f"[OK]   Already exists: {local_path}")
            continue

        print(f"[DL]   Downloading {local_path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, local_path, quiet=False)
            print(f"[DONE] {local_path}")
        except Exception as e:
            print(f"[ERR]  Failed to download {local_path}: {e}")


if __name__ == "__main__":
    download_if_missing()
    print("\nAll model files ready.")
