"""
download_models.py
------------------
Run on Render startup to download large model files from Google Drive.
Build command in render.yaml:
    pip install -r requirements.txt && python download_models.py
"""

import os
import gdown

# ─── Google Drive File IDs (from folder: 1sYiyqQI4YarrQkciFY0uehE_eyKXLaoQ) ──
MODEL_FILES = {
    "model/random_forest.pkl":       "1Ew8F3r4kSBR3Gc2vOMfSkwpfCjKiLtvD",
    "model/logistic_regression.pkl": "1dfzisyRIEazHNv-SaxpoQmZTtQGqjOb_",
    "model/xgboost.pkl":             "1TM1GVRhMYjKrr7sTBviydaqcvGjO5gPs",
    "model/meta_learner.pkl":        "1nnFA_lSvJn9HOJWCcz1v4_jTkQqqyB7G",
    "model/tfidf.pkl":               "1hy-xzXxpYj3M39HomRmqwvUIN0Dnj9VG",
}

def download_if_missing():
    os.makedirs("model", exist_ok=True)

    for local_path, file_id in MODEL_FILES.items():
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"[OK]   Already exists: {local_path} ({size_mb:.1f} MB)")
            continue

        print(f"[DL]   Downloading {local_path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, local_path, quiet=False, fuzzy=True)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"[DONE] {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[ERR]  Failed to download {local_path}: {e}")

if __name__ == "__main__":
    download_if_missing()
    print("\nAll model files ready.")
