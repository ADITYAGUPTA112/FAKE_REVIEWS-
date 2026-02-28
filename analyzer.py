import torch
import numpy as np
import pandas as pd
import os
from serpapi import GoogleSearch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder
import re
import joblib
le = joblib.load("./models/label_encoder.pkl")

# =========================
# CONFIG
# =========================
SERP_API_KEY = os.getenv("SERP_API_KEY", "894bf919c29bd261838dd97a18cced3971f17f0077e190f7f4e4f33bbe47468c")
MODEL_PATH = "./models/distilbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert")
tokenizer = DistilBertTokenizerFast.from_pretrained("./models/distilbert")
model.to(device)
model.eval()

# Load label encoder
df_temp = pd.read_csv("fake reviews dataset.csv")
le = LabelEncoder()
le.fit(df_temp["label"])

# =========================
# FETCH MULTIPLE PAGES
# =========================
def fetch_reviews_multi_page(asin, target_domain="amazon.com", pages=3, max_reviews=55):
    all_reviews = []
    
    # We will try the target domain first, then fallback to other English domains
    # to aggregate enough reviews to reach our goal (50+)
    domains = [target_domain, "amazon.co.uk", "amazon.ca", "amazon.in", "amazon.com.au"]
    
    for domain in domains:
        if len(all_reviews) >= max_reviews:
            break
            
        params = {
            "engine": "amazon_product",
            "amazon_domain": domain,
            "asin": asin,
            "api_key": SERP_API_KEY
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except:
            continue

        reviews_info = results.get("reviews_information", {})
        
        # 1. Extract authors_reviews
        for review in reviews_info.get("authors_reviews", []):
            if "text" in review and review["text"] not in all_reviews:
                all_reviews.append(review["text"])

        # 2. Extract other_countries_reviews
        for review in reviews_info.get("other_countries_reviews", []):
            if "text" in review and review["text"] not in all_reviews:
                all_reviews.append(review["text"])
                
        # 3. Extract top_reviews
        for review in reviews_info.get("top_reviews", []):
            if "text" in review and review["text"] not in all_reviews:
                all_reviews.append(review["text"])

        # 4. Extract insights examples (lots of snippets here)
        summary_info = reviews_info.get("summary", {})
        for insight in summary_info.get("insights", []):
            for example in insight.get("examples", []):
                snippet = example.get("snippet", "")
                if snippet and snippet not in all_reviews:
                    all_reviews.append(snippet)
        
        # 5. Extract fallback native reviews format
        for review in results.get("reviews", []):
            text = review.get("content", review.get("body", ""))
            if text and text not in all_reviews:
                all_reviews.append(text)
                
    print("REVIEWS COUNT:", len(all_reviews))

    # Cap at exactly the target limit or a little over to keep processing fast
    return all_reviews[:100]
    
# =========================
# PREDICTION
# =========================
def predict_review(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_class = np.argmax(probs)

    original_label = le.inverse_transform([pred_class])[0]
    readable = "Fake" if original_label == "CG" else "Genuine"

    confidence = probs[pred_class] * 100

    return readable, confidence

# =========================
# ANALYZE PRODUCT
# =========================
def analyze_product(asin, pages=3):

    domain = extract_domain(asin)
    asin_id = extract_asin(asin)
    reviews = fetch_reviews_multi_page(asin_id, domain, pages)

    results_data = []
    fake_count = 0
    total_confidence = 0

    for review in reviews:

        label, confidence = predict_review(review)

        if label == "Fake":
            fake_count += 1

        total_confidence += confidence

        results_data.append({
            "review": review,
            "prediction": label,
            "confidence": round(confidence, 2)
        })

    total_reviews = len(reviews)
    fake_percent = (fake_count / total_reviews) * 100 if total_reviews else 0
    avg_confidence = total_confidence / total_reviews if total_reviews else 0

    df_results = pd.DataFrame(results_data)
    print("Total fetched reviews:", len(reviews))

    # Save CSV
    df_results.to_csv("analysis_results.csv", index=False)

    summary = {
        "total_reviews": int(total_reviews),
        "fake_percent": float(round(fake_percent, 2)),
        "genuine_percent": float(round(100 - fake_percent, 2)),
        "avg_confidence": float(round(avg_confidence, 2))
    }

    return summary, df_results

def extract_domain(url_or_asin):
    url_or_asin = url_or_asin.strip()
    match = re.search(r"amazon\.([a-z\.]+)/", url_or_asin)
    if match:
        return f"amazon.{match.group(1)}"
    return "amazon.com"

def extract_asin(url_or_asin):
    """
    Extract ASIN from full Amazon URL or return ASIN if already provided.
    Supports .com, .in, shortened URLs, and all common formats.
    """

    url_or_asin = url_or_asin.strip()

    # If already ASIN
    if re.fullmatch(r"[A-Z0-9]{10}", url_or_asin):
        return url_or_asin

    # Universal ASIN pattern (anywhere in string)
    match = re.search(r"([A-Z0-9]{10})", url_or_asin)

    if match:
        return match.group(1)

    raise ValueError("Invalid Amazon URL or ASIN.")