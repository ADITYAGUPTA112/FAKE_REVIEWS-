import torch
import numpy as np
import pandas as pd
import os
from serpapi import GoogleSearch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder
import re
import joblib
import dateparser
from lime.lime_text import LimeTextExplainer

le = joblib.load("./models/label_encoder.pkl")

# =========================
# CONFIG
# =========================
SERP_API_KEY = os.getenv("SERP_API_KEY", "add your api key")
MODEL_PATH = "./models/distilbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert")
tokenizer = DistilBertTokenizerFast.from_pretrained("./models/distilbert")
model.to(device)
model.eval()

explainer = LimeTextExplainer(class_names=['Fake', 'Genuine'])

# Load label encoder
df_temp = pd.read_csv("fake reviews dataset.csv")
le = LabelEncoder()
le.fit(df_temp["label"])

def get_prediction_probs(texts):
    batch_size = 32
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)

# =========================
# FETCH MULTIPLE PAGES
# =========================
def fetch_reviews_multi_page(asin, target_domain="amazon.com", pages=3, max_reviews=55):
    all_reviews = []
    seen_texts = set()
    
    domains = [target_domain, "amazon.co.uk", "amazon.ca", "amazon.in", "amazon.com.au"]
    
    def add_review(text, date_str):
        if text and text not in seen_texts:
            parsed_date = None
            if date_str:
                d = dateparser.parse(date_str)
                if d:
                    parsed_date = d.isoformat()
            
            all_reviews.append({
                "text": text,
                "date": parsed_date or ""
            })
            seen_texts.add(text)

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
            add_review(review.get("text"), review.get("date"))

        # 2. Extract other_countries_reviews
        for review in reviews_info.get("other_countries_reviews", []):
            add_review(review.get("text"), review.get("date"))
                
        # 3. Extract top_reviews
        for review in reviews_info.get("top_reviews", []):
            add_review(review.get("text"), review.get("date"))

        # 4. Extract insights examples
        summary_info = reviews_info.get("summary", {})
        for insight in summary_info.get("insights", []):
            for example in insight.get("examples", []):
                snippet = example.get("snippet", "")
                add_review(snippet, "")
        
        # 5. Extract fallback native reviews format
        for review in results.get("reviews", []):
            text = review.get("content", review.get("body", ""))
            add_review(text, review.get("date"))
                
    print("REVIEWS COUNT:", len(all_reviews))
    return all_reviews[:100]
    
# =========================
# PREDICTION
# =========================
def predict_review(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
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

def predict_reviews_batch(texts):
    batch_size = 32
    all_labels = []
    all_confidences = []
    
    if not texts:
        return [], []
        
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        pred_classes = np.argmax(probs, axis=1)
        
        for j, pred_class in enumerate(pred_classes):
            original_label = le.inverse_transform([pred_class])[0]
            readable = "Fake" if original_label == "CG" else "Genuine"
            confidence = probs[j][pred_class] * 100
            
            all_labels.append(readable)
            all_confidences.append(confidence)
            
    return all_labels, all_confidences

def explain_review(text, label):
    try:
        # Reduced num_samples to 100 to significantly speed up explanation time
        exp = explainer.explain_instance(text, get_prediction_probs, num_features=5, num_samples=100)
        # Class 0 usually CG/Fake, 1 usually OR/Genuine
        # Find which index the label actually mapped to in the model
        fake_idx = list(le.classes_).index("CG") if "CG" in le.classes_ else 0
        gen_idx = list(le.classes_).index("OR") if "OR" in le.classes_ else 1
        
        target_idx = fake_idx if label == "Fake" else gen_idx
        
        explanation_list = exp.as_list(label=target_idx)
        return [{"word": w, "weight": float(wt)} for w, wt in explanation_list]
    except Exception as e:
        print("LIME explanation error:", e)
        return []

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

    if not reviews:
        return {"total_reviews": 0, "fake_percent": 0.0, "genuine_percent": 0.0, "avg_confidence": 0.0}, pd.DataFrame()

    texts = [rev["text"] for rev in reviews]
    labels, confidences = predict_reviews_batch(texts)

    for i, rev_dict in enumerate(reviews):
        text = rev_dict["text"]
        label = labels[i]
        confidence = confidences[i]

        if label == "Fake":
            fake_count += 1

        total_confidence += confidence
        
        # Apply LIME only to high confidence fake reviews or sample to keep it fast
        # Let's run LIME on a few of them
        explanation = []
        if len(results_data) < 3:  # Only explain first 3 for performance to avoid bottleneck
            explanation = explain_review(text, label)

        results_data.append({
            "review": text,
            "date": rev_dict["date"],
            "prediction": label,
            "confidence": round(confidence, 2),
            "explanation": explanation
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
        "genuine_percent": float(round(100 - fake_percent, 2)) if total_reviews else 0.0,
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
