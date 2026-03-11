import torch
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

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
SERP_API_KEY = os.getenv("SERP_API_KEY", "894bf919c29bd261838dd97a18cced3971f17f0077e190f7f4e4f33bbe47468c")
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
    product_details = {}
    
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

        if not product_details and "product_results" in results:
            prod = results.get("product_results", {})
            product_details = {
                "title": prod.get("title", f"Amazon Product ({asin})"),
                "thumbnail": prod.get("thumbnail", ""),
                "rating": prod.get("rating", 0),
                "reviews_total": prod.get("reviews", 0)
            }

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
    return all_reviews[:100], product_details
    
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
    reviews, product_details = fetch_reviews_multi_page(asin_id, domain, pages)

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
        "avg_confidence": float(round(avg_confidence, 2)),
        "product_title": product_details.get("title", f"Amazon Product ({asin_id})"),
        "product_image": product_details.get("thumbnail", ""),
        "product_rating": product_details.get("rating", 0),
        "product_reviews_total": product_details.get("reviews_total", 0)
    }

    return summary, df_results

def get_review_flags(text):
    """
    Analyse a single review text for common heuristic red-flags associated
    with fake / paid reviews.  Returns a list of flag dicts:
        {"label": str, "description": str, "severity": "low"|"medium"|"high"}
    """
    flags = []
    words = text.split()
    word_count = len(words)
    lower = text.lower()

    # 1. Suspiciously short review
    if word_count < 8:
        flags.append({
            "label": "Very Short Review",
            "description": f"Only {word_count} word(s). Genuine reviews typically provide detail.",
            "severity": "medium"
        })

    # 2. Excessive capitalisation
    upper_words = [w for w in words if w.isupper() and len(w) > 2]
    if len(upper_words) >= 3:
        flags.append({
            "label": "Excessive Caps",
            "description": "Unusually high number of ALL-CAPS words — a common attention-seeking tactic.",
            "severity": "low"
        })

    # 3. Excessive exclamation / question marks
    exclaim_count = text.count('!') + text.count('?')
    if exclaim_count >= 4:
        flags.append({
            "label": "Over-Punctuated",
            "description": f"{exclaim_count} exclamation/question marks detected. Overly enthusiastic tone is a fake-review signal.",
            "severity": "low"
        })

    # 4. Promotional / incentivised language
    promo_keywords = [
        "received free", "received this for free", "free product", "discount code",
        "in exchange for", "promotional", "gifted", "sponsored", "i was given",
        "sent to me", "provided for review", "complimentary", "received at a discount",
        "i got this for free"
    ]
    for kw in promo_keywords:
        if kw in lower:
            flags.append({
                "label": "Incentivised Language",
                "description": f"Phrase '{kw}' suggests a paid or incentivised review.",
                "severity": "high"
            })
            break

    # 5. Generic / template phrases
    generic_phrases = [
        "highly recommend", "five stars", "best product ever", "love this product",
        "great product", "amazing product", "works as described", "exactly as described",
        "fast shipping", "worth every penny", "would definitely buy again", "two thumbs up"
    ]
    matched_generic = [p for p in generic_phrases if p in lower]
    if len(matched_generic) >= 2:
        flags.append({
            "label": "Generic Phrases",
            "description": f"Contains {len(matched_generic)} template-like phrases: {', '.join(matched_generic[:3])}.",
            "severity": "medium"
        })

    # 6. Repetitive words
    if word_count > 5:
        from collections import Counter
        content_words = [w.lower().strip('.,!?;:"\'') for w in words if len(w) > 3]
        freq = Counter(content_words)
        repeated = [(w, c) for w, c in freq.items() if c >= 4]
        if repeated:
            top = sorted(repeated, key=lambda x: -x[1])[0]
            flags.append({
                "label": "Word Repetition",
                "description": f"Word '{top[0]}' appears {top[1]} times — possible bot-generated text.",
                "severity": "medium"
            })

    # 7. No specific product mention (no numbers, model names, measurements, etc.)
    has_specifics = bool(re.search(r'\b\d+\b', text))
    if not has_specifics and word_count >= 20:
        flags.append({
            "label": "Lacks Specific Details",
            "description": "No numbers or measurements found. Genuine reviews usually reference specific product details.",
            "severity": "low"
        })

    return flags


def check_single_review(text):
    """
    Analyse a single review text and return prediction, confidence, LIME
    explanation and heuristic flags.
    """
    label, confidence = predict_review(text)
    explanation = explain_review(text, label)
    flags = get_review_flags(text)

    # Compute a composite risk score (0–100) that blends model confidence and flag count
    flag_severity_weights = {"low": 5, "medium": 10, "high": 20}
    flag_score = sum(flag_severity_weights.get(f["severity"], 5) for f in flags)
    if label == "Fake":
        risk_score = min(100, round((confidence * 0.75) + min(flag_score, 25)))
    else:
        risk_score = min(100, round(((100 - confidence) * 0.5) + min(flag_score, 25)))

    return {
        "prediction": label,
        "confidence": round(float(confidence), 2),
        "explanation": explanation,
        "flags": flags,
        "risk_score": int(risk_score)
    }


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
