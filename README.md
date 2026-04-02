# 🛡️ TrustLens AI — Forensic Review Intelligence

> **Evidence-driven fake review detection powered by ensemble ML, NLP, and reviewer network analysis.**

TrustLens AI is a full-stack web application that exposes fake reviews on e-commerce platforms by applying three forensic layers: language pattern analysis, reviewer behavior profiling, and coordination network graph detection. It outputs a clear **Trust Score**, an **Adjusted Authentic Rating**, and per-review fraud evidence — all in under 30 seconds.

**🌐 Live Demo:** [https://trustlens-ai-01sk.onrender.com](https://trustlens-ai-01sk.onrender.com)

---

## 📸 Preview

| Home Dashboard | Live Analysis | Leaderboard |
|---|---|---|
| Real-time trust metrics & product scan history | Full forensic output: score, ring graph, suspicious reviews | Category-level fake review rankings from training data |

---

## 🏗️ Architecture

```
TrustLens AI
├── app.py                          ← Flask backend (routes, auth, scoring, history)
├── analyzer.py                     ← Core ML inference pipeline
├── scrape_reviews_dataset.py       ← Amazon/Flipkart review scraper
├── Review_ring_detector.py         ← Reviewer coordination ring detection
├── Reviewer_timeline_detector.py   ← Burst activity & timeline anomaly detection
├── Text_rating_mismatch.py         ← Sentiment vs. star-rating mismatch detection
├── price_fake_correlation.py       ← Price-point fraud correlation analysis
├── product_intelligence.py         ← Discrepancy scoring & bot network detection
│
├── model/
│   ├── random_forest.pkl           ← Random Forest classifier
│   ├── logistic_regression.pkl     ← Logistic Regression classifier
│   ├── xgboost.pkl                 ← XGBoost classifier
│   ├── meta_learner.pkl            ← Stacking meta-learner (ensemble)
│   ├── tfidf.pkl                   ← TF-IDF vectorizer
│   └── roberta/                    ← Fine-tuned RoBERTa model
│
├── templates/                      ← Jinja2 HTML templates (Tailwind CSS)
│   ├── home.html
│   ├── analyze.html
│   ├── login.html / register.html
│   ├── profile.html
│   ├── leaderboard.html
│   └── scan.html
│
├── static/                         ← JS, CSS, assets
├── trust_history.csv               ← Local scan score history
├── scan_history_local.jsonl        ← Full scan result log
└── fake reviews dataset.csv        ← Training dataset (~15 MB)
```

---

## 🔬 Detection Methodology

### Layer 1 — Language Pattern Scan
- TF-IDF feature extraction on review text
- Detects repetitive phrasing, excessive hype language, and unnatural sentiment curves
- RoBERTa transformer model for deep semantic fake probability

### Layer 2 — Reviewer Behavior Signals
- Timeline burst detection (high-velocity review clusters)
- Narrow rating distribution flags (e.g., only 5-star reviewers)
- Cross-product reviewer overlap anomalies

### Layer 3 — Coordination Network Graph
- Builds reviewer–product relationship graphs using `networkx`
- Louvain community detection to surface hidden reviewer rings
- Outputs ring count, cluster size, and coordination score

### Ensemble Scoring
The final **Fraud Score** is a stacked ensemble output from:

| Model | Role |
|---|---|
| Random Forest | Structural/behavioral features |
| Logistic Regression | Linear text signal baseline |
| XGBoost | Gradient-boosted combined features |
| RoBERTa | Deep NLP fake probability |
| **Meta-Learner** | **Final stacked verdict** |

**Trust Score = 100 − Fraud Score**
**Adjusted Rating** discounts up to 2 stars based on fraud level.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask 3.0 |
| ML / NLP | scikit-learn, XGBoost, HuggingFace Transformers (RoBERTa), NLTK |
| Data | pandas, numpy, scipy |
| Graph Analysis | networkx, python-louvain |
| Auth & DB | Firebase Admin SDK (Firestore + Auth) |
| Scraping | BeautifulSoup4, Requests, SerpAPI |
| AI Insights | Google Generative AI (`google-generativeai`) |
| Frontend | Jinja2 templates, Tailwind CSS, Vanilla JS |
| Visualization | Plotly, custom D3-style forensic graph |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A Firebase project (Firestore + Authentication enabled)
- SerpAPI key (for live Amazon scraping)
- Google Generative AI API key (for AI-powered insights)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fake-reviews.git
cd fake-reviews
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv311
# Windows
.venv311\Scripts\activate
# macOS/Linux
source .venv311/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the root directory:
```env
FIREBASE_WEB_API_KEY=your_firebase_web_api_key
SERPAPI_KEY=your_serpapi_key
GOOGLE_API_KEY=your_google_generative_ai_key
MODEL_ACCURACY=92.0
```

Also place your Firebase service account credentials JSON at the root or configure `GOOGLE_APPLICATION_CREDENTIALS`.

### 5. Run the application
```bash
flask run
# or
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📄 Key Pages

| Route | Description |
|---|---|
| `/` | Home — live trust metrics, product overview |
| `/analyze` | Run a forensic scan by ASIN or Amazon URL |
| `/history` | Full scan history with per-scan details |
| `/leaderboard` | Category-level trust rankings from training data |
| `/profile` | User profile & avatar management |
| `/login` / `/register` | Firebase-backed authentication |

---

## 🧪 Training the Models

The `fake_review_detection.ipynb` Jupyter notebook contains the full training pipeline:
1. Load and preprocess `fake reviews dataset.csv`
2. Extract TF-IDF + behavioral features
3. Train Random Forest, Logistic Regression, XGBoost
4. Fine-tune RoBERTa on review text
5. Train meta-learner on base model outputs
6. Export all models to `model/`

To retrain:
```bash
jupyter notebook fake_review_detection.ipynb
```

---

## 📊 Dataset

- **Primary**: `fake reviews dataset.csv` — ~15 MB, labeled `OR` (Original) / `CG` (Computer Generated)
- **Supplementary**: `scraped_training_reviews.csv` — live-scraped Amazon reviews used for incremental training

---

## 🗂️ Local Data Files

| File | Purpose |
|---|---|
| `trust_history.csv` | Appended per-scan trust & fraud scores |
| `scan_history_local.jsonl` | Full JSON scan results (ASIN, summary, chart data) |
| `analysis_results.csv` | Detailed per-review analysis output |

---

## 🔒 Security Notes

- Firebase Authentication handles user login/registration
- Firestore security rules defined in `firestore.rules`
- `.env` and service account JSON are excluded via `.gitignore`
- Session secret key should be rotated for production deployment

---

## 📁 Project Structure (Expanded)

```
extension/          ← Browser extension (future: in-page trust badge)
dashboard/          ← Standalone dashboard experiments
new wave/           ← Experimental detection modules
dataconnect/        ← Firebase Data Connect configuration
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add: your feature"`
4. Push and open a Pull Request

---

## 📜 License

This project is intended for academic and research purposes.

---

## 👤 Author

**Aditya Gupta**
- GitHub: [@ADITYAGUPTA112](https://github.com/ADITYAGUPTA112)

---

> *"See the real product trust score before you buy."*
