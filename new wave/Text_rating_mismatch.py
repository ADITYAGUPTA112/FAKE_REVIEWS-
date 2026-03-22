"""
========================================================
  MODULE 2 — Text–Rating Mismatch Detector
========================================================
Detects reviews where the text's emotional tone contradicts
the numerical star rating — a hallmark of low-quality fake reviews
written by outsourced workers who paste a positive text onto
any star rating or vice-versa.

Pipeline:
    1. Compute VADER compound sentiment score  [-1.0 → +1.0]
    2. Normalise star rating (1–5) to the same scale
    3. Calculate absolute discrepancy
    4. Classify into mismatch categories
    5. Return enriched DataFrame with flags

Expected DataFrame columns:
    review_id : str / int
    text      : str
    rating    : int  (1–5)
========================================================
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Literal

# ── Thresholds ────────────────────────────────────────────────────────────────
MISMATCH_THRESHOLD = 0.50   # discrepancy ≥ this → flagged
HIGH_MISMATCH      = 0.80   # discrepancy ≥ this → high severity
NEUTRAL_BAND       = 0.20   # |sentiment| < this → text is "neutral"
# ─────────────────────────────────────────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()   # instantiate once, reuse


def get_vader_sentiment(text: str) -> dict:
    """
    Returns VADER scores for a single text.
    Keys: neg, neu, pos, compound  (compound is the primary signal: -1 → +1)
    """
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return _vader.polarity_scores(text)


def normalize_rating(
    rating: float,
    min_stars: int = 1,
    max_stars: int = 5,
) -> float:
    """
    Maps a star rating linearly onto [-1.0, +1.0].

        1 ★  →  -1.00
        2 ★  →  -0.50
        3 ★   →   0.00
        4 ★  →  +0.50
        5 ★  →  +1.00
    """
    return ((rating - min_stars) / (max_stars - min_stars)) * 2.0 - 1.0


def _classify_mismatch(
    normalized_rating: float,
    sentiment_score: float,
    discrepancy: float,
) -> str:
    """
    Returns a human-readable mismatch category.
    """
    if discrepancy < MISMATCH_THRESHOLD:
        return "NORMAL"

    sev = "HIGH" if discrepancy >= HIGH_MISMATCH else "MEDIUM"

    if normalized_rating >= 0.5 and sentiment_score <= -NEUTRAL_BAND:
        return f"{sev}_POSITIVE_RATING_NEGATIVE_TEXT"
    if normalized_rating <= -0.5 and sentiment_score >= NEUTRAL_BAND:
        return f"{sev}_NEGATIVE_RATING_POSITIVE_TEXT"
    if normalized_rating >= 0.5 and abs(sentiment_score) < NEUTRAL_BAND:
        return f"{sev}_HIGH_RATING_NEUTRAL_TEXT"
    if normalized_rating <= -0.5 and abs(sentiment_score) < NEUTRAL_BAND:
        return f"{sev}_LOW_RATING_NEUTRAL_TEXT"

    return f"{sev}_GENERIC_MISMATCH"


def detect_text_rating_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyses a DataFrame of reviews for sentiment–rating mismatches.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: review_id, text, rating

    Returns
    -------
    pd.DataFrame
        Original columns + enriched analysis columns:
            sentiment_neg, sentiment_neu, sentiment_pos,
            sentiment_compound, normalized_rating,
            discrepancy, is_mismatch, mismatch_type
    """
    required = {"review_id", "text", "rating"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["text"]   = df["text"].fillna("").astype(str)

    # ── Step 1: VADER sentiment ───────────────────────────────────────────────
    vader_scores = df["text"].apply(get_vader_sentiment)
    df["sentiment_neg"]      = vader_scores.apply(lambda s: s["neg"])
    df["sentiment_neu"]      = vader_scores.apply(lambda s: s["neu"])
    df["sentiment_pos"]      = vader_scores.apply(lambda s: s["pos"])
    df["sentiment_compound"] = vader_scores.apply(lambda s: s["compound"])

    # ── Step 2: Normalise ratings ─────────────────────────────────────────────
    df["normalized_rating"] = df["rating"].apply(normalize_rating)

    # ── Step 3: Discrepancy ───────────────────────────────────────────────────
    df["discrepancy"] = (df["normalized_rating"] - df["sentiment_compound"]).abs()

    # ── Step 4: Flag & classify ───────────────────────────────────────────────
    df["is_mismatch"]   = df["discrepancy"] >= MISMATCH_THRESHOLD
    df["mismatch_type"] = df.apply(
        lambda r: _classify_mismatch(
            r["normalized_rating"], r["sentiment_compound"], r["discrepancy"]
        ),
        axis=1,
    )

    # Round floats for readability
    float_cols = [
        "sentiment_neg", "sentiment_neu", "sentiment_pos",
        "sentiment_compound", "normalized_rating", "discrepancy",
    ]
    df[float_cols] = df[float_cols].round(4)

    return df


def mismatch_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of mismatch categories and counts.
    """
    total = len(result_df)
    flagged = result_df[result_df["is_mismatch"]]
    summary = (
        flagged["mismatch_type"]
        .value_counts()
        .rename_axis("mismatch_type")
        .reset_index(name="count")
    )
    summary["pct_of_total"] = (summary["count"] / total * 100).round(1)
    return summary


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_reviews = [
        # Clearly genuine
        {
            "review_id": "R001",
            "text": "Absolutely love this product! Works perfectly and arrived fast.",
            "rating": 5,
        },
        # Obvious fake: positive text, 1-star rating
        {
            "review_id": "R002",
            "text": "Great quality, really impressed with the build. Highly recommend!",
            "rating": 1,
        },
        # Classic paid fake: positive text, 5-stars but text is weak/generic
        {
            "review_id": "R003",
            "text": "ok",
            "rating": 5,
        },
        # Negative text, 5-star rating
        {
            "review_id": "R004",
            "text": "Absolute garbage. Broke after two days. Terrible waste of money.",
            "rating": 5,
        },
        # Genuine critical review
        {
            "review_id": "R005",
            "text": "Disappointing product. Doesn't match the description at all.",
            "rating": 2,
        },
        # Neutral text, very high rating — suspicious
        {
            "review_id": "R006",
            "text": "This is a product. It arrived in a box.",
            "rating": 5,
        },
    ]

    demo_df = pd.DataFrame(sample_reviews)
    print("=== Demo: Text–Rating Mismatch Detector ===\n")
    result = detect_text_rating_mismatch(demo_df)

    display_cols = [
        "review_id", "rating", "sentiment_compound",
        "normalized_rating", "discrepancy", "is_mismatch", "mismatch_type",
    ]
    print(result[display_cols].to_string(index=False))
    print()
    print("── Mismatch Summary ──")
    print(mismatch_summary(result).to_string(index=False))
    print()

    # Show worst offenders
    worst = result[result["is_mismatch"]].sort_values("discrepancy", ascending=False)
    print("── Flagged Reviews ──")
    for _, row in worst.iterrows():
        print(f"  [{row['review_id']}]  ★{row['rating']}  |  "
              f"VADER={row['sentiment_compound']:+.2f}  |  "
              f"Gap={row['discrepancy']:.2f}  |  {row['mismatch_type']}")
        print(f"       \"{row['text'][:80]}\"")
        print()