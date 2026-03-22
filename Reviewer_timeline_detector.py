"""
========================================================
  MODULE 1 — Reviewer Behavior Timeline Detector
========================================================
Flags suspicious users based on review-posting patterns.

Expected DataFrame columns:
    user_id    : str   — unique user identifier
    product_id : str   — product being reviewed
    timestamp  : str / datetime — when the review was posted
    rating     : int   — star rating (1–5)

Anomaly flags detected:
    A) HIGH_VELOCITY     — >5 reviews posted within any rolling 1-hour window
    B) DORMANT_BURST     — account inactive 90+ days, then 3+ reviews in one day
    C) RATING_EXTREMISM  — user gives ONLY 1-star or ONLY 5-star ratings (≥5 reviews)
========================================================
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional


# ── Constants (tune these for your use case) ─────────────────────────────────
VELOCITY_WINDOW_HOURS   = 1     # rolling window size
VELOCITY_THRESHOLD      = 5     # reviews within that window = suspicious
DORMANCY_DAYS           = 90    # inactivity gap that defines "dormant"
BURST_THRESHOLD         = 3     # reviews in one day after dormancy = burst
EXTREMISM_MIN_REVIEWS   = 5     # minimum reviews before flagging extremism
# ─────────────────────────────────────────────────────────────────────────────


def _flag_high_velocity(group: pd.DataFrame) -> Optional[dict]:
    """
    Detect >VELOCITY_THRESHOLD reviews within any VELOCITY_WINDOW_HOURS window.
    Uses a sliding-window approach over sorted timestamps.
    """
    if len(group) < VELOCITY_THRESHOLD:
        return None

    timestamps = group["timestamp"].sort_values().reset_index(drop=True)
    window = timedelta(hours=VELOCITY_WINDOW_HOURS)

    for i, ts_start in enumerate(timestamps):
        ts_end = ts_start + window
        count_in_window = ((timestamps >= ts_start) & (timestamps <= ts_end)).sum()
        if count_in_window > VELOCITY_THRESHOLD:
            return {
                "flag_type": "HIGH_VELOCITY",
                "detail": (
                    f"{count_in_window} reviews posted within "
                    f"{VELOCITY_WINDOW_HOURS}h starting {ts_start.strftime('%Y-%m-%d %H:%M')}"
                ),
            }
    return None


def _flag_dormant_burst(group: pd.DataFrame) -> Optional[dict]:
    """
    Detect accounts that are inactive for DORMANCY_DAYS, then post
    BURST_THRESHOLD+ reviews on a single day.
    """
    if len(group) < 2:
        return None

    group = group.sort_values("timestamp").reset_index(drop=True)
    group["_gap"] = group["timestamp"].diff()

    dormant_rows = group[group["_gap"] > timedelta(days=DORMANCY_DAYS)]

    for idx in dormant_rows.index:
        burst_date = group.loc[idx, "timestamp"].date()
        reviews_that_day = (group["timestamp"].dt.date == burst_date).sum()
        if reviews_that_day >= BURST_THRESHOLD:
            gap_days = group.loc[idx, "_gap"].days
            return {
                "flag_type": "DORMANT_BURST",
                "detail": (
                    f"Inactive {gap_days} days, then "
                    f"{reviews_that_day} reviews on {burst_date}"
                ),
            }
    return None


def _flag_rating_extremism(group: pd.DataFrame) -> Optional[dict]:
    """
    Detect users who exclusively give 1-star or 5-star ratings
    (no nuanced feedback — classic bot trait).
    """
    if len(group) < EXTREMISM_MIN_REVIEWS:
        return None

    unique_ratings = set(group["rating"].unique())

    # Must have at least one extreme AND no middle ratings
    has_extremes = bool(unique_ratings & {1, 5})
    has_middle   = bool(unique_ratings & {2, 3, 4})

    if has_extremes and not has_middle:
        counts = group["rating"].value_counts().to_dict()
        return {
            "flag_type": "RATING_EXTREMISM",
            "detail": (
                f"Only extreme ratings given across {len(group)} reviews — "
                f"distribution: {counts}"
            ),
        }
    return None


def detect_reviewer_anomalies(
    df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Main entry point. Analyses all users and returns a summary DataFrame
    of flagged accounts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: user_id, product_id, timestamp, rating
    verbose : bool
        Print progress while processing

    Returns
    -------
    pd.DataFrame with columns:
        user_id, total_reviews, flag_count, risk_score, flags (list of dicts)
    """
    # ── Input validation ─────────────────────────────────────────────────────
    required_cols = {"user_id", "product_id", "timestamp", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["rating"]    = pd.to_numeric(df["rating"], errors="coerce")

    # ── Per-user analysis ────────────────────────────────────────────────────
    results = []
    groups  = df.groupby("user_id")
    total   = len(groups)

    for i, (user_id, group) in enumerate(groups):
        if verbose and i % 500 == 0:
            print(f"  Processing user {i}/{total}...")

        active_flags = []

        for detector in (_flag_high_velocity, _flag_dormant_burst, _flag_rating_extremism):
            flag = detector(group)
            if flag:
                active_flags.append(flag)

        if active_flags:
            # Risk score: each flag adds weight; capped at 100
            weights = {"HIGH_VELOCITY": 50, "DORMANT_BURST": 35, "RATING_EXTREMISM": 30}
            score   = min(sum(weights.get(f["flag_type"], 20) for f in active_flags), 100)

            results.append(
                {
                    "user_id":       user_id,
                    "total_reviews": len(group),
                    "flag_count":    len(active_flags),
                    "risk_score":    score,
                    "flags":         active_flags,
                }
            )

    if not results:
        print("No anomalies detected.")
        return pd.DataFrame(
            columns=["user_id", "total_reviews", "flag_count", "risk_score", "flags"]
        )

    result_df = pd.DataFrame(results).sort_values("risk_score", ascending=False)
    result_df = result_df.reset_index(drop=True)
    return result_df


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime, timedelta
    import random

    random.seed(42)

    def make_user_reviews(user_id, timestamps, ratings, products=None):
        rows = []
        for i, (ts, r) in enumerate(zip(timestamps, ratings)):
            rows.append({
                "user_id":    user_id,
                "product_id": (products[i] if products else f"P{random.randint(1, 200):04d}"),
                "timestamp":  ts,
                "rating":     r,
            })
        return rows

    base = datetime(2024, 1, 1)
    rows = []

    # Normal user
    for i in range(8):
        rows += make_user_reviews(
            "normal_user_1",
            [base + timedelta(days=i * 15)],
            [random.randint(2, 4)],
        )

    # Bot: HIGH_VELOCITY — 8 reviews in 30 minutes
    rows += make_user_reviews(
        "bot_velocity",
        [base + timedelta(minutes=m) for m in range(0, 31, 4)],
        [5] * 8,
    )

    # Bot: DORMANT_BURST — 6-month silence then 4 same-day reviews
    burst_base = base + timedelta(days=180)
    rows += make_user_reviews(
        "bot_dormant",
        [base] + [burst_base + timedelta(hours=h) for h in range(4)],
        [5, 5, 5, 5, 5],
    )

    # Bot: RATING_EXTREMISM — 10 reviews, all 1s or 5s
    rows += make_user_reviews(
        "bot_extremist",
        [base + timedelta(days=d) for d in range(10)],
        [5, 1, 5, 5, 1, 5, 1, 5, 1, 5],
    )

    demo_df = pd.DataFrame(rows)
    print("=== Demo: Reviewer Behavior Timeline Detector ===\n")
    flagged = detect_reviewer_anomalies(demo_df, verbose=False)

    if not flagged.empty:
        for _, row in flagged.iterrows():
            print(f"👤 User        : {row['user_id']}")
            print(f"   Reviews     : {row['total_reviews']}")
            print(f"   Risk Score  : {row['risk_score']}/100")
            for f in row["flags"]:
                print(f"   ⚠️  {f['flag_type']} — {f['detail']}")
            print()
    else:
        print("No flagged users.")
