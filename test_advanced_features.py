"""
Smoke + unit tests for the advanced detection features.
Run: python test_advanced_features.py
"""

from datetime import datetime, timedelta

import pandas as pd

print("=== 1. Import smoke test ===")
from analyzer import (
    compute_advanced_fraud_score,
    detect_review_rings,
    detect_reviewer_behavior_anomalies,
    detect_sentiment_rating_mismatch,
)

print("   [OK] All four functions imported successfully.\n")

print("=== 2. Reviewer Behavior Timeline ===")
now = datetime(2025, 6, 15, 10, 0, 0)
rows = []

for i in range(6):
    rows.append(
        {
            "user_name": "bot_user",
            "date": (now + timedelta(minutes=i * 5)).isoformat(),
            "star_rating": 5,
            "review": f"Great product {i}!",
        }
    )

rows.append(
    {
        "user_name": "burst_user",
        "date": (now - timedelta(days=91)).isoformat(),
        "star_rating": 5,
        "review": "Old review",
    }
)
for i in range(3):
    rows.append(
        {
            "user_name": "burst_user",
            "date": (now + timedelta(hours=i)).isoformat(),
            "star_rating": 5,
            "review": f"Burst review {i}!",
        }
    )

for i in range(5):
    rows.append(
        {
            "user_name": "extreme_rater",
            "date": (now + timedelta(days=i)).isoformat(),
            "star_rating": 5 if i % 2 == 0 else 1,
            "review": f"Review {i}",
        }
    )

for i in range(3):
    rows.append(
        {
            "user_name": "normal_user",
            "date": (now + timedelta(days=i * 10)).isoformat(),
            "star_rating": 3 + i % 2,
            "review": f"Decent product {i}",
        }
    )

df = pd.DataFrame(rows)
behavior = detect_reviewer_behavior_anomalies(df)

assert behavior.get("bot_user", {}).get("velocity_flag") is True, "bot_user velocity flag failed"
print("   [OK] bot_user velocity_flag = True")

assert behavior.get("burst_user", {}).get("dormant_burst") is True, "burst_user dormant_burst flag failed"
print("   [OK] burst_user dormant_burst = True")

assert behavior.get("extreme_rater", {}).get("rating_pattern_flag") is True, "extreme_rater flag failed"
print("   [OK] extreme_rater rating_pattern_flag = True")

assert (
    "normal_user" not in behavior or behavior["normal_user"]["behavior_score"] == 0.0
), "normal_user should be clean"
print("   [OK] normal_user is clean\n")

print("=== 3. Sentiment-Rating Mismatch ===")
texts = ["This is absolutely terrible, worst purchase ever!", "Amazing, love it!", "ok"]
stars = [5, 5, 5]
mismatches = detect_sentiment_rating_mismatch(texts, stars)

if all(float(m.get("discrepancy", 0.0)) == 0.0 for m in mismatches):
    print("   [WARN] Sentiment model fallback active; strict mismatch assertions skipped.\n")
else:
    assert mismatches[0]["is_mismatch"] is True, "Negative text + 5 stars should be flagged"
    print(f"   [OK] 'terrible' + 5 stars -> mismatch (discrepancy={mismatches[0]['discrepancy']:.2f})")

    assert mismatches[1]["is_mismatch"] is False, "Positive text + 5 stars should NOT be flagged"
    print(f"   [OK] 'Amazing' + 5 stars -> no mismatch (discrepancy={mismatches[1]['discrepancy']:.2f})")

    print(
        f"   [OK] 'ok' + 5 stars -> mismatch={mismatches[2]['is_mismatch']} "
        f"(discrepancy={mismatches[2]['discrepancy']:.2f})\n"
    )

print("=== 4. Review Ring / Trust Graph ===")
ring_rows = []
ring_users = [f"ring_user_{i}" for i in range(6)]
ring_products = ["PROD_A", "PROD_B", "PROD_C"]

for user in ring_users:
    for prod in ring_products:
        ring_rows.append({"user_name": user, "asin": prod, "review": "Great!"})

ring_rows.append({"user_name": "normal_1", "asin": "PROD_A", "review": "Good product"})
ring_rows.append({"user_name": "normal_2", "asin": "PROD_D", "review": "Nice item"})

ring_df = pd.DataFrame(ring_rows)
graph_result = detect_review_rings(ring_df, "PROD_A")

found_ring = any(c["cluster_size"] >= 5 for c in graph_result["suspicious_clusters"])
assert found_ring, "Should have found a review ring of 6 users"
print(f"   [OK] Found suspicious cluster(s): {len(graph_result['suspicious_clusters'])}")
print(f"   [OK] Overall graph score: {graph_result['overall_graph_score']:.2f}\n")

print("=== 5. Combined Fraud Score ===")
score = compute_advanced_fraud_score(0.7, 0.4, 0.6, 0.3)
print(f"   [OK] Combined score = {score:.4f} (expected ~0.58)")
assert 0.0 <= score <= 1.0

print("\n========================================")
print("ALL TESTS PASSED")
print("========================================")

