"""
========================================================
  MODULE 3 — Reviewer–Product Trust Graph
  Coordinated Review-Ring Detector using NetworkX
========================================================
Builds a bipartite graph (Users ↔ Products) from review data,
then projects it to a user–user co-review graph to surface
suspicious clusters ("review rings") where groups of users
have collectively reviewed the same set of obscure products —
a near-impossible coincidence in real traffic.

Pipeline:
    1. Build bipartite graph  Users ↔ Products
    2. Identify "obscure" products (few unique reviewers)
    3. Project onto a User–User graph where edge weight =
       number of shared obscure products reviewed
    4. Filter to strong edges (≥ min_shared_products)
    5. Detect connected components → suspicious clusters
    6. Score & return ranked list of ring candidates

Expected DataFrame columns:
    user_id    : str
    product_id : str
    timestamp  : str / datetime
    rating     : int (1–5)
========================================================
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from itertools import combinations
from collections import defaultdict
from datetime import timedelta
from typing import List, Dict, Optional


# ── Tunable parameters ────────────────────────────────────────────────────────
OBSCURE_MAX_REVIEWERS   = 50    # products reviewed by ≤ this many users → "obscure"
MIN_SHARED_PRODUCTS     = 3     # shared obscure products needed to link two users
MIN_CLUSTER_SIZE        = 4     # minimum users in a ring to flag it
COORDINATED_WINDOW_DAYS = 7     # users who reviewed same product within this window
                                #   get an extra "temporal coordination" flag
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Constructs a bipartite graph with two node types:
        - 'user'    nodes (bipartite=0)
        - 'product' nodes (bipartite=1)

    Each review becomes a weighted edge.  Multiple reviews of the same
    product by the same user accumulate as edge weight.

    Returns a NetworkX Graph with node attribute ``bipartite``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    B = nx.Graph()

    # Add typed nodes
    for uid in df["user_id"].unique():
        B.add_node(uid, bipartite=0, node_type="user")
    for pid in df["product_id"].unique():
        B.add_node(pid, bipartite=1, node_type="product")

    # Add edges (reviews)
    for _, row in df.iterrows():
        u, p = row["user_id"], row["product_id"]
        if B.has_edge(u, p):
            B[u][p]["weight"] += 1
            B[u][p]["ratings"].append(row["rating"])
        else:
            B.add_edge(
                u, p,
                weight=1,
                ratings=[row["rating"]],
                first_review=row["timestamp"],
            )
    return B


# ─────────────────────────────────────────────────────────────────────────────
# OBSCURE PRODUCT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_obscure_products(df: pd.DataFrame) -> set:
    """
    Returns product IDs reviewed by ≤ OBSCURE_MAX_REVIEWERS unique users.
    These are the products a genuine organic audience is unlikely to share.
    """
    counts = df.groupby("product_id")["user_id"].nunique()
    return set(counts[counts <= OBSCURE_MAX_REVIEWERS].index)


# ─────────────────────────────────────────────────────────────────────────────
# USER–USER PROJECTION
# ─────────────────────────────────────────────────────────────────────────────

def build_user_graph(
    df: pd.DataFrame,
    obscure_products: set,
) -> nx.Graph:
    """
    Projects bipartite data onto a User–User graph:
        - Each node = a user
        - Each edge = the two users co-reviewed ≥ 1 obscure product
        - Edge attribute ``shared_products`` lists all common obscure products
        - Edge attribute ``shared_count`` is the overlap size

    Only edges with shared_count ≥ MIN_SHARED_PRODUCTS are kept.
    """
    obscure_df     = df[df["product_id"].isin(obscure_products)]
    user_to_prods  = obscure_df.groupby("user_id")["product_id"].apply(set).to_dict()

    G = nx.Graph()
    G.add_nodes_from(user_to_prods.keys())

    users = list(user_to_prods.keys())
    for u1, u2 in combinations(users, 2):
        shared = user_to_prods[u1] & user_to_prods[u2]
        if len(shared) >= MIN_SHARED_PRODUCTS:
            G.add_edge(
                u1, u2,
                shared_products=list(shared),
                shared_count=len(shared),
            )

    return G


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL COORDINATION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_temporal_coordination(
    user_ids: List[str],
    df: pd.DataFrame,
    product_ids: List[str],
) -> Dict:
    """
    For a cluster's shared products, checks whether the users reviewed them
    within a suspiciously tight time window (COORDINATED_WINDOW_DAYS).

    Returns a dict:
        coordinated_pairs : int   — # (user, product) pairs within the window
        coordination_rate : float — fraction of shared reviews that are timed
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    relevant = df[
        df["user_id"].isin(user_ids) & df["product_id"].isin(product_ids)
    ]

    total_pairs  = 0
    coord_pairs  = 0

    for product_id, product_group in relevant.groupby("product_id"):
        if len(product_group) < 2:
            continue
        timestamps = product_group["timestamp"].sort_values().values
        for i in range(len(timestamps)):
            for j in range(i + 1, len(timestamps)):
                total_pairs += 1
                gap = pd.Timestamp(timestamps[j]) - pd.Timestamp(timestamps[i])
                if gap <= timedelta(days=COORDINATED_WINDOW_DAYS):
                    coord_pairs += 1

    rate = coord_pairs / total_pairs if total_pairs else 0.0
    return {
        "coordinated_pairs": coord_pairs,
        "total_pairs": total_pairs,
        "coordination_rate": round(rate, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _score_cluster(
    cluster_size: int,
    density: float,
    avg_shared: float,
    coordination_rate: float,
) -> int:
    """
    Heuristic risk score [0–100].
    Weighted combination of:
        - Cluster size       (larger = more suspicious)
        - Graph density      (denser = more suspicious)
        - Avg shared products (higher = more suspicious)
        - Temporal coordination rate
    """
    size_score   = min(cluster_size / MIN_CLUSTER_SIZE * 20, 30)
    density_score = density * 25
    shared_score  = min(avg_shared / MIN_SHARED_PRODUCTS * 20, 25)
    time_score    = coordination_rate * 20

    return min(int(size_score + density_score + shared_score + time_score), 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_review_rings(
    df: pd.DataFrame,
    verbose: bool = False,
) -> List[Dict]:
    """
    Full pipeline: detects coordinated review rings.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: user_id, product_id, timestamp, rating

    verbose : bool
        Print step-by-step progress

    Returns
    -------
    List of cluster dicts, sorted by risk_score descending.
    Each dict contains:
        cluster_id, users, cluster_size, graph_density,
        shared_products, avg_shared_per_pair,
        temporal_coordination, risk_score
    """
    # ── Validate input ────────────────────────────────────────────────────────
    required = {"user_id", "product_id", "timestamp", "rating"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Step 1: Identify obscure products ─────────────────────────────────────
    obscure = find_obscure_products(df)
    if verbose:
        print(f"  Step 1: {len(obscure)} obscure products "
              f"(≤{OBSCURE_MAX_REVIEWERS} reviewers each)")

    if not obscure:
        print("⚠️  No obscure products found — raise OBSCURE_MAX_REVIEWERS threshold.")
        return []

    # ── Step 2: Build user–user co-review graph ───────────────────────────────
    user_graph = build_user_graph(df, obscure)
    if verbose:
        print(f"  Step 2: User–user graph: "
              f"{user_graph.number_of_nodes()} nodes, "
              f"{user_graph.number_of_edges()} edges")

    # ── Step 3: Find connected components ────────────────────────────────────
    clusters = []
    for component in nx.connected_components(user_graph):
        if len(component) < MIN_CLUSTER_SIZE:
            continue

        subgraph = user_graph.subgraph(component)
        density  = nx.density(subgraph)
        users    = list(component)

        # Collect all shared products across the cluster
        all_shared = set()
        shared_counts = []
        for u1, u2, data in subgraph.edges(data=True):
            all_shared.update(data.get("shared_products", []))
            shared_counts.append(data.get("shared_count", 0))

        avg_shared = np.mean(shared_counts) if shared_counts else 0

        # Temporal coordination
        time_info = check_temporal_coordination(
            users, df, list(all_shared)
        )

        risk = _score_cluster(
            len(component), density, avg_shared,
            time_info["coordination_rate"],
        )

        clusters.append({
            "cluster_id":            len(clusters) + 1,
            "users":                 users,
            "cluster_size":          len(component),
            "graph_density":         round(density, 3),
            "shared_products":       list(all_shared),
            "num_shared_products":   len(all_shared),
            "avg_shared_per_pair":   round(avg_shared, 2),
            "temporal_coordination": time_info,
            "risk_score":            risk,
        })

    # ── Step 4: Sort by risk ──────────────────────────────────────────────────
    clusters.sort(key=lambda c: c["risk_score"], reverse=True)
    if verbose:
        print(f"  Step 3: Found {len(clusters)} suspicious clusters")

    return clusters


def print_ring_report(clusters: List[Dict]) -> None:
    """Pretty-prints the ring detection results."""
    if not clusters:
        print("✅  No review rings detected.")
        return

    print(f"\n{'='*60}")
    print(f"  REVIEW-RING DETECTOR — {len(clusters)} cluster(s) flagged")
    print(f"{'='*60}\n")

    for c in clusters:
        risk_icon = "🔴" if c["risk_score"] >= 75 else "🟡" if c["risk_score"] >= 45 else "🟢"
        tc = c["temporal_coordination"]
        print(f"  {risk_icon} Cluster #{c['cluster_id']}  |  Risk: {c['risk_score']}/100")
        print(f"     Users ({c['cluster_size']})  : {', '.join(c['users'][:8])}"
              + ("…" if c["cluster_size"] > 8 else ""))
        print(f"     Graph density       : {c['graph_density']}")
        print(f"     Shared products     : {c['num_shared_products']} "
              f"(avg {c['avg_shared_per_pair']} per pair)")
        print(f"     Temporal sync rate  : {tc['coordination_rate']:.1%} "
              f"({tc['coordinated_pairs']}/{tc['total_pairs']} pairs within "
              f"{COORDINATED_WINDOW_DAYS} days)")
        print(f"     Product IDs         : {', '.join(c['shared_products'][:6])}"
              + ("…" if len(c["shared_products"]) > 6 else ""))
        print()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime
    import random

    random.seed(99)
    base = datetime(2024, 3, 1)

    rows = []

    # ── Ring A: 6 users all reviewing the same 4 obscure products ────────────
    ring_a_users    = [f"ring_a_user_{i}" for i in range(6)]
    ring_a_products = ["OBSCURE_001", "OBSCURE_002", "OBSCURE_003", "OBSCURE_004"]

    for u in ring_a_users:
        for p in ring_a_products:
            # All reviews within 3 days (highly coordinated)
            rows.append({
                "user_id":    u,
                "product_id": p,
                "timestamp":  base + timedelta(days=random.randint(0, 3)),
                "rating":     5,
            })

    # ── Ring B: 5 users sharing 3 obscure products, loosely timed ────────────
    ring_b_users    = [f"ring_b_user_{i}" for i in range(5)]
    ring_b_products = ["OBSCURE_005", "OBSCURE_006", "OBSCURE_007"]

    for u in ring_b_users:
        for p in ring_b_products:
            rows.append({
                "user_id":    u,
                "product_id": p,
                "timestamp":  base + timedelta(days=random.randint(0, 30)),
                "rating":     5,
            })

    # ── Normal users: spread across many popular products ────────────────────
    popular_products = [f"POP_{i:03d}" for i in range(200)]
    for u_id in range(80):
        for _ in range(random.randint(1, 5)):
            rows.append({
                "user_id":    f"normal_user_{u_id}",
                "product_id": random.choice(popular_products),
                "timestamp":  base + timedelta(days=random.randint(0, 365)),
                "rating":     random.randint(1, 5),
            })

    demo_df = pd.DataFrame(rows)
    print("=== Demo: Reviewer–Product Trust Graph ===\n")

    # Build and visualise bipartite graph info
    bipartite_g = build_bipartite_graph(demo_df)
    users_nodes = {n for n, d in bipartite_g.nodes(data=True) if d.get("bipartite") == 0}
    prod_nodes  = {n for n, d in bipartite_g.nodes(data=True) if d.get("bipartite") == 1}
    print(f"Bipartite graph: {len(users_nodes)} users, "
          f"{len(prod_nodes)} products, "
          f"{bipartite_g.number_of_edges()} review edges\n")

    # Detect rings
    rings = detect_review_rings(demo_df, verbose=True)
    print_ring_report(rings)