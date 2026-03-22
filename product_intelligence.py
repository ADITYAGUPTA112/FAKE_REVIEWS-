from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class SuspiciousCluster:
    cluster_id: int
    users: List[str]
    products: List[str]
    cluster_size: int
    density: float
    avg_edge_weight: float
    total_co_review_events: int
    risk_score: int


def _as_timedelta(window: Any) -> pd.Timedelta:
    if isinstance(window, pd.Timedelta):
        return window
    return pd.to_timedelta(window)


def build_reviewer_coordination_graph(
    df: pd.DataFrame,
    user_col: str = "user_id",
    product_col: str = "product_id",
    timestamp_col: str = "timestamp",
    min_shared_products: int = 2,
    min_co_review_events: int = 2,
    time_window: Any = "48h",
) -> nx.Graph:
    """
    Builds a user-user graph where an edge links users that reviewed the same
    products within `time_window`.

    Edge attributes:
      - shared_products: list of shared products
      - shared_product_count: number of shared products
      - co_review_events: number of within-window pair events
      - avg_time_delta_hours: average posting gap for co-review events
      - weight: composite edge weight used for scoring
    """
    required = {user_col, product_col, timestamp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[[user_col, product_col, timestamp_col]].copy()
    work = work.dropna(subset=[user_col, product_col, timestamp_col])
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col])
    if work.empty:
        return nx.Graph()

    window = _as_timedelta(time_window)
    edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
    user_review_counts = work[user_col].value_counts().to_dict()
    user_product_counts = work.groupby(user_col)[product_col].nunique().to_dict()

    for product, grp in work.groupby(product_col):
        grp = grp.sort_values(timestamp_col).reset_index(drop=True)
        users = grp[user_col].astype(str).tolist()
        times = grp[timestamp_col].tolist()

        left = 0
        for right in range(len(grp)):
            while left < right and (times[right] - times[left]) > window:
                left += 1
            for i in range(left, right):
                u1, u2 = users[i], users[right]
                if u1 == u2:
                    continue
                pair = tuple(sorted((u1, u2)))
                if pair not in edge_data:
                    edge_data[pair] = {
                        "shared_products": set(),
                        "co_review_events": 0,
                        "time_deltas_hours": [],
                    }
                event_delta_h = abs((times[right] - times[i]).total_seconds()) / 3600.0
                edge_data[pair]["shared_products"].add(str(product))
                edge_data[pair]["co_review_events"] += 1
                edge_data[pair]["time_deltas_hours"].append(event_delta_h)

    g = nx.Graph()
    for user in work[user_col].astype(str).unique():
        g.add_node(
            user,
            review_count=int(user_review_counts.get(user, 0)),
            unique_products=int(user_product_counts.get(user, 0)),
        )

    for (u1, u2), stats in edge_data.items():
        shared_products = stats["shared_products"]
        co_events = int(stats["co_review_events"])
        if len(shared_products) < min_shared_products or co_events < min_co_review_events:
            continue

        avg_delta_h = (
            float(np.mean(stats["time_deltas_hours"]))
            if stats["time_deltas_hours"]
            else float("nan")
        )
        # Higher when many co-events + many shared products + very small delay.
        recency_factor = 1.0 / (1.0 + max(avg_delta_h, 0.0))
        weight = (co_events * 0.7) + (len(shared_products) * 1.2) + (recency_factor * 2.0)
        g.add_edge(
            u1,
            u2,
            shared_products=sorted(shared_products),
            shared_product_count=len(shared_products),
            co_review_events=co_events,
            avg_time_delta_hours=round(avg_delta_h, 3),
            weight=round(weight, 3),
        )

    return g


def _cluster_risk_score(
    cluster_size: int,
    density: float,
    avg_edge_weight: float,
    total_co_events: int,
) -> int:
    size_component = min(cluster_size / 10.0, 1.0) * 25.0
    density_component = min(max(density, 0.0), 1.0) * 35.0
    weight_component = min(avg_edge_weight / 8.0, 1.0) * 25.0
    volume_component = min(total_co_events / 40.0, 1.0) * 15.0
    return int(round(size_component + density_component + weight_component + volume_component))


def flag_connected_subgraphs(
    graph: nx.Graph,
    min_nodes: int = 3,
    min_density: float = 0.45,
    min_avg_edge_weight: float = 2.5,
) -> List[Dict[str, Any]]:
    """
    Flags connected components likely to be coordinated bot/reviewer rings.
    """
    flagged: List[Dict[str, Any]] = []
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return flagged

    for component in nx.connected_components(graph):
        if len(component) < min_nodes:
            continue

        sub = graph.subgraph(component).copy()
        density = float(nx.density(sub))
        edge_weights = [float(data.get("weight", 0.0)) for _, _, data in sub.edges(data=True)]
        avg_edge_weight = float(np.mean(edge_weights)) if edge_weights else 0.0

        total_co_events = int(
            sum(int(data.get("co_review_events", 0)) for _, _, data in sub.edges(data=True))
        )
        products = set()
        for _, _, data in sub.edges(data=True):
            products.update(data.get("shared_products", []))

        if density < min_density or avg_edge_weight < min_avg_edge_weight:
            continue

        risk = _cluster_risk_score(
            cluster_size=len(component),
            density=density,
            avg_edge_weight=avg_edge_weight,
            total_co_events=total_co_events,
        )
        flagged.append(
            {
                "users": sorted(component),
                "products": sorted(products),
                "cluster_size": len(component),
                "density": round(density, 3),
                "avg_edge_weight": round(avg_edge_weight, 3),
                "total_co_review_events": total_co_events,
                "risk_score": risk,
            }
        )

    flagged.sort(key=lambda x: x["risk_score"], reverse=True)
    for idx, item in enumerate(flagged, start=1):
        item["cluster_id"] = idx
    return flagged


def detect_coordinated_bot_networks(
    df: pd.DataFrame,
    user_col: str = "user_id",
    product_col: str = "product_id",
    timestamp_col: str = "timestamp",
    time_window: Any = "48h",
    min_shared_products: int = 2,
    min_co_review_events: int = 2,
    min_nodes: int = 3,
    min_density: float = 0.45,
    min_avg_edge_weight: float = 2.5,
) -> Dict[str, Any]:
    graph = build_reviewer_coordination_graph(
        df=df,
        user_col=user_col,
        product_col=product_col,
        timestamp_col=timestamp_col,
        min_shared_products=min_shared_products,
        min_co_review_events=min_co_review_events,
        time_window=time_window,
    )
    suspicious_clusters = flag_connected_subgraphs(
        graph=graph,
        min_nodes=min_nodes,
        min_density=min_density,
        min_avg_edge_weight=min_avg_edge_weight,
    )
    return {
        "suspicious_clusters": suspicious_clusters,
        "graph_summary": {
            "total_nodes": int(graph.number_of_nodes()),
            "total_edges": int(graph.number_of_edges()),
            "total_flagged_clusters": len(suspicious_clusters),
        },
    }


def calculate_discrepancy_score(
    platform_scores: Dict[str, float],
    outlier_z_threshold: float = 1.5,
    absolute_delta_threshold: float = 15.0,
) -> Dict[str, Any]:
    """
    Computes discrepancy among trust scores from multiple platforms.
    """
    clean_scores = {
        str(platform): float(score)
        for platform, score in (platform_scores or {}).items()
        if score is not None
    }
    if len(clean_scores) < 2:
        return {
            "platform_scores": clean_scores,
            "discrepancy_score": 0.0,
            "spread": 0.0,
            "mean_score": float(np.mean(list(clean_scores.values()))) if clean_scores else 0.0,
            "std_score": 0.0,
            "outlier_platforms": [],
            "is_wildly_inconsistent": False,
        }

    values = np.array(list(clean_scores.values()), dtype=float)
    mean_score = float(values.mean())
    std_score = float(values.std(ddof=0))
    spread = float(values.max() - values.min())
    std_safe = std_score if std_score > 1e-9 else 1.0

    z_scores = {p: float((s - mean_score) / std_safe) for p, s in clean_scores.items()}
    outliers = []
    for platform, score in clean_scores.items():
        z = abs(z_scores[platform])
        abs_delta = abs(score - mean_score)
        if z >= outlier_z_threshold or abs_delta >= absolute_delta_threshold:
            outliers.append(platform)

    normalized_spread = min(spread / 100.0, 1.0)
    max_abs_z = max(abs(v) for v in z_scores.values())
    normalized_z = min(max_abs_z / 3.0, 1.0)
    discrepancy_score = round((0.7 * normalized_spread + 0.3 * normalized_z) * 100.0, 1)

    return {
        "platform_scores": clean_scores,
        "mean_score": round(mean_score, 3),
        "std_score": round(std_score, 3),
        "spread": round(spread, 3),
        "z_scores": {k: round(v, 3) for k, v in z_scores.items()},
        "outlier_platforms": sorted(outliers),
        "discrepancy_score": discrepancy_score,
        "is_wildly_inconsistent": discrepancy_score >= 35.0 or len(outliers) > 0,
    }
