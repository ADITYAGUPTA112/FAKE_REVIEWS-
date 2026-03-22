from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class DiscountSpikeEvent:
    event_date: pd.Timestamp
    price_before: float
    price_after: float
    discount_pct: float
    pre_fake_volume: float
    baseline_fake_volume: float
    pre_spike_zscore: float
    ratio_to_baseline: float
    flagged: bool


def analyze_price_fake_correlation(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    fake_col: str = "fake_review_volume",
    discount_threshold_pct: float = -10.0,
    lookback_days: int = 7,
    baseline_window_days: int = 30,
    min_ratio_to_baseline: float = 1.5,
    min_pre_spike_zscore: float = 2.0,
) -> Dict[str, object]:
    """
    Detect whether fake review volume spikes before discount events.

    Method:
    1. Detect discount events where day-over-day price change <= threshold.
    2. For each event, compare fake-volume in the pre-window (lookback_days)
       against a longer baseline period before that window.
    3. Compute:
       - z-score of pre-window sum against rolling historical distribution
       - ratio to baseline average
    4. Flag events passing both thresholds.
    """
    required = {date_col, price_col, fake_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[[date_col, price_col, fake_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work[fake_col] = pd.to_numeric(work[fake_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=[price_col])

    if work.empty or len(work) < max(lookback_days + baseline_window_days, 10):
        return {
            "events": [],
            "summary": {
                "total_discount_events": 0,
                "flagged_events": 0,
                "flag_rate": 0.0,
                "avg_pre_spike_zscore": 0.0,
                "avg_pre_to_baseline_ratio": 0.0,
                "lead_lag_peak_days": None,
                "lead_lag_peak_corr": None,
            },
        }

    work["price_change_pct"] = work[price_col].pct_change() * 100.0
    discount_idx = work.index[work["price_change_pct"] <= discount_threshold_pct].tolist()

    events: List[DiscountSpikeEvent] = []
    pre_event_sums: List[float] = []
    baseline_means: List[float] = []
    zscores: List[float] = []
    flagged_count = 0

    for idx in discount_idx:
        event_date = work.loc[idx, date_col]
        price_after = float(work.loc[idx, price_col])
        price_before = float(work.loc[idx - 1, price_col]) if idx > 0 else price_after
        discount_pct = float(work.loc[idx, "price_change_pct"])

        pre_start = event_date - pd.Timedelta(days=lookback_days)
        base_start = pre_start - pd.Timedelta(days=baseline_window_days)

        pre_slice = work[(work[date_col] >= pre_start) & (work[date_col] < event_date)]
        base_slice = work[(work[date_col] >= base_start) & (work[date_col] < pre_start)]

        pre_sum = float(pre_slice[fake_col].sum())
        baseline_mean = float(base_slice[fake_col].mean()) if not base_slice.empty else 0.0
        baseline_std = float(base_slice[fake_col].std(ddof=0)) if len(base_slice) > 1 else 0.0

        # Compare pre-window sum vs expected sum from baseline daily average.
        expected_sum = baseline_mean * lookback_days
        std_safe = (baseline_std * np.sqrt(max(lookback_days, 1))) if baseline_std > 0 else 1.0
        z = float((pre_sum - expected_sum) / std_safe)
        ratio = float(pre_sum / max(expected_sum, 1e-9)) if expected_sum > 0 else float("inf")

        flagged = bool(z >= min_pre_spike_zscore and ratio >= min_ratio_to_baseline)
        flagged_count += int(flagged)

        events.append(
            DiscountSpikeEvent(
                event_date=event_date,
                price_before=price_before,
                price_after=price_after,
                discount_pct=discount_pct,
                pre_fake_volume=pre_sum,
                baseline_fake_volume=expected_sum,
                pre_spike_zscore=z,
                ratio_to_baseline=ratio,
                flagged=flagged,
            )
        )
        pre_event_sums.append(pre_sum)
        baseline_means.append(expected_sum)
        zscores.append(z)

    # Lead-lag correlation: positive lag means fake reviews lead price changes.
    fake_series = work[fake_col].astype(float)
    drop_signal = (work["price_change_pct"] <= discount_threshold_pct).astype(float)
    lags = range(-14, 15)
    lag_corr = {}
    for lag in lags:
        shifted = fake_series.shift(lag)
        valid = shifted.notna() & drop_signal.notna()
        if valid.sum() < 5:
            continue
        corr = shifted[valid].corr(drop_signal[valid])
        if pd.notna(corr):
            lag_corr[int(lag)] = float(corr)

    peak_lag = None
    peak_corr = None
    if lag_corr:
        peak_lag, peak_corr = max(lag_corr.items(), key=lambda kv: abs(kv[1]))

    event_dicts = [
        {
            "event_date": e.event_date.strftime("%Y-%m-%d"),
            "price_before": round(e.price_before, 4),
            "price_after": round(e.price_after, 4),
            "discount_pct": round(e.discount_pct, 3),
            "pre_fake_volume": round(e.pre_fake_volume, 4),
            "baseline_fake_volume": round(e.baseline_fake_volume, 4),
            "pre_spike_zscore": round(e.pre_spike_zscore, 4),
            "ratio_to_baseline": round(e.ratio_to_baseline, 4)
            if np.isfinite(e.ratio_to_baseline)
            else None,
            "flagged": e.flagged,
        }
        for e in events
    ]

    summary = {
        "total_discount_events": len(events),
        "flagged_events": int(flagged_count),
        "flag_rate": round((flagged_count / len(events)) if events else 0.0, 4),
        "avg_pre_spike_zscore": round(float(np.mean(zscores)) if zscores else 0.0, 4),
        "avg_pre_to_baseline_ratio": round(
            float(np.mean([x for x in (e["ratio_to_baseline"] for e in event_dicts) if x is not None]))
            if event_dicts
            else 0.0,
            4,
        ),
        "lead_lag_peak_days": int(peak_lag) if peak_lag is not None else None,
        "lead_lag_peak_corr": round(float(peak_corr), 4) if peak_corr is not None else None,
    }
    return {"events": event_dicts, "summary": summary, "lead_lag_correlation": lag_corr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect fake-review spikes before price discounts.")
    parser.add_argument("--input", required=True, help="Path to CSV file.")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--price-col", default="price")
    parser.add_argument("--fake-col", default="fake_review_volume")
    parser.add_argument("--discount-threshold-pct", type=float, default=-10.0)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--baseline-window-days", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = analyze_price_fake_correlation(
        df=df,
        date_col=args.date_col,
        price_col=args.price_col,
        fake_col=args.fake_col,
        discount_threshold_pct=args.discount_threshold_pct,
        lookback_days=args.lookback_days,
        baseline_window_days=args.baseline_window_days,
    )

    print("\n=== Summary ===")
    for k, v in result["summary"].items():
        print(f"{k}: {v}")
    print("\n=== Discount Events ===")
    print(pd.DataFrame(result["events"]).to_string(index=False))


if __name__ == "__main__":
    main()
