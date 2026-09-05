"""
Drift Monitor — Track 02: AI Risk Manager (supplementary feature)

A model trained today will get stale as return patterns shift (new product
lines, new season, new fraud patterns). This module checks whether the
model's predictions are still calibrated against REALITY by comparing
predicted risk buckets against actual outcomes on a rolling recent window —
without needing to retrain anything to detect the problem.

This is deliberately simple and explainable (bucket comparison), not a
black-box drift-detection algorithm — consistent with this track's "honest,
explainable" philosophy.
"""

import numpy as np
import pandas as pd


BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DRIFT_ALERT_THRESHOLD = 0.15  # if actual rate in a bucket differs from the
                                # bucket's midpoint by more than this, flag drift


def check_calibration(y_true, y_proba, bucket_edges=BUCKET_EDGES):
    """
    Returns a per-bucket table: predicted score range, count of orders,
    and ACTUAL return rate observed in that bucket. A well-calibrated model
    should show actual rate roughly matching the bucket's midpoint.
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    rows = []
    drift_detected = False

    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        bucket = df[(df["y_proba"] >= lo) & (df["y_proba"] < hi)]
        if len(bucket) == 0:
            continue
        actual_rate = bucket["y_true"].mean()
        expected_mid = (lo + hi) / 2
        gap = abs(actual_rate - expected_mid)
        flagged = gap > DRIFT_ALERT_THRESHOLD

        if flagged:
            drift_detected = True

        rows.append({
            "score_bucket": f"{lo:.1f}-{hi:.1f}",
            "n_orders": len(bucket),
            "actual_return_rate": round(actual_rate, 3),
            "expected_midpoint": round(expected_mid, 3),
            "calibration_gap": round(gap, 3),
            "drift_flag": flagged,
        })

    return pd.DataFrame(rows), drift_detected


def check_recent_window_drift(recent_predicted_avg, recent_actual_rate, alert_threshold=0.10):
    """
    Simpler day-to-day operational check: compare the model's average
    predicted risk score over the last N orders against the actual observed
    return rate over that same window. Big divergence = retrain signal.
    """
    gap = abs(recent_predicted_avg - recent_actual_rate)
    drifted = gap > alert_threshold
    return {
        "recent_predicted_avg": round(recent_predicted_avg, 3),
        "recent_actual_rate": round(recent_actual_rate, 3),
        "gap": round(gap, 3),
        "drift_detected": drifted,
        "recommendation": (
            "Retrain recommended — model's risk estimates have drifted from "
            "observed reality." if drifted else
            "Model still well-calibrated against recent outcomes."
        ),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from risk_scorer import load_and_split, build_pipeline

    X_train, X_test, y_train, y_test, full_df = load_and_split("../data/orders.csv")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("--- Calibration check on held-out test set ---")
    table, drifted = check_calibration(y_test.values, y_proba)
    print(table.to_string(index=False))
    print(f"\nOverall drift detected: {drifted}")

    print("\n--- Simulated operational drift check (last 100 orders) ---")
    # Simulate a scenario: actual return rate spiked recently (e.g. new
    # product line launched with unexpectedly high returns) vs. what the
    # model — trained on older data — is predicting
    simulated_recent_actual = y_test.values[-100:].mean() + 0.12  # inject a shift
    simulated_recent_predicted = y_proba[-100:].mean()
    result = check_recent_window_drift(simulated_recent_predicted, simulated_recent_actual)
    for k, v in result.items():
        print(f"  {k}: {v}")
