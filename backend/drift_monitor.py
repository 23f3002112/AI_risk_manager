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


def check_calibration(X_test, y_true, y_proba, bucket_edges=BUCKET_EDGES):
    """
    Returns a per-bucket table: predicted score range, count of orders,
    and ACTUAL return rate observed in that bucket. A well-calibrated model
    should show actual rate roughly matching the bucket's midpoint.
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    # Combine with X_test to analyze feature differences
    df = pd.concat([df.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
    
    rows = []
    drift_detected = False
    
    well_calibrated_indices = []
    bucket_indices = []

    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        bucket = df[(df["y_proba"] >= lo) & (df["y_proba"] < hi)]
        if len(bucket) == 0:
            bucket_indices.append(None)
            continue
            
        actual_rate = bucket["y_true"].mean()
        expected_mid = (lo + hi) / 2
        gap = abs(actual_rate - expected_mid)
        flagged = gap > DRIFT_ALERT_THRESHOLD

        if flagged:
            drift_detected = True
        else:
            well_calibrated_indices.extend(bucket.index.tolist())

        bucket_indices.append({
            "lo": lo, "hi": hi, "bucket_df": bucket, "flagged": flagged,
            "actual_rate": actual_rate, "expected_mid": expected_mid, "gap": gap
        })

    baseline_df = df.loc[well_calibrated_indices]
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    baseline_means = baseline_df[numeric_cols].mean() if len(baseline_df) > 0 else None

    miscalibration_reasons = {}

    for i in range(len(bucket_edges) - 1):
        b_info = bucket_indices[i]
        if b_info is None:
            continue
            
        bucket_name = f"{b_info['lo']:.1f}-{b_info['hi']:.1f}"
        rows.append({
            "score_bucket": bucket_name,
            "n_orders": len(b_info['bucket_df']),
            "actual_return_rate": round(b_info['actual_rate'], 3),
            "expected_midpoint": round(b_info['expected_mid'], 3),
            "calibration_gap": round(b_info['gap'], 3),
            "drift_flag": b_info['flagged'],
        })
        
        if b_info['flagged'] and baseline_means is not None:
            bucket_means = b_info['bucket_df'][numeric_cols].mean()
            # find biggest diffs relative to baseline (percent diff)
            diffs = abs(bucket_means - baseline_means) / (baseline_means.replace(0, 1e-9))
            top_diff_feats = diffs.sort_values(ascending=False).head(3)
            
            reasons = []
            for feat, _ in top_diff_feats.items():
                b_val = bucket_means[feat]
                base_val = baseline_means[feat]
                reasons.append(f"{feat} is {b_val:.2f} vs baseline {base_val:.2f}")
                
            miscalibration_reasons[bucket_name] = reasons

    return pd.DataFrame(rows), drift_detected, miscalibration_reasons


def check_recent_window_drift(recent_predicted_avg, recent_actual_rate, days_since_last_retrain=None, alert_threshold=0.10):
    """
    Simpler day-to-day operational check: compare the model's average
    predicted risk score over the last N orders against the actual observed
    return rate over that same window. Big divergence = retrain signal.
    """
    gap = abs(recent_predicted_avg - recent_actual_rate)
    drifted = gap > alert_threshold
    
    recommendation = "Model still well-calibrated against recent outcomes."
    if drifted:
        if days_since_last_retrain is not None and days_since_last_retrain >= 90:
            recommendation = "Retrain immediately — model's risk estimates have drifted and it has been >= 90 days since last retrain."
        else:
            recommendation = "Consider retraining soon — model's risk estimates have drifted from observed reality."

    return {
        "recent_predicted_avg": round(recent_predicted_avg, 3),
        "recent_actual_rate": round(recent_actual_rate, 3),
        "gap": round(gap, 3),
        "days_since_last_retrain": days_since_last_retrain,
        "drift_detected": drifted,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    import sys
    import scipy.stats
    sys.path.insert(0, ".")
    from risk_scorer import load_and_split, build_pipeline, FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL
    from risk_responder import RiskResponder

    X_train, X_test, y_train, y_test, full_df = load_and_split("../data/orders.csv")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("--- Calibration check on held-out test set ---")
    table, drifted, miscalibration_reasons = check_calibration(X_test, y_test.values, y_proba)
    print(table.to_string(index=False))
    print(f"\nOverall drift detected: {drifted}")
    
    if miscalibration_reasons:
        print("\n--- Miscalibration Reasons (Feature Diff vs Well-Calibrated Baseline) ---")
        for bucket, reasons in miscalibration_reasons.items():
            print(f"Bucket {bucket}:")
            for r in reasons:
                print(f"  - {r}")

    print("\n--- Simulated operational drift check (last 100 orders) ---")
    simulated_recent_actual = y_test.values[-100:].mean() + 0.25  # inject a shift
    simulated_recent_predicted = y_proba[-100:].mean()
    result = check_recent_window_drift(simulated_recent_predicted, simulated_recent_actual, days_since_last_retrain=95)
    for k, v in result.items():
        print(f"  {k}: {v}")
        
    print("\n--- Explainability Demo ---")
    feature_percentiles = {}
    for col in FEATURE_COLUMNS_NUMERIC:
        train_vals = X_train[col].values
        # Use default arguments in lambda to capture current value of train_vals
        feature_percentiles[col] = lambda x, vals=train_vals: scipy.stats.percentileofscore(vals, x)
        
    responder = RiskResponder()
    model = pipeline.named_steps["model"]
    cat_features = list(pipeline.named_steps["preprocess"]
                         .named_transformers_["cat"].get_feature_names_out(FEATURE_COLUMNS_CATEGORICAL))
    all_features = cat_features + FEATURE_COLUMNS_NUMERIC
    feature_importances = dict(zip(all_features, model.feature_importances_))
    
    high_risk_idx = y_proba.argmax()
    order_features = X_test.iloc[high_risk_idx].to_dict()
    
    responder.audit_log.append({
        "order_id": "TEST_ORDER_1",
        "risk_score": float(y_proba[high_risk_idx]),
        "action": "route_to_manual_review",
        "reason": "Test"
    })
    
    print("\nWithout percentiles context:")
    exp_basic = responder.explain_action("TEST_ORDER_1", order_features, feature_importances)
    for f in exp_basic["top_contributing_factors"]:
        print(f"  {f}")
        
    print("\nWith percentiles context:")
    exp_enhanced = responder.explain_action("TEST_ORDER_1", order_features, feature_importances, feature_percentiles)
    for f in exp_enhanced["top_contributing_factors"]:
        print(f"  {f}")
