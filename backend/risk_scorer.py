"""
Return-Risk Scorer — Track 02: AI Risk Manager
Razorpay AI Buildathon 2026

Trains a classifier to predict return probability per order, evaluated on a
TRUE held-out test set (never seen during training), reporting:
  - Precision, Recall, F1 at multiple thresholds
  - ROC-AUC
  - FALSE-POSITIVE COST: what it actually costs the business when the model
    wrongly flags a genuine order as high-risk (this is what "THE BAR"
    explicitly demands: "honest metrics including false-positive cost")

Design note: this is a SCORER, not a blocker. It outputs a risk score used to
route orders to manual review / extra verification — it never auto-cancels
an order on its own. That's what keeps this "strictly defense-only": it adds
friction/review, it does not take unilateral punitive action against anyone.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# --- Business cost assumptions (make these explicit and editable — this IS
#     the "honest metrics including false-positive cost" requirement).
#     These are DEFAULTS — override via CLI (--cost-fp / --cost-fn) so a judge
#     or merchant can see live how the optimal threshold shifts with their
#     actual cost structure, instead of trusting one hardcoded assumption. ---
COST_FALSE_POSITIVE = 15     # INR-equivalent friction cost: extra verification step
                              # on a genuine customer (support time + customer annoyance/churn risk)
COST_FALSE_NEGATIVE = 180    # INR-equivalent: average cost of an unflagged, unprevented
                              # return (reverse logistics + restocking + payment processing fees)

FEATURE_COLUMNS_NUMERIC = [
    "price", "order_hour", "is_weekend", "discount_percent",
    "customer_account_age_days", "customer_past_orders", "customer_past_return_rate",
    "variants_in_order",  # bracketing behavior signal: 2+ = ordering multiple
                          # sizes/colors of the same item intending to return most
]
FEATURE_COLUMNS_CATEGORICAL = ["category", "payment_method"]


def load_and_split(path, test_size=0.25, seed=42):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL]
    y = df["returned"]
    # stratify=y ensures test set has same return-rate balance as training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, df


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS_CATEGORICAL),
    ], remainder="passthrough")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_at_threshold(y_true, y_proba, threshold, n_total_orders,
                           cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE):
    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fp * cost_fp) + (fn * cost_fn)
    # Baseline: cost of doing NOTHING (flag zero orders) = cost of every actual
    # return happening unflagged. This is the honest comparison point.
    n_actual_positives = tp + fn
    baseline_cost = n_actual_positives * cost_fn

    return {
        "threshold": threshold,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
        "total_cost_inr": round(total_cost, 2),
        "baseline_cost_do_nothing_inr": round(baseline_cost, 2),
        "net_savings_inr": round(baseline_cost - total_cost, 2),
    }


def run(cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE):
    X_train, X_test, y_train, y_test, full_df = load_and_split("../data/orders.csv")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"Cost assumptions in use: false-positive=INR {cost_fp}, false-negative=INR {cost_fn}")

    print(f"Test set size: {len(y_test)} orders (held out, never seen during training)")
    print(f"Test set actual return rate: {y_test.mean():.1%}")
    print(f"ROC-AUC: {auc:.3f}\n")

    print("=" * 100)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>6} "
          f"{'FP':>5} {'FN':>5} {'Cost (INR)':>12} {'Baseline (INR)':>15} {'Net Savings':>12}")
    print("=" * 100)

    results = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = evaluate_at_threshold(y_test.values, y_proba, threshold, len(y_test),
                                   cost_fp=cost_fp, cost_fn=cost_fn)
        results.append(r)
        print(f"{r['threshold']:>10} {r['precision']:>10} {r['recall']:>8} {r['f1']:>6} "
              f"{r['false_positives']:>5} {r['false_negatives']:>5} "
              f"{r['total_cost_inr']:>12} {r['baseline_cost_do_nothing_inr']:>15} "
              f"{r['net_savings_inr']:>12}")

    best = max(results, key=lambda r: r["net_savings_inr"])
    print(f"\nBEST THRESHOLD by net cost savings: {best['threshold']} "
          f"(saves INR {best['net_savings_inr']} vs. flagging nothing)")

    print("\n--- Full classification report at threshold=0.5 ---")
    y_pred_50 = (y_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_50, target_names=["Not Returned", "Returned"]))

    # Feature importance (explainability — required for "honest" system)
    model = pipeline.named_steps["model"]
    cat_features = list(pipeline.named_steps["preprocess"]
                         .named_transformers_["cat"].get_feature_names_out(FEATURE_COLUMNS_CATEGORICAL))
    all_features = cat_features + FEATURE_COLUMNS_NUMERIC
    importances = sorted(zip(all_features, model.feature_importances_), key=lambda x: -x[1])
    print("\n--- Top 8 risk signals (feature importance) ---")
    for name, imp in importances[:8]:
        print(f"  {name:<35} {imp:.3f}")

    pd.DataFrame(results).to_csv("threshold_evaluation.csv", index=False)
    print("\nFull threshold evaluation saved to: threshold_evaluation.csv")

    return pipeline, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-fp", type=float, default=COST_FALSE_POSITIVE,
                         help="Cost (INR) of wrongly flagging a genuine order")
    parser.add_argument("--cost-fn", type=float, default=COST_FALSE_NEGATIVE,
                         help="Cost (INR) of missing an actual return")
    args = parser.parse_args()
    run(cost_fp=args.cost_fp, cost_fn=args.cost_fn)
