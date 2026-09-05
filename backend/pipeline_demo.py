"""
End-to-end demo — Track 02: AI Risk Manager
Trains the return-risk scorer, applies it to held-out test orders, and routes
each through the bounded, defense-only response layer. This is the single
script that demonstrates the full loop for a live demo/pitch.
"""

import pandas as pd
from risk_scorer import load_and_split, build_pipeline
from risk_responder import RiskResponder


def run_pipeline():
    X_train, X_test, y_train, y_test, full_df = load_and_split("../data/orders.csv")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    responder = RiskResponder(threshold_review=0.3, threshold_verify=0.6)

    # Extract global feature importances once, used by explain_action() for
    # per-order explanations
    model = pipeline.named_steps["model"]
    from risk_scorer import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC
    cat_features = list(pipeline.named_steps["preprocess"]
                         .named_transformers_["cat"].get_feature_names_out(FEATURE_COLUMNS_CATEGORICAL))
    all_features = cat_features + FEATURE_COLUMNS_NUMERIC
    feature_importances = dict(zip(all_features, model.feature_importances_))

    order_feature_lookup = {}
    for idx, (order_idx, proba) in enumerate(zip(X_test.index, y_proba)):
        order_details = full_df.loc[order_idx, ["order_id", "category", "payment_method", "price"]].to_dict()
        responder.respond(order_details["order_id"], proba, order_details)
        order_feature_lookup[order_details["order_id"]] = full_df.loc[
            order_idx, FEATURE_COLUMNS_NUMERIC
        ].to_dict()

    responder.save_log("risk_response_log.json")

    print("=" * 60)
    print("ACTION SUMMARY across held-out test set")
    print("=" * 60)
    summary = responder.summary()
    for action, count in summary.items():
        print(f"  {action:<35} {count}")

    print(f"\nTotal orders processed: {len(y_test)}")
    print(f"Total actual returns in this batch: {int(y_test.sum())}")

    print("\n--- Sample of 3 flagged orders WITH per-order explanations ---")
    flagged = [e for e in responder.audit_log if e["action"] != "no_action"][:3]
    for e in flagged:
        explanation = responder.explain_action(
            e["order_id"],
            order_features=order_feature_lookup[e["order_id"]],
            feature_importances=feature_importances,
        )
        print(f"\n  Order: {explanation['order_id']} | Score: {explanation['risk_score']} | "
              f"Action: {explanation['action_taken']}")
        for factor in explanation["top_contributing_factors"]:
            print(f"    - {factor}")

    print("\nFull audit log saved to: risk_response_log.json")


if __name__ == "__main__":
    run_pipeline()
