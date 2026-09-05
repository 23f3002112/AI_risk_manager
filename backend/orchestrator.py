"""
Unified Orchestrator — Track 02: AI Risk Manager

This orchestrator ties together the Risk Scorer, Risk Responder, and Drift Monitor
into a single, cohesive engine that a UI or API can consume directly.
"""

import pandas as pd
import scipy.stats
import json

from risk_scorer import (
    load_and_split, build_pipeline, FEATURE_COLUMNS_NUMERIC, FEATURE_COLUMNS_CATEGORICAL,
    COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE, evaluate_at_threshold
)
from risk_responder import RiskResponder
from drift_monitor import check_calibration, check_recent_window_drift


class RiskManagerOrchestrator:
    def __init__(self, data_path="../data/orders.csv", threshold_review=0.3, threshold_verify=0.6):
        print("Initializing RiskManagerOrchestrator... (Training model)")
        self.X_train, self.X_test, self.y_train, self.y_test, self.full_df = load_and_split(data_path)
        
        self.pipeline = build_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        
        self.responder = RiskResponder(threshold_review=threshold_review, threshold_verify=threshold_verify)
        
        # Compute feature importances
        model = self.pipeline.named_steps["model"]
        cat_features = list(self.pipeline.named_steps["preprocess"]
                             .named_transformers_["cat"].get_feature_names_out(FEATURE_COLUMNS_CATEGORICAL))
        all_features = cat_features + FEATURE_COLUMNS_NUMERIC
        self.feature_importances = dict(zip(all_features, model.feature_importances_))
        
        # Compute feature percentiles for contextual explanations
        self.feature_percentiles = {}
        for col in FEATURE_COLUMNS_NUMERIC:
            train_vals = self.X_train[col].values
            # Default argument binds the current array
            self.feature_percentiles[col] = lambda x, vals=train_vals: scipy.stats.percentileofscore(vals, x)
            
        print("Orchestrator ready.\n")
        
    def score_order(self, order_dict):
        """
        Scores a single order, logs the action, and returns the full explanation.
        """
        # Convert single dict to DataFrame
        df_order = pd.DataFrame([order_dict])
        
        proba = float(self.pipeline.predict_proba(df_order)[:, 1][0])
        
        order_id = order_dict.get("order_id", "UNKNOWN_ORDER")
        
        # Log action via responder
        self.responder.respond(order_id, proba, order_details=order_dict)
        
        # Get explanation
        numeric_features = {k: v for k, v in order_dict.items() if k in FEATURE_COLUMNS_NUMERIC}
        explanation = self.responder.explain_action(
            order_id,
            order_features=numeric_features,
            feature_importances=self.feature_importances,
            feature_percentiles=self.feature_percentiles
        )
        return explanation

    def run_batch(self, df_batch=None, y_true=None):
        """
        Scores a batch of orders (e.g. the test set).
        Returns action summary and cost metrics.
        """
        if df_batch is None:
            df_batch = self.X_test
            y_true = self.y_test
            
        y_proba = self.pipeline.predict_proba(df_batch)[:, 1]
        
        # Log all orders in responder
        for idx, (order_idx, proba) in enumerate(zip(df_batch.index, y_proba)):
            # Fallback to index if order_id not in df
            order_id = self.full_df.loc[order_idx, "order_id"] if "order_id" in self.full_df.columns else f"batch_order_{idx}"
            self.responder.respond(order_id, proba)
            
        action_summary = self.responder.summary()
        
        # Calculate cost savings using the best threshold logic from risk_scorer
        metrics = {}
        if y_true is not None:
            res = evaluate_at_threshold(y_true.values, y_proba, self.responder.threshold_review, len(y_true),
                                        cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE)
            metrics = res
            
        return {
            "action_summary": action_summary,
            "metrics": metrics,
            "y_proba": y_proba
        }
        
    def check_drift(self, X_batch=None, y_true=None, y_proba=None):
        """
        Wraps drift monitor logic.
        """
        if X_batch is None:
            X_batch = self.X_test
            y_true = self.y_test
            y_proba = self.pipeline.predict_proba(X_batch)[:, 1]
            
        table, drifted, miscalibration_reasons = check_calibration(X_batch, y_true.values, y_proba)
        
        return {
            "drift_detected": drifted,
            "calibration_table": table.to_dict(orient="records"),
            "miscalibration_reasons": miscalibration_reasons
        }
        
    def get_daily_summary(self, df_batch=None, y_true=None):
        """
        High-level orchestration method that combines run_batch and check_drift
        into a single dashboard-ready dictionary.
        """
        if df_batch is None:
            df_batch = self.X_test
            y_true = self.y_test
            
        batch_results = self.run_batch(df_batch, y_true)
        drift_results = self.check_drift(df_batch, y_true, batch_results["y_proba"])
        
        # Remove raw probabilities from the final output for clean JSON
        del batch_results["y_proba"]
        
        summary = {
            "batch_processing": batch_results,
            "drift_monitoring": drift_results
        }
        return summary


if __name__ == "__main__":
    orchestrator = RiskManagerOrchestrator()
    
    print("=" * 60)
    print("RUNNING GET_DAILY_SUMMARY() ON TEST SET")
    print("=" * 60)
    
    daily_summary = orchestrator.get_daily_summary()
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
            
    print(json.dumps(daily_summary, indent=2, cls=NpEncoder))
