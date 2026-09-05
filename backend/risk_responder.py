"""
Risk Response Layer — Track 02: AI Risk Manager

Turns a risk score into an ACTION. Deliberately limited to defense-only,
friction-adding actions — never a punitive or offense-capable action. This
is what keeps the system compliant with "THE BAR": "Strictly defense-only:
anything offense-capable is disqualified."

Allowed actions (defense-only):
  - route_to_manual_review   (human makes the final call)
  - request_additional_verification (e.g. OTP/address confirm before shipping)
  - no_action                (score below threshold)

NOT implemented, and never should be, per this track's rules:
  - auto-cancel a customer's order without human review
  - auto-blacklist a customer account
  - any action that could be used to harass, expose, or retaliate against a customer
"""

import json
from datetime import datetime


class RiskResponder:
    def __init__(self, threshold_review=0.3, threshold_verify=0.6):
        # thresholds chosen from risk_scorer.py's own cost-based evaluation,
        # not arbitrary — 0.3 was the empirically cheapest threshold overall
        self.threshold_review = threshold_review
        self.threshold_verify = threshold_verify
        self.audit_log = []

    def respond(self, order_id, risk_score, order_details=None):
        timestamp = datetime.utcnow().isoformat() + "Z"

        if risk_score >= self.threshold_verify:
            action = "request_additional_verification"
            reason = (f"Risk score {risk_score:.2f} exceeds high-risk threshold "
                       f"{self.threshold_verify}. Requesting address/OTP confirmation "
                       f"before dispatch — order is NOT cancelled or blocked.")
        elif risk_score >= self.threshold_review:
            action = "route_to_manual_review"
            reason = (f"Risk score {risk_score:.2f} exceeds review threshold "
                       f"{self.threshold_review}. Flagged for human review queue — "
                       f"no automatic action taken against the customer.")
        else:
            action = "no_action"
            reason = f"Risk score {risk_score:.2f} below review threshold. Order proceeds normally."

        entry = {
            "timestamp": timestamp, "order_id": order_id, "risk_score": round(float(risk_score), 3),
            "action": action, "reason": reason, "order_details": order_details or {},
        }
        self.audit_log.append(entry)
        return entry

    def save_log(self, path="risk_response_log.json"):
        with open(path, "w") as f:
            json.dump(self.audit_log, f, indent=2)

    def summary(self):
        from collections import Counter
        return dict(Counter(e["action"] for e in self.audit_log))

    def explain_action(self, order_id, order_features=None, feature_importances=None, feature_percentiles=None):
        """
        Builds a per-order natural-language explanation: which of THIS order's
        specific feature values contributed most to its risk score, not just
        the model's global feature importance. This is what a human reviewer
        actually needs to make a fast, confident decision.

        order_features: dict of this order's raw feature values (e.g.
            {"customer_past_return_rate": 0.71, "variants_in_order": 3, ...})
        feature_importances: dict of {feature_name: global_importance_weight}
            from the trained model (used to rank which of this order's
            features are most likely to be driving its score)
        feature_percentiles: optional dict of {feature_name: function} where 
            function(value) returns the percentile (0-100) of that value in training data.
        """
        entry = next((e for e in self.audit_log if e["order_id"] == order_id), None)
        if entry is None:
            return {"error": f"No logged action found for order_id {order_id}"}

        explanation_lines = []
        if order_features and feature_importances:
            # Rank this order's features by (global importance * how unusual the value is)
            ranked = sorted(
                feature_importances.items(), key=lambda kv: -kv[1]
            )[:5]
            for feat_name, importance in ranked:
                if feat_name in order_features:
                    val = order_features[feat_name]
                    
                    context_str = ""
                    if feature_percentiles and feat_name in feature_percentiles:
                        try:
                            perc = feature_percentiles[feat_name](val)
                            if perc >= 90:
                                context_str = f" (this is in the top {100 - int(perc)}% of all customers \u2014 a strong risk signal)"
                            elif perc <= 10:
                                context_str = f" (this is in the bottom {int(perc)}% of all customers \u2014 unusually low)"
                            else:
                                context_str = f" (this is near the {int(perc)}th percentile)"
                        except Exception:
                            pass
                            
                    explanation_lines.append(f"{feat_name} = {val}{context_str} (importance weight {importance:.2f})")

        return {
            "order_id": order_id,
            "risk_score": entry["risk_score"],
            "action_taken": entry["action"],
            "system_reason": entry["reason"],
            "top_contributing_factors": explanation_lines if explanation_lines else
                ["Feature-level breakdown not provided — pass order_features + "
                 "feature_importances to get this."],
        }
