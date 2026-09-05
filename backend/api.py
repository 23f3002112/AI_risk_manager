"""
Backend API — Track 02: AI Risk Manager
FastAPI wrapper around the unified RiskManagerOrchestrator.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime
import json
import os
import sys
import numpy as np

# Ensure backend modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import RiskManagerOrchestrator
from risk_scorer import evaluate_at_threshold

app = FastAPI(title="AI Risk Manager API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global orchestrator
orchestrator = RiskManagerOrchestrator()


class OrderPayload(BaseModel):
    order_id: str
    price: float
    order_hour: int
    is_weekend: int
    discount_percent: float
    customer_account_age_days: int
    customer_past_orders: int
    customer_past_return_rate: float
    variants_in_order: int
    delivery_address_change_count: int
    category: str
    payment_method: str


class ResolveDecision(BaseModel):
    decision: str  # "approve" or "reject"


class CostSimRequest(BaseModel):
    cost_fp: float
    cost_fn: float


def clean_numpy(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError

@app.post("/score-order")
def score_order(order: OrderPayload):
    try:
        explanation = orchestrator.score_order(order.model_dump())
        return json.loads(json.dumps(explanation, default=clean_numpy))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/review-queue")
def review_queue():
    """
    Returns flagged orders that don't have a resolution yet.
    """
    flagged = []
    resolved_ids = set()
    
    # Process audit log backwards to find resolved
    for entry in reversed(orchestrator.responder.audit_log):
        if entry["action"] == "human_review_resolved":
            resolved_ids.add(entry["order_id"])
            
    for entry in orchestrator.responder.audit_log:
        if entry["action"] in ["route_to_manual_review", "request_additional_verification"]:
            if entry["order_id"] not in resolved_ids:
                # Add explanation
                numeric_features = {k: v for k, v in entry.get("order_details", {}).items() if k in orchestrator.feature_importances}
                exp = orchestrator.responder.explain_action(
                    entry["order_id"],
                    order_features=numeric_features,
                    feature_importances=orchestrator.feature_importances,
                    feature_percentiles=orchestrator.feature_percentiles
                )
                flagged.append({
                    "order_id": entry["order_id"],
                    "risk_score": entry["risk_score"],
                    "action": entry["action"],
                    "reason": entry["reason"],
                    "explanation": exp["top_contributing_factors"]
                })
                
    return {"queue": flagged}


@app.post("/review-queue/{order_id}/resolve")
def resolve_order(order_id: str, payload: ResolveDecision):
    orchestrator.responder.audit_log.append({
        "order_id": order_id,
        "action": "human_review_resolved",
        "decision": payload.decision,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "risk_score": 0.0,
        "reason": f"Manually resolved by human: {payload.decision}",
        "order_details": {}
    })
    return {"status": "success", "order_id": order_id, "decision": payload.decision}


@app.get("/daily-summary")
def daily_summary():
    summary = orchestrator.get_daily_summary()
    return json.loads(json.dumps(summary, default=clean_numpy))


@app.post("/simulate-cost")
def simulate_cost(req: CostSimRequest):
    """
    Re-evaluates thresholds dynamically using the test set probabilities.
    """
    y_proba = orchestrator.pipeline.predict_proba(orchestrator.X_test)[:, 1]
    y_true = orchestrator.y_test.values
    
    results = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r = evaluate_at_threshold(y_true, y_proba, threshold, len(y_true),
                                  cost_fp=req.cost_fp, cost_fn=req.cost_fn)
        results.append(r)
        
    return {"results": json.loads(json.dumps(results, default=clean_numpy))}


@app.get("/drift-status")
def drift_status():
    drift_data = orchestrator.check_drift()
    return json.loads(json.dumps(drift_data, default=clean_numpy))
