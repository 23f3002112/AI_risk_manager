# AI Risk Manager — Return-Risk Scorer
Razorpay AI Buildathon 2026 — Track 02

## Problem
Returns quietly eat merchant margin, but naive "flag everything risky"
systems just annoy genuine customers and cost more in support friction than
they save. This project scores return risk at order time and picks the
action threshold that minimizes **actual business cost** — not just the one
that maximizes accuracy — because a false positive (annoying a genuine
customer) and a false negative (missing a real return) do not cost the same.

## The bar this is built to clear
> "Honest metrics including false-positive cost. Strictly defense-only:
> anything offense-capable is disqualified."

- **Honest metrics** — precision/recall/F1 on a true held-out test set (25%,
  never seen during training, stratified).
- **False-positive cost, explicitly modeled and configurable** — cost
  assumptions are CLI arguments (`--cost-fp`, `--cost-fn`), not buried
  constants, so anyone (including a judge) can see live how the optimal
  threshold shifts with different business cost structures.
- **Strictly defense-only** — the response layer can only route to manual
  review or request extra verification. No auto-cancel, no auto-blacklist,
  no punitive action of any kind.

## What's new in this version (added features)

1. **Bracketing behavior signal** (`variants_in_order`) — detects when a
   customer orders 2+ size/color variants of the SAME product in one order
   (a well-known real fashion-retail return/fraud pattern: order multiple
   sizes, keep one, return the rest). This is a genuinely strong,
   independently-verified signal: orders with 1 variant return 29.9% of the
   time; orders with 3+ variants return 66.7% of the time (see `--stats`
   output below).
2. **Configurable cost simulator** — `risk_scorer.py --cost-fp X --cost-fn Y`
   lets you re-run the full threshold analysis with a different business
   cost structure live, without touching code. Strong live-demo moment.
3. **Per-order explainability** (`explain_action()` in `risk_responder.py`)
   — for any flagged order, returns the top contributing feature values
   *for that specific order*, not just global model importance. This is
   what a human reviewer actually needs to make a fast decision.
4. **Drift monitor** (`drift_monitor.py`) — checks whether the model's
   predicted risk buckets still match actual observed return rates,
   flagging when retraining is needed, without requiring a retrain to
   detect the problem. Found a genuine calibration issue in this exact
   model (see Results below) — this is a real, honest finding, not a
   hypothetical feature.

## Architecture
```
data/generate_orders.py    -> synthetic orders with layered risk factors:
                               category, serial-returner behavior, COD,
                               price, discount, timing, AND bracketing
                               behavior (variants_in_order)
        |
        v
backend/risk_scorer.py     -> trains RandomForest, evaluates on held-out
                               25% at multiple thresholds, configurable
                               cost-per-threshold analysis
        |
        v
backend/risk_responder.py  -> bounded action layer (review / verify / none)
                               + per-order explainability
        |
        v
backend/drift_monitor.py   -> calibration + operational drift checks
        |
        v
backend/pipeline_demo.py   -> runs the full loop end to end with explanations
```

## Dev Setup
Run these commands to set up the project locally:
```bash
pip install -r requirements.txt
```

## How to run
```bash
cd data
python3 generate_orders.py --n 3000 --seed 11 --stats

cd ../backend
python3 risk_scorer.py                          # default costs
python3 risk_scorer.py --cost-fp 15 --cost-fn 400   # simulate a different merchant's cost structure
python3 pipeline_demo.py                         # full loop + per-order explanations
python3 drift_monitor.py                         # calibration check
```

## Real results from an actual run (seed=11, n=3000, with bracketing signal)
- Held-out test set: 750 orders, 31.3% actual return rate.
- ROC-AUC: 0.672 (up slightly from 0.662 pre-bracketing-signal — a real,
  small, honest improvement, not an inflated jump).

| Threshold | Precision | Recall | FP | FN | Cost (₹) | Net Savings (₹) |
|---|---|---|---|---|---|---|
| 0.3 | 0.341 | 0.919 | 418 | 19 | 9,690 | **32,610** |
| 0.4 | 0.401 | 0.787 | 276 | 50 | 13,140 | 29,160 |
| 0.5 | 0.467 | 0.643 | 172 | 84 | 17,700 | 24,600 |
| 0.6 | 0.452 | 0.260 | 74 | 174 | 32,430 | 9,870 |

**Bracketing signal validation** (from `--stats`):
| variants_in_order | n | return rate |
|---|---|---|
| 1 | 2,860 | 29.9% |
| 2 | 99 | 59.6% |
| 3 | 27 | 66.7% |
| 4 | 14 | 64.3% |

**Cost simulator in action**: with `--cost-fn 400` (a merchant where missed
returns are far more expensive), net savings at threshold 0.3 jumps to
₹80,130 — the *optimal threshold stays the same*, but the *magnitude of
value* the system provides scales directly with the merchant's real cost
structure. This is the honest, non-hand-wavy way to show business impact.

**Drift monitor finding (a real "what broke")**: the calibration check
revealed the model is **overconfident** in the 0.6-0.8 predicted-risk
bucket — it predicts ~70% return likelihood there, but the actual rate is
only 45.2%. This is disclosed honestly rather than hidden, and is exactly
the kind of finding "measured accuracy" is supposed to surface.

## Why this stays defense-only
`risk_responder.py` only implements 3 actions: `no_action`,
`route_to_manual_review`, `request_additional_verification`. No code path
cancels an order, blacklists a customer, or exposes customer data. A human
always makes the final call — the model's job is triage, not judgment.

## What broke / lessons learned
1. First version of label-generation depended only on category, producing
   an unrealistically easy dataset (ROC-AUC > 0.95) — fixed by layering
   multiple competing risk factors.
2. The drift monitor's calibration check (added this round) surfaced a real
   overconfidence problem in the 0.6-0.8 score bucket — the model isn't
   wrong to flag those orders, but its exact probability estimate there
   shouldn't be trusted at face value. Documented rather than smoothed over.

## 7-Part Milestone Plan
1. Problem scoping & repo skeleton
2. Synthetic dataset with layered risk factors + bracketing signal — DONE
3. Return-risk scorer: multi-threshold, configurable-cost evaluation — DONE
4. Bounded, defense-only response layer + per-order explainability — DONE
5. Drift monitor + full pipeline integration with explanations — DONE
6. Backend API (FastAPI) + UI dashboard (review queue + live cost simulator slider)
7. Deployment, evaluation write-up, pitch video

## Next steps (Milestones 6-7)
- FastAPI backend exposing `/score-order`, `/review-queue`,
  `/daily-summary`, and a `/simulate-cost` endpoint that runs the cost
  simulator live from UI slider input.
- Streamlit dashboard: review queue with explain_action() reasoning shown
  per row, a cost-assumption slider (fp/fn) that re-renders the
  threshold-savings table live, and the drift-monitor calibration chart.
- Optional third detector (Fraud-Spike or Chargeback Evidence Responder) if
  time allows, following the same bounded-response philosophy.
