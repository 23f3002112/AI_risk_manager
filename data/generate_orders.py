"""
Synthetic Order + Return Dataset Generator — Track 02: AI Risk Manager
Razorpay AI Buildathon 2026 — Direction: Return-Risk Scorer

Generates orders with realistic risk signals (customer history, product
category, payment method, price, order timing) and a REALISTIC label:
whether the order was eventually returned. The label generation deliberately
encodes real-world return-risk patterns (high-value fashion items with COD
have higher return rates, repeat "serial returners" exist, etc.) so a model
trained on this has genuine signal to learn, not random noise.

Usage:
    python generate_orders.py --n 3000 --seed 11
"""

import argparse
import csv
import random


CATEGORIES = ["Apparel", "Footwear", "Electronics", "Home & Kitchen", "Beauty"]
PAYMENT_METHODS = ["UPI", "Card", "COD", "Netbanking", "Wallet"]

# Base return-rate priors per category (real e-commerce pattern: apparel/footwear
# return far more than electronics/home goods)
CATEGORY_RETURN_BASE = {
    "Apparel": 0.28, "Footwear": 0.24, "Beauty": 0.10,
    "Electronics": 0.07, "Home & Kitchen": 0.06,
}


def generate_customers(n_customers, rng):
    """Some customers are 'serial returners' — a real, well-known risk pattern.
    A separate, smaller group are 'bracketers' — customers who order multiple
    sizes/colors of the SAME item in one order, intending to keep only one and
    return the rest. This is a distinct behavior from serial returning (a
    bracketer might have a low overall return rate on OTHER items) and is a
    well-documented, high-value signal in real fashion e-commerce fraud/return
    analytics."""
    customers = []
    for i in range(n_customers):
        cust_id = f"CUST{1000+i}"
        is_serial_returner = rng.random() < 0.08
        is_bracketer = rng.random() < 0.06  # independent from serial-returner flag
        is_high_risk_geo_hopper = rng.random() < 0.05  # NEW: hidden trait for address changes
        account_age_days = rng.randint(1, 900)
        past_orders = rng.randint(0, 40)
        past_return_rate = (
            round(rng.uniform(0.55, 0.9), 2) if is_serial_returner
            else round(rng.uniform(0.0, 0.25), 2)
        )
        customers.append({
            "customer_id": cust_id,
            "account_age_days": account_age_days,
            "past_orders": past_orders,
            "past_return_rate": past_return_rate,
            "is_serial_returner": is_serial_returner,  # hidden ground-truth signal
            "is_bracketer": is_bracketer,               # hidden ground-truth signal
            "is_high_risk_geo_hopper": is_high_risk_geo_hopper, # hidden ground-truth signal
        })
    return customers


def generate_orders(n, seed, n_customers=600):
    rng = random.Random(seed)
    customers = generate_customers(n_customers, rng)

    rows = []
    for i in range(n):
        cust = rng.choice(customers)
        category = rng.choice(CATEGORIES)
        payment_method = rng.choice(PAYMENT_METHODS)
        price = round(rng.uniform(200, 8000), 2)
        order_hour = rng.randint(0, 23)
        is_weekend = rng.random() < 0.3
        discount_percent = rng.choice([0, 0, 5, 10, 20, 30])

        # --- Bracketing signal: in apparel/footwear, a bracketer sometimes
        # orders multiple size/color variants of the SAME product in one order
        # (observable at order time — this is what makes it a usable FEATURE,
        # unlike "is_bracketer" itself which is hidden ground truth) ---
        variants_in_order = 1
        if category in ("Apparel", "Footwear") and cust["is_bracketer"] and rng.random() < 0.7:
            variants_in_order = rng.randint(2, 4)
        elif rng.random() < 0.03:
            # small baseline noise: even non-bracketers occasionally order 2 sizes
            variants_in_order = 2

        # --- Delivery address change signal: a geo-hopper frequently changes
        # their delivery address post-order creation (observable at order time) ---
        delivery_address_change_count = 0
        if cust["is_high_risk_geo_hopper"] and rng.random() < 0.8:
            delivery_address_change_count = rng.randint(1, 3)
        elif rng.random() < 0.02:
            # small baseline noise: normal customers sometimes change address once or twice
            delivery_address_change_count = rng.randint(1, 2)

        # --- Compute TRUE underlying return probability from real risk factors ---
        p = CATEGORY_RETURN_BASE[category]

        # Bracketing is a STRONG standalone signal — ordering 2+ variants of the
        # same product very reliably predicts at least one gets returned
        if variants_in_order >= 3:
            p += 0.35
        elif variants_in_order == 2:
            p += 0.18

        # Geo-hoppers changing address frequently increases return and fraud risk
        if delivery_address_change_count >= 2:
            p += 0.25
        elif delivery_address_change_count == 1:
            p += 0.10

        # Serial returners massively increase risk
        p += cust["past_return_rate"] * 0.5

        # COD orders return more (no upfront payment commitment)
        if payment_method == "COD":
            p += 0.10

        # High-value orders in fashion categories return more (size/fit issues)
        if category in ("Apparel", "Footwear") and price > 3000:
            p += 0.08

        # New accounts with no history are riskier (less signal, but slightly higher risk)
        if cust["account_age_days"] < 30 and cust["past_orders"] == 0:
            p += 0.05

        # Heavy discount can correlate with impulse buys that get returned
        if discount_percent >= 20:
            p += 0.04

        # Late-night orders (impulse buying pattern)
        if order_hour >= 23 or order_hour <= 3:
            p += 0.03

        p = min(max(p, 0.02), 0.95)  # keep bounded
        returned = 1 if rng.random() < p else 0

        rows.append({
            "order_id": f"ORD{10000+i}",
            "customer_id": cust["customer_id"],
            "category": category,
            "payment_method": payment_method,
            "price": price,
            "order_hour": order_hour,
            "is_weekend": int(is_weekend),
            "discount_percent": discount_percent,
            "customer_account_age_days": cust["account_age_days"],
            "customer_past_orders": cust["past_orders"],
            "customer_past_return_rate": cust["past_return_rate"],
            "variants_in_order": variants_in_order,  # NEW: bracketing behavior feature
            "delivery_address_change_count": delivery_address_change_count, # NEW: geo-hopper feature
            "returned": returned,  # LABEL
        })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--outfile", type=str, default="orders.csv")
    parser.add_argument("--stats", action="store_true", help="print return-rate breakdowns")
    args = parser.parse_args()

    rows = generate_orders(args.n, args.seed)

    with open(args.outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return_rate = sum(r["returned"] for r in rows) / len(rows)
    print(f"Generated {len(rows)} orders")
    print(f"Overall return rate: {return_rate:.1%}")
    print(f"Written to: {args.outfile}")

    if args.stats:
        def breakdown(key):
            groups = {}
            for r in rows:
                groups.setdefault(r[key], []).append(r["returned"])
            print(f"\n--- Return rate by {key} ---")
            for val, outcomes in sorted(groups.items(), key=lambda x: str(x[0])):
                rate = sum(outcomes) / len(outcomes)
                print(f"  {str(val):<20} n={len(outcomes):<6} return_rate={rate:.1%}")

        breakdown("category")
        breakdown("payment_method")
        breakdown("variants_in_order")
        
        # Breakdown with buckets for delivery_address_change_count
        groups = {}
        for r in rows:
            val = r["delivery_address_change_count"]
            bucket = "2+" if val >= 2 else str(val)
            groups.setdefault(bucket, []).append(r["returned"])
        print("\n--- Return rate by delivery_address_change_count ---")
        for val, outcomes in sorted(groups.items(), key=lambda x: str(x[0])):
            rate = sum(outcomes) / len(outcomes)
            print(f"  {str(val):<20} n={len(outcomes):<6} return_rate={rate:.1%}")
