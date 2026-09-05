import requests

API_URL = "http://localhost:8000"

# 1. Check daily summary
print("Getting daily summary to populate review queue...")
requests.get(f"{API_URL}/daily-summary")

# 2. Score a sample order
print("Scoring a sample order...")
order = {
    "order_id": "TEST_API_ORDER_1",
    "price": 15000,
    "order_hour": 2,
    "is_weekend": 1,
    "discount_percent": 15.0,
    "customer_account_age_days": 5,
    "customer_past_orders": 0,
    "customer_past_return_rate": 0.9,
    "variants_in_order": 3,
    "delivery_address_change_count": 2,
    "category": "Electronics",
    "payment_method": "Credit Card"
}
resp = requests.post(f"{API_URL}/score-order", json=order)
print("Score result:", resp.json())

# 3. Check review queue
print("\nChecking review queue...")
queue = requests.get(f"{API_URL}/review-queue").json()["queue"]
print(f"Items in queue: {len(queue)}")
if queue:
    first_item = queue[0]["order_id"]
    print(f"Resolving first item: {first_item}")
    res = requests.post(f"{API_URL}/review-queue/{first_item}/resolve", json={"decision": "approve"})
    print("Resolve response:", res.json())

# 4. Check queue again
queue2 = requests.get(f"{API_URL}/review-queue").json()["queue"]
print(f"Items in queue after resolve: {len(queue2)}")

# 5. Check simulate cost
print("\nSimulating cost (fp=100, fn=500)...")
sim = requests.post(f"{API_URL}/simulate-cost", json={"cost_fp": 100, "cost_fn": 500})
print("Cost Sim Length:", len(sim.json()["results"]))

# 6. Check drift status
print("\nChecking drift status...")
drift = requests.get(f"{API_URL}/drift-status")
print("Drift Detected:", drift.json()["drift_detected"])
