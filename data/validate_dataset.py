"""
Dataset Validation Tool — Track 02: AI Risk Manager
Validates the generated synthetic orders dataset for sanity and effect sizes.
"""
import pandas as pd
import sys

def validate_dataset(filepath="orders.csv"):
    df = pd.read_csv(filepath)
    print(f"Validating {filepath} ({len(df)} rows)...")

    # 1. No nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("ERROR: Null values found in the dataset!")
        print(null_counts[null_counts > 0])
        sys.exit(1)
    else:
        print("[OK] No null values found.")

    # 2. No negative prices
    if (df["price"] < 0).any():
        print("ERROR: Negative prices found!")
        sys.exit(1)
    else:
        print("[OK] No negative prices found.")

    # 3. Return rate within plausible 15-45% overall range
    overall_return_rate = df["returned"].mean()
    if not (0.15 <= overall_return_rate <= 0.45):
        print(f"ERROR: Overall return rate {overall_return_rate:.1%} is outside the plausible 15-45% range!")
        sys.exit(1)
    else:
        print(f"[OK] Overall return rate is plausible ({overall_return_rate:.1%}).")

    # 4. Effect size checks for engineered features
    engineered_features = ["variants_in_order", "delivery_address_change_count", "category", "payment_method"]
    print("\nChecking effect sizes for engineered features...")
    
    for feature in engineered_features:
        if feature not in df.columns:
            continue
        
        rates = df.groupby(feature)["returned"].mean()
        min_rate = rates.min()
        max_rate = rates.max()
        effect_size = max_rate - min_rate
        
        print(f"  {feature}: min_rate={min_rate:.1%}, max_rate={max_rate:.1%} (diff: {effect_size:.1%})")
        
        if effect_size < 0.05:
            print(f"    WARNING: Effect size for {feature} looks too weak (< 5 percentage points).")
        elif effect_size > 0.50:
            print(f"    WARNING: Effect size for {feature} looks too strong (> 50 percentage points). Might be unrealistically easy.")

if __name__ == "__main__":
    validate_dataset()
