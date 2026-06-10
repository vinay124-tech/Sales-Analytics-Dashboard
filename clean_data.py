"""
=============================================================================
  Antigravity Corp -- Data Cleaning & Outlier Handling Pipeline  (Phase 2)
  Author : Senior Data Engineer
  Date   : 2026-06-10
=============================================================================
  Reads the raw CSVs produced by generate_data.py and applies a sequential
  cleaning pipeline:

    Step 1  -  Deduplicate orders on order_id
    Step 2  -  Fix negative / zero quantities
    Step 3  -  IQR-based price outlier detection & capping
    Step 4  -  Data-type validation & referential integrity

  Outputs: cleaned_users.csv, cleaned_products.csv, cleaned_orders.csv
=============================================================================
"""

import os
import sys

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_USERS    = os.path.join(BASE_DIR, "users.csv")
RAW_PRODUCTS = os.path.join(BASE_DIR, "products.csv")
RAW_ORDERS   = os.path.join(BASE_DIR, "orders.csv")

OUT_USERS    = os.path.join(BASE_DIR, "cleaned_users.csv")
OUT_PRODUCTS = os.path.join(BASE_DIR, "cleaned_products.csv")
OUT_ORDERS   = os.path.join(BASE_DIR, "cleaned_orders.csv")

# ---------------------------------------------------------------------------
# Helper: section banner
# ---------------------------------------------------------------------------
def banner(step_num: int, title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  STEP {step_num} : {title}")
    print(f"{'=' * 64}")


def load_raw_files():
    """Load the three raw CSV files and abort if any are missing."""
    missing = [p for p in (RAW_USERS, RAW_PRODUCTS, RAW_ORDERS)
               if not os.path.isfile(p)]
    if missing:
        print("[ERROR] Missing raw files:")
        for m in missing:
            print(f"        - {m}")
        print("\nRun generate_data.py first.")
        sys.exit(1)

    users    = pd.read_csv(RAW_USERS)
    products = pd.read_csv(RAW_PRODUCTS)
    orders   = pd.read_csv(RAW_ORDERS)

    print("  Raw files loaded successfully:")
    print(f"    users.csv     -> {len(users):,} rows")
    print(f"    products.csv  -> {len(products):,} rows")
    print(f"    orders.csv    -> {len(orders):,} rows")

    return users, products, orders


# ===================================================================
#  STEP 1 : Deduplicate orders on order_id
# ===================================================================
def step1_deduplicate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    banner(1, "HANDLING DUPLICATE ORDER IDs")

    before = len(orders)
    dup_mask = orders.duplicated(subset=["order_id"], keep="first")
    num_dups = dup_mask.sum()

    # Show a sample of the duplicates before dropping
    if num_dups > 0:
        sample_ids = orders.loc[dup_mask, "order_id"].head(5).tolist()
        print(f"  Sample duplicate order_ids: {sample_ids}")

    orders_clean = orders.loc[~dup_mask].copy().reset_index(drop=True)
    after = len(orders_clean)

    print(f"\n  Rows before    : {before:,}")
    print(f"  Duplicates found: {num_dups:,}")
    print(f"  Rows after     : {after:,}")
    print(f"  --> Removed {num_dups} duplicate order_id rows (kept first occurrence).")

    return orders_clean


# ===================================================================
#  STEP 2 : Fix negative / zero quantities
# ===================================================================
def step2_fix_quantities(orders: pd.DataFrame) -> pd.DataFrame:
    banner(2, "HANDLING NEGATIVE / ZERO QUANTITIES")

    neg_mask  = orders["quantity"] < 0
    zero_mask = orders["quantity"] == 0
    num_neg   = neg_mask.sum()
    num_zero  = zero_mask.sum()

    print(f"  Negative quantities found : {num_neg}")
    print(f"  Zero quantities found     : {num_zero}")

    # ---- Strategy A: Convert negatives -> absolute value ----
    # Rationale: a value like -3 is most likely a manual data-entry error
    # where the operator accidentally typed a leading hyphen.
    if num_neg > 0:
        print("\n  Strategy: converting negative quantities to absolute values")
        print("  (assumption: leading-hyphen data-entry error)\n")

        sample = orders.loc[neg_mask, ["order_id", "quantity"]].head(5)
        print("  Before correction (sample):")
        for _, row in sample.iterrows():
            print(f"    order_id {int(row['order_id']):>6}  qty = {int(row['quantity'])}")

        orders.loc[neg_mask, "quantity"] = orders.loc[neg_mask, "quantity"].abs()

        print("\n  After correction (same rows):")
        for _, row in orders.loc[sample.index, ["order_id", "quantity"]].iterrows():
            print(f"    order_id {int(row['order_id']):>6}  qty = {int(row['quantity'])}")

    # ---- Strategy B: Drop zero-quantity rows (corrupted test data) ----
    if num_zero > 0:
        print(f"\n  Dropping {num_zero} zero-quantity rows (likely corrupted test rows).")
        orders = orders.loc[~zero_mask].copy().reset_index(drop=True)

    total_fixed = num_neg + num_zero
    print(f"\n  --> Fixed {num_neg} negative quantities (abs conversion).")
    print(f"  --> Dropped {num_zero} zero-quantity rows.")
    print(f"  --> Total rows affected: {total_fixed}")

    return orders


# ===================================================================
#  STEP 3 : IQR-based price outlier detection & capping
# ===================================================================
def step3_outlier_detection(products: pd.DataFrame) -> pd.DataFrame:
    banner(3, "STATISTICAL OUTLIER DETECTION (IQR METHOD)")

    prices = products["base_price"]

    q1  = prices.quantile(0.25)
    q3  = prices.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    print(f"  Price distribution statistics:")
    print(f"    Min       : ${prices.min():>12,.2f}")
    print(f"    Q1  (25%) : ${q1:>12,.2f}")
    print(f"    Median    : ${prices.median():>12,.2f}")
    print(f"    Q3  (75%) : ${q3:>12,.2f}")
    print(f"    Max       : ${prices.max():>12,.2f}")
    print(f"    IQR       : ${iqr:>12,.2f}")
    print(f"    Upper fence (Q3 + 1.5*IQR) : ${upper_fence:>12,.2f}")

    outlier_mask = products["base_price"] > upper_fence
    num_outliers = outlier_mask.sum()

    print(f"\n  Outliers detected: {num_outliers}")

    if num_outliers > 0:
        # Compute the replacement price as the median of non-outlier products
        non_outlier_prices = products.loc[~outlier_mask, "base_price"]
        replacement_price  = round(non_outlier_prices.median(), 2)

        print(f"  Replacement strategy: cap at median of non-outlier prices "
              f"(${replacement_price:,.2f})\n")

        for idx in products.index[outlier_mask]:
            row = products.loc[idx]
            original = row["base_price"]
            print(f"  CORRECTED: '{row['product_name']}' (product_id {row['product_id']})")
            print(f"             original price  = ${original:>12,.2f}")
            print(f"             corrected price = ${replacement_price:>12,.2f}")
            print(f"             delta           = -${(original - replacement_price):>12,.2f}")

            products.loc[idx, "base_price"] = replacement_price
    else:
        print("  No outliers to correct.")

    print(f"\n  --> {num_outliers} price outlier(s) capped.")
    return products


# ===================================================================
#  STEP 4 : Data-type validation & referential integrity
# ===================================================================
def step4_validate(users: pd.DataFrame,
                   products: pd.DataFrame,
                   orders: pd.DataFrame):
    banner(4, "DATA-TYPE VALIDATION & REFERENTIAL INTEGRITY")

    issues_found = 0

    # ---- 4a. Parse & normalise date columns ----
    print("  [4a] Parsing date columns ...")

    users["signup_date"] = pd.to_datetime(users["signup_date"], errors="coerce")
    bad_signup = users["signup_date"].isna().sum()
    if bad_signup:
        print(f"       WARNING: {bad_signup} unparseable signup_date values found.")
        issues_found += bad_signup
    else:
        print(f"       signup_date -> parsed OK ({users['signup_date'].dtype})")

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    bad_order_dt = orders["order_date"].isna().sum()
    if bad_order_dt:
        print(f"       WARNING: {bad_order_dt} unparseable order_date values found.")
        issues_found += bad_order_dt
    else:
        print(f"       order_date  -> parsed OK ({orders['order_date'].dtype})")

    # Normalise to ISO format strings for CSV output
    users["signup_date"]  = users["signup_date"].dt.strftime("%Y-%m-%d")
    orders["order_date"]  = orders["order_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # ---- 4b. Validate numeric types ----
    print("\n  [4b] Validating numeric columns ...")

    numeric_checks = {
        "users.user_id":       users["user_id"],
        "products.product_id": products["product_id"],
        "products.base_price": products["base_price"],
        "orders.order_id":     orders["order_id"],
        "orders.user_id":      orders["user_id"],
        "orders.product_id":   orders["product_id"],
        "orders.quantity":     orders["quantity"],
    }
    for col_name, series in numeric_checks.items():
        if pd.api.types.is_numeric_dtype(series):
            nulls = series.isna().sum()
            status = "OK" if nulls == 0 else f"WARNING: {nulls} nulls"
            print(f"       {col_name:<25s} {str(series.dtype):<10s}  {status}")
        else:
            print(f"       {col_name:<25s} NOT NUMERIC ({series.dtype}) -- needs fix!")
            issues_found += 1

    # ---- 4c. Referential integrity ----
    print("\n  [4c] Referential integrity checks ...")

    valid_user_ids    = set(users["user_id"])
    valid_product_ids = set(products["product_id"])

    orphan_users = ~orders["user_id"].isin(valid_user_ids)
    orphan_prods = ~orders["product_id"].isin(valid_product_ids)

    num_orphan_u = orphan_users.sum()
    num_orphan_p = orphan_prods.sum()

    if num_orphan_u:
        print(f"       WARNING: {num_orphan_u} orders reference non-existent user_ids.")
        issues_found += num_orphan_u
    else:
        print(f"       orders.user_id    -> all {len(orders):,} rows match users.csv  OK")

    if num_orphan_p:
        print(f"       WARNING: {num_orphan_p} orders reference non-existent product_ids.")
        issues_found += num_orphan_p
    else:
        print(f"       orders.product_id -> all {len(orders):,} rows match products.csv  OK")

    # ---- 4d. Validate order_stage values ----
    print("\n  [4d] Validating order_stage values ...")
    valid_stages = {"Visited", "Added to Cart", "Checkout Initiated", "Purchased"}
    actual_stages = set(orders["order_stage"].unique())
    unknown = actual_stages - valid_stages
    if unknown:
        print(f"       WARNING: Unknown order stages found: {unknown}")
        issues_found += len(unknown)
    else:
        print(f"       All stages valid: {sorted(actual_stages)}")

    # ---- Summary ----
    if issues_found == 0:
        print(f"\n  --> Validation PASSED. No structural issues detected.")
    else:
        print(f"\n  --> Validation completed with {issues_found} issue(s) flagged above.")

    return users, products, orders


# ===================================================================
#  MAIN PIPELINE
# ===================================================================
def main():
    print("=" * 64)
    print("  ANTIGRAVITY CORP -- DATA CLEANING PIPELINE (Phase 2)")
    print("=" * 64)

    # Load
    users, products, orders = load_raw_files()

    # Step 1: Deduplicate
    orders = step1_deduplicate_orders(orders)

    # Step 2: Fix quantities
    orders = step2_fix_quantities(orders)

    # Step 3: Outlier detection
    products = step3_outlier_detection(products)

    # Step 4: Validate
    users, products, orders = step4_validate(users, products, orders)

    # ---------------------------------------------------------------
    #  EXPORT CLEANED FILES
    # ---------------------------------------------------------------
    banner(5, "EXPORTING CLEANED FILES")

    users.to_csv(OUT_USERS, index=False)
    products.to_csv(OUT_PRODUCTS, index=False)
    orders.to_csv(OUT_ORDERS, index=False)

    print(f"  cleaned_users.csv    -> {len(users):,} rows  "
          f"({os.path.getsize(OUT_USERS):,} bytes)")
    print(f"  cleaned_products.csv -> {len(products):,} rows  "
          f"({os.path.getsize(OUT_PRODUCTS):,} bytes)")
    print(f"  cleaned_orders.csv   -> {len(orders):,} rows  "
          f"({os.path.getsize(OUT_ORDERS):,} bytes)")

    # ---------------------------------------------------------------
    #  FINAL SUMMARY
    # ---------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print("  CLEANING PIPELINE COMPLETE -- FINAL SUMMARY")
    print(f"{'=' * 64}")
    print(f"  Files written to: {BASE_DIR}")
    print(f"    - cleaned_users.csv")
    print(f"    - cleaned_products.csv")
    print(f"    - cleaned_orders.csv")
    print(f"\n  Quick stats on cleaned orders:")
    print(f"    Total orders       : {len(orders):,}")
    print(f"    Unique users       : {orders['user_id'].nunique()}")
    print(f"    Unique products    : {orders['product_id'].nunique()}")
    print(f"    Date range         : "
          f"{orders['order_date'].min()} -> {orders['order_date'].max()}")

    print(f"\n  Funnel after cleaning:")
    for stage in ["Visited", "Added to Cart", "Checkout Initiated", "Purchased"]:
        count = (orders["order_stage"] == stage).sum()
        pct   = count / len(orders) * 100
        print(f"    {stage:<22s} {count:>5,}  ({pct:5.1f}%)")

    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
