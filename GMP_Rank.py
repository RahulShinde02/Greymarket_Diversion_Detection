# Heuristic Pipeline with Downstream (Secondary Market) Dip Validation
#
# Logic:
# Sometimes a stockist dumps inventory into another district for quick profit
# via unauthorized bulk orders, or they share stock with friends/family who
# aren't official distributors. Other times, company reps just dump stock
# to hit their own sales targets.
#
# This is a major issue because it feeds the company bad ground-truth data,
# leading to wrong decisions and flawed strategies.
#
# We won't see these black-market sales directly in the data. But we will
# see the fallout: the authorized stockist in the receiving district takes
# a hit on their regular orders because their local market is already
# saturated with the dumped goods.
#
# Detection: Source spike + Victim dip (same product, 14-day window) = High probability of diversion.

import os
import pandas as pd
import numpy as np

# ----------------- PATHS -----------------
data_dir = "data"
sales_file = os.path.join(data_dir, "sales_transactions.csv")
truth_file = os.path.join(data_dir, "ground_truth.csv")
output_file = os.path.join(data_dir, "paired_risk_scores.csv")

FORWARD_LOOK_DAYS = 14       # 2-week window to look for the secondary market dip
DIP_THRESHOLD = 0.55         # Drops below 55% of baseline are suspicious

# ----------------- DATA INGESTION -----------------
print("-" * 60)
print("Executing Two-Sided Validation Pipeline")
print("-" * 60)
sales_data = pd.read_csv(sales_file, parse_dates=["date"])
print(f"Loaded {len(sales_data):,} rows | {sales_data['stockist_id'].nunique()} Stockists | {sales_data['sku'].nunique()} SKUs\n")

# ----------------- SOURCE-SIDE SUSPICION METRICS -----------------
# 1. Baseline median sales for each stockist
median_qty = sales_data.groupby(["stockist_id", "sku"])["units_sold"].transform("median")
sales_data["median_vol"] = median_qty.replace(0, 1)
sales_data["vol_ratio"] = (sales_data["units_sold"] / sales_data["median_vol"]).round(2)

# 2. Irregular order booking days (off-cycle POB)
standard_day = sales_data.groupby("stockist_id")["weekday"].transform(lambda x: x.mode().iloc[0])
sales_data["is_irregular_day"] = (sales_data["weekday"] != standard_day).astype(int)
sales_data["day_weight"] = 1 + 2.0 * sales_data["is_irregular_day"]

# 3. High-velocity SKUs (the usual suspects for gray market dumping)
top_skus = sales_data.groupby("sku")["total_value"].sum().nlargest(2).index
sales_data["is_fast_mover"] = sales_data["sku"].isin(top_skus).astype(int)
sales_data["product_weight"] = 1 + 1.0 * sales_data["is_fast_mover"]

# 4. Month-end target pressure (every medical rep's nightmare)
days_remaining = sales_data["date"].dt.days_in_month - sales_data["date"].dt.day
sales_data["is_month_end"] = (days_remaining < 6).astype(int)
sales_data["month_weight"] = 1 + 0.1 * sales_data["is_month_end"]

# Initial standalone risk score
sales_data["source_risk"] = (
    sales_data["vol_ratio"]
    * sales_data["day_weight"]
    * sales_data["product_weight"]
    * sales_data["month_weight"]
).round(2)

# ----------------- CROSS-DISTRICT CORRELATION -----------------
# Flagging dips where authorized stockists lose secondary sales to dumped stock
sales_data["is_dip"] = sales_data["vol_ratio"] <= DIP_THRESHOLD
slumps_df = sales_data[sales_data["is_dip"]][["order_id", "date", "territory_id", "stockist_id", "sku", "units_sold", "median_vol"]].copy()
slumps_df["lost_units"] = (slumps_df["median_vol"] - slumps_df["units_sold"]).clip(lower=0)

dips_by_sku = {sku: group.sort_values("date") for sku, group in slumps_df.groupby("sku")}
validation_data = []

for idx, row in sales_data.iterrows():
    # Ignore normal or low volumes for source evaluation
    if row["vol_ratio"] < 1.3 or row["sku"] not in dips_by_sku:
        validation_data.append({
            "validated_dip": 0, "total_lost_units": 0.0,
            "victim_stockist_id": None, "victim_territory": None
        })
        continue

    item = row["sku"]
    src_date = row["date"]
    src_region = row["territory_id"]
    sku_slumps = dips_by_sku[item]

    # Look for a corresponding volume drop in a DIFFERENT territory
    window_close = src_date + pd.Timedelta(days=FORWARD_LOOK_DAYS)
    valid_matches = sku_slumps[
        (sku_slumps["territory_id"] != src_region) &
        (sku_slumps["date"] >= src_date) &
        (sku_slumps["date"] <= window_close)
    ]

    if len(valid_matches) > 0:
        total_deficit = valid_matches["lost_units"].sum()
        worst_hit = valid_matches.sort_values("lost_units", ascending=False).iloc[0]
        validation_data.append({
            "validated_dip": 1,
            "total_lost_units": round(total_deficit, 2),
            "victim_stockist_id": worst_hit["stockist_id"],
            "victim_territory": worst_hit["territory_id"]
        })
    else:
        validation_data.append({
            "validated_dip": 0, "total_lost_units": 0.0,
            "victim_stockist_id": None, "victim_territory": None
        })

evidence_df = pd.DataFrame(validation_data)
sales_data = pd.concat([sales_data, evidence_df], axis=1)

# Combined scoring: Boost score if confirmed by a secondary market drop
sales_data["correlation_multiplier"] = np.where(sales_data["validated_dip"] == 1, 1.8, 0.7)
sales_data["final_risk_score"] = (sales_data["source_risk"] * sales_data["correlation_multiplier"]).round(2)

suspicious_orders = sales_data.sort_values("final_risk_score", ascending=False).reset_index(drop=True)

# ----------------- EXPORTING SCORES -----------------
cols_to_save = [
    "order_id", "date", "territory_id", "stockist_id", "sku", "product_name",
    "units_sold", "vol_ratio", "is_irregular_day", "is_fast_mover",
    "validated_dip", "victim_stockist_id", "victim_territory",
    "total_lost_units", "final_risk_score"
]
suspicious_orders[cols_to_save].to_csv(output_file, index=False)
print(f"Paired risk scores saved to: {output_file}")

# ----------------- OUTPUT TOP DIVERSION ALERTS -----------------
print("\n" + "=" * 115)
print("TOP 15 CROSS-VALIDATED DIVERSION ALERTS")
print("=" * 115)
for idx, r in suspicious_orders.head(15).iterrows():
    vic_str = f"{r['victim_stockist_id']} ({r['victim_territory']})" if r["validated_dip"] else "None"
    print(
        f"#{idx + 1:<2} | Order: {r['order_id']:<28} | Stockist: {r['stockist_id']:<10} | "
        f"Vol Spike: {r['vol_ratio']:<4.1f}x | Irregular: {r['is_irregular_day']} | "
        f"Corroborated: {r['validated_dip']} -> Victim: {vic_str:<20} | SCORE: {r['final_risk_score']:<5.1f}"
    )

# ----------------- MODEL EVALUATION -----------------
if os.path.exists(truth_file):
    print("\n" + "=" * 60)
    print("CHECKING ACCURACY AGAINST GROUND TRUTH")
    print("=" * 60)
    truth_data = pd.read_csv(truth_file, parse_dates=["date"])
    actual_sources = truth_data[truth_data["role"] == "source"]
    truth_keys = set(zip(actual_sources["stockist_id"], actual_sources["sku"], actual_sources["date"].dt.strftime("%Y-%m-%d")))

    suspicious_orders["search_key"] = list(zip(suspicious_orders["stockist_id"], suspicious_orders["sku"], suspicious_orders["date"].dt.strftime("%Y-%m-%d")))
    suspicious_orders["is_actual_dumping"] = suspicious_orders["search_key"].isin(truth_keys)

    total_real_events = len(truth_keys)
    for limit in [20, 50, 100]:
        subset = suspicious_orders.head(limit)
        true_positives = subset["is_actual_dumping"].sum()
        precision = (true_positives / limit) * 100
        recall = (true_positives / total_real_events) * 100 if total_real_events else 0
        print(f"  Top-{limit:<3} -> Precision: {precision:5.1f}% ({true_positives}/{limit})   Recall: {recall:5.1f}% ({true_positives}/{total_real_events})")
