# %%
# Pharma Gray Market Poaching & Diversion Detection
# Script to generate synthetic sales data and simulate gray market dumping behavior

import os
import sys
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# %%
# ----------------- CONFIG & SETUP -----------------

# default settings in case config file is missing from repo or not provided in Arg
DEFAULT_SEED = 42
DEFAULT_DAYS = 365
DEFAULT_GROWTH = 0.065         # ~6.5% standard YoY growth for Indian pharma market (Demnd growth is simulated, price hike is not simulated)
DEFAULT_NUM_EVENTS = 18        # number of diversion scenarios to inject
DEFAULT_START_DATE = datetime(year=2025, month=1, day=1, tzinfo=timezone(offset=timedelta(hours=5, minutes=30))) # IST setup

# util to handle date formats
def parse_start_date(dt_val, fallback_dt):
    try:
        return datetime.fromisoformat(str(dt_val)).replace(tzinfo=fallback_dt.tzinfo)
    except (ValueError, TypeError):
        return fallback_dt

def load_config():
    cfg_path = "config.json"
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        cfg_path = sys.argv[1]

    if not os.path.exists(cfg_path):
        print(f"config file '{cfg_path}' not found. Using defaults.")
        return {}

    print(f"Loading params from: {cfg_path}")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        print(f"Error reading '{cfg_path}': {err}. Using defaults.")
        return {}

cfg = load_config()

# fallback to defaults safely
try: seed = int(cfg.get("seed", DEFAULT_SEED))
except: seed = DEFAULT_SEED

try: num_days = int(cfg.get("days", DEFAULT_DAYS))
except: num_days = DEFAULT_DAYS

try: growth_rate = float(cfg.get("growth_rate", DEFAULT_GROWTH))
except: growth_rate = DEFAULT_GROWTH

try: num_events = int(cfg.get("n_events", DEFAULT_NUM_EVENTS))
except: num_events = DEFAULT_NUM_EVENTS

start_date = parse_start_date(cfg.get("start_date"), DEFAULT_START_DATE)

print("Running with parameters:")
print(f"  - seed:        {seed}")
print(f"  - days:        {num_days}")
print(f"  - growth_rate: {growth_rate}")
print(f"  - n_events:    {num_events}")
print(f"  - start_date:  {start_date.strftime('%Y-%m-%d %z')}\n")

output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# locking the seed so project guide gets the exact same data on review
rng = np.random.default_rng(seed)
np.random.seed(seed)

min_stockists = 3
max_stockists = 6

# %%
# ----------------- MAHARASHTRA TERRITORIES & QUOTAS -----------------
# monthly quota targets for the sales reps in these regions

TERRITORIES = [
    {"id": "MH_MUM_01", "name": "Mumbai-South", "monthly_quota": 1400},
    {"id": "MH_MUM_02", "name": "Mumbai-Suburbs", "monthly_quota": 1600},
    {"id": "MH_THN_01", "name": "Thane-Urban", "monthly_quota": 1500},
    {"id": "MH_NVM_01", "name": "Navi-Mumbai", "monthly_quota": 1300},
    {"id": "MH_PUN_01", "name": "Pune-City", "monthly_quota": 1550},
    {"id": "MH_PUN_02", "name": "Pune-PCMC", "monthly_quota": 1450},
    {"id": "MH_PUN_03", "name": "Pune-Rural", "monthly_quota": 900},
    {"id": "MH_NAG_01", "name": "Nagpur-Urban", "monthly_quota": 1250},
    {"id": "MH_NAG_02", "name": "Nagpur-Rural", "monthly_quota": 850},
    {"id": "MH_NSK_01", "name": "Nashik-City", "monthly_quota": 1200},
    {"id": "MH_NSK_02", "name": "Nashik-Rural", "monthly_quota": 800},
    {"id": "MH_AUR_01", "name": "Aurangabad-Hub", "monthly_quota": 1100},
    {"id": "MH_AUR_02", "name": "Aurangabad-Rural", "monthly_quota": 750},
    {"id": "MH_KOL_01", "name": "Kolhapur", "monthly_quota": 1050},
    {"id": "MH_SOL_01", "name": "Solapur", "monthly_quota": 1000},
    {"id": "MH_AMR_01", "name": "Amravati", "monthly_quota": 950},
    {"id": "MH_NAN_01", "name": "Nanded", "monthly_quota": 900},
    {"id": "MH_JAL_01", "name": "Jalgaon", "monthly_quota": 850},
    {"id": "MH_AKO_01", "name": "Akola", "monthly_quota": 800},
    {"id": "MH_LAT_01", "name": "Latur", "monthly_quota": 700},
    {"id": "MH_DHU_01", "name": "Dhule", "monthly_quota": 650},
    {"id": "MH_AHM_01", "name": "Ahmednagar", "monthly_quota": 1150},
    {"id": "MH_CND_01", "name": "Chandrapur", "monthly_quota": 600},
    {"id": "MH_PAR_01", "name": "Parbhani", "monthly_quota": 550},
    {"id": "MH_JLN_01", "name": "Jalna", "monthly_quota": 500},
    {"id": "MH_BHU_01", "name": "Bhusawal", "monthly_quota": 450},
    {"id": "MH_SAT_01", "name": "Satara", "monthly_quota": 700},
    {"id": "MH_BED_01", "name": "Beed", "monthly_quota": 600},
    {"id": "MH_YAV_01", "name": "Yavatmal", "monthly_quota": 550},
    {"id": "MH_BRM_01", "name": "Baramati", "monthly_quota": 750},
]

# %%
# ----------------- PHARMA PRODUCTS PORTFOLIO -----------------
# core SKUs with PTR (Price to Retailer) and target rev share

PRODUCTS = [
    {"sku": "SKU_HV_HD_01", "name": "Pulmoshield 50mg", "base_price": 520.0, "revenue_share": 0.40, "margin":0.40},
    {"sku": "SKU_HV_HD_02", "name": "Aerovent 30s", "base_price": 300.0, "revenue_share": 0.30 ,"margin":0.30},
    {"sku": "SKU_HV_LD_03", "name": "Respicure 10mg", "base_price": 950.0, "revenue_share": 0.075,"margin":0.30},
    {"sku": "SKU_MD_MD_04", "name": "Bronchodil 10s", "base_price": 85.0, "revenue_share": 0.075,"margin":0.328},
    {"sku": "SKU_LV_LD_05", "name": "Nebusal 24pk", "base_price": 28.0, "revenue_share": 0.0375,"margin":0.20},
    {"sku": "SKU_LV_MD_06", "name": "Clinisept 5L", "base_price": 45.0, "revenue_share": 0.0375,"margin":0.28},
    {"sku": "SKU_MD_HD_07", "name": "Aerohale MDI", "base_price": 220.0, "revenue_share": 0.075 ,"margin":0.25},
]

# adjusting volume weights so cheap doesn't mess up the overall share
implied_vols = [p["revenue_share"] / p["base_price"] for p in PRODUCTS]
tot_implied_vol = sum(implied_vols)
sku_weights = [v / tot_implied_vol for v in implied_vols]

product_skus = [p["sku"] for p in PRODUCTS]
price_map = {p["sku"]: p["base_price"] for p in PRODUCTS}
name_map = {p["sku"]: p["name"] for p in PRODUCTS}

# fast-moving items (>30% rev) are the ones that actually get dumped in gray markets
fast_moving_skus = [p["sku"] for p in PRODUCTS if p["revenue_share"] >= 0.30]

# %%
# ----------------- SETUP STOCKISTS -----------------

REGION_MAP = {
    "MUM": "Mumbai",
    "THN": "Thane",
    "NVM": "Navi Mumbai",
    "PUN": "Pune",
    "NAG": "Nagpur",
    "NSK": "Nashik",
    "AUR": "Chhatrapati Sambhajinagar",
    "KOL": "Kolhapur",
    "SOL": "Solapur",
    "AMR": "Amravati",
    "NAN": "Nanded",
    "JAL": "Jalgaon",
    "AKO": "Akola",
    "LAT": "Latur",
    "DHU": "Dhule",
    "AHM": "Ahmednagar",
    "CND": "Chandrapur",
    "PAR": "Parbhani",
    "JLN": "Jalna",
    "BHU": "Bhusawal",
    "SAT": "Satara",
    "BED": "Beed",
    "YAV": "Yavatmal",
    "BRM": "Baramati"
}

def get_region_code(terr_id: str) -> str:
    code = terr_id.split("_")[1]
    return REGION_MAP.get(code, code)

def setup_stockists(terr_list, random_gen):
    s_list = []
    for terr in terr_list:
        n_stock = int(random_gen.integers(min_stockists, max_stockists + 1))
        # dirichlet to make some stockists big players and some small
        shares = random_gen.dirichlet(np.ones(n_stock) * 2.0)

        for idx, share in enumerate(shares):
            s_id = f"{terr['id']}_S{idx + 1}"
            s_list.append({
                "id": s_id,
                "name": f"{terr['name']} Pharma Dist {idx + 1}",
                "territory_id": terr["id"],
                "territory_name": terr["name"],
                "region": get_region_code(terr["id"]),
                "monthly_quota": max(5, terr["monthly_quota"] * share),
            })
    return s_list

STOCKISTS = setup_stockists(TERRITORIES, rng)

terr_stockist_map = {}
for s in STOCKISTS:
    terr_stockist_map.setdefault(s["territory_id"], []).append(s)

print(f"Created {len(STOCKISTS)} stockists across {len(TERRITORIES)} Maharashtra territories")

# %%
# ----------------- WEEKLY Order SCHEDULE -----------------
# mapping ordering days for stockists (Mon=0 to Sat=5, Sunday closed)

order_days = [0, 1, 2, 3, 4, 5]
order_day_weights = [0.25, 0.15, 0.2, 0.1, 0.1, 0.2] # heavier on mondays and saturdays

chosen_days = np.random.choice(order_days, size=len(STOCKISTS), p=order_day_weights)
stockist_schedule = {s["id"]: day for s, day in zip(STOCKISTS, chosen_days)}

# %%
# ----------------- CLEAN SALES BASELINE -----------------

sales_tracker = []
order_id_seq = 100001

for day_idx in range(num_days):
    curr_date = start_date + timedelta(days=day_idx)
    curr_wday = curr_date.weekday()
    # compounding growth factor
    trend_factor = 1 + (growth_rate * (day_idx / 365.0))

    for stockist in STOCKISTS:
        st_id = stockist["id"]

        if curr_wday != stockist_schedule[st_id]:
            continue

        exp_vol = stockist["monthly_quota"] * trend_factor
        # standard 8% variation for natural noise
        act_vol = max(1, int(np.random.normal(loc=exp_vol, scale=exp_vol * 0.08)))

        order_id = f"ORD-{order_id_seq}"
        order_id_seq += 1

        chosen_items = np.random.choice(product_skus, size=act_vol, p=sku_weights)
        skus, counts = np.unique(chosen_items, return_counts=True)

        for sku, qty in zip(skus, counts):
            ptr_price = price_map[sku]
            sales_tracker.append({
                "order_id": order_id,
                "date": curr_date.strftime("%Y-%m-%d"),
                "weekday": curr_wday,
                "territory_id": stockist["territory_id"],
                "stockist_id": st_id,
                "stockist_name": stockist["name"],
                "sku": sku,
                "product_name": name_map[sku],
                "unit_price": ptr_price,
                "units_sold": int(qty),
                "total_value": float(ptr_price * qty),
            })

sales_df = pd.DataFrame(sales_tracker)
sales_df["is_diverted"] = False
sales_df["poaching_event_id"] = None

print(f"Generated {len(sales_df)} genuine transaction records")

# %%
# ----------------- DUMPING / POACHING INJECTION -----------------
# The classic month-end target scenario:
# 1. Rep forces bulk order into 1-3 friendly stockists in source territory.
#    That extra inventory gets sold to traders in a different district.
# 2. The authorized stockist in the victim district sees a dip in chemist orders
#    because the market is flooded with the dumped stock.

max_day = num_days - 1
ground_truth = []
poaching_events = []
new_entries = []

valid_terrs = [t["id"] for t in TERRITORIES if len(terr_stockist_map[t["id"]]) >= 1]

# only stockists doing decent volume can absorb a dumping hit
vol_threshold = np.median([s["monthly_quota"] for s in STOCKISTS]) * 0.8
major_stockists = [s for s in STOCKISTS if s["monthly_quota"] >= vol_threshold]

terr_major_map = {}
for s in major_stockists:
    terr_major_map.setdefault(s["territory_id"], []).append(s)

valid_terrs = [t for t in valid_terrs if terr_major_map.get(t)]

for event_num in range(num_events):
    event_id = f"DUMP-{event_num + 1:03d}"

    src_terr = rng.choice(valid_terrs)
    helper_pool = terr_major_map[src_terr]

    n_helpers = min(int(rng.integers(1, 4)), len(helper_pool))
    chosen_idx = rng.choice(len(helper_pool), size=n_helpers, replace=False)
    helpers = [helper_pool[i] for i in chosen_idx]

    # find the victim stockist in another territory
    impacted_stockist = major_stockists[int(rng.integers(0, len(major_stockists)))]
    while impacted_stockist["territory_id"] == src_terr:
        impacted_stockist = major_stockists[int(rng.integers(0, len(major_stockists)))]

    vic_id = impacted_stockist["id"]
    vic_terr = impacted_stockist["territory_id"]

    sku = rng.choice(fast_moving_skus)
    price = price_map[sku]

    # month end pressure timing
    period_start = int(rng.integers(0, max(1, max_day - 35)))
    window_dates = [start_date + timedelta(days=d)
                    for d in range(period_start + 24, min(period_start + 30, max_day) + 1)]

    if not window_dates: continue

    spike_multiplier = rng.uniform(0.6, 1.4)
    dip_multiplier = rng.uniform(0.5, 0.9)

    # 1. SOURCE DUMP
    for helper in helpers:
        h_id = helper["id"]
        inj_date = window_dates[int(rng.integers(0, len(window_dates)))]
        inj_date_str = inj_date.strftime("%Y-%m-%d")

        hist = sales_df[(sales_df.stockist_id == h_id) & (sales_df.sku == sku)]
        avg_qty = hist["units_sold"].tail(6).mean() if len(hist) else helper["monthly_quota"] * 0.3
        dump_qty = max(3, int(avg_qty * spike_multiplier))

        existing = sales_df[(sales_df.stockist_id == h_id) &
                            (sales_df.date == inj_date_str) &
                            (sales_df.sku == sku)]

        if len(existing):
            row_idx = existing.index[0]
            sales_df.loc[row_idx, "units_sold"] += dump_qty
            sales_df.loc[row_idx, "total_value"] = sales_df.loc[row_idx, "units_sold"] * price
            sales_df.loc[row_idx, "is_diverted"] = True
            sales_df.loc[row_idx, "poaching_event_id"] = event_id

            ground_truth.append({
                "event_id": event_id, "role": "source", "order_id": sales_df.loc[row_idx, "order_id"],
                "stockist_id": h_id, "territory_id": helper["territory_id"], "sku": sku,
                "date": inj_date_str, "units_delta": dump_qty,
            })
        else:
            tot_qty = max(3, int(avg_qty)) + dump_qty
            new_entry = {
                "order_id": f"ORD-DUMP-{event_id}-{h_id}",
                "date": inj_date_str,
                "weekday": inj_date.weekday(),
                "territory_id": helper["territory_id"],
                "stockist_id": h_id,
                "stockist_name": helper["name"],
                "sku": sku,
                "product_name": name_map[sku],
                "unit_price": price,
                "units_sold": tot_qty,
                "total_value": tot_qty * price,
                "is_diverted": True,
                "poaching_event_id": event_id,
            }
            new_entries.append(new_entry)

            ground_truth.append({
                "event_id": event_id, "role": "source", "order_id": new_entry["order_id"],
                "stockist_id": h_id, "territory_id": helper["territory_id"], "sku": sku,
                "date": inj_date_str, "units_delta": dump_qty,
            })

    # 2. VICTIM DIP (Secondary market effect)
    victim_hist = sales_df[(sales_df.stockist_id == vic_id) &
                          (sales_df.sku == sku) &
                          (sales_df.date >= window_dates[0].strftime("%Y-%m-%d"))].sort_values("date")

    for v_idx in victim_hist.index[:2]:
        current_vol = sales_df.loc[v_idx, "units_sold"]
        slump_vol = max(1, int(current_vol * (1 - dip_multiplier)))

        sales_df.loc[v_idx, "units_sold"] = slump_vol
        sales_df.loc[v_idx, "total_value"] = slump_vol * price
        sales_df.loc[v_idx, "is_diverted"] = True
        sales_df.loc[v_idx, "poaching_event_id"] = event_id

        ground_truth.append({
            "event_id": event_id, "role": "victim", "order_id": sales_df.loc[v_idx, "order_id"],
            "stockist_id": vic_id, "territory_id": vic_terr, "sku": sku,
            "date": sales_df.loc[v_idx, "date"], "units_delta": slump_vol - current_vol,
        })

    poaching_events.append({
        "event_id": event_id,
        "source_territory": src_terr,
        "victim_territory": vic_terr,
        "sku": sku,
        "window_start": window_dates[0].strftime("%Y-%m-%d"),
        "window_end": window_dates[-1].strftime("%Y-%m-%d"),
        "helper_stockists": ", ".join(h["id"] for h in helpers),
        "impacted_stockist": vic_id,
    })

if new_entries:
    sales_df = pd.concat([sales_df, pd.DataFrame(new_entries)], ignore_index=True)

gt_df = pd.DataFrame(ground_truth)
events_df = pd.DataFrame(poaching_events)

print(f"Injected {len(events_df)} dumping scenarios "
      f"({(sales_df['is_diverted']).sum()} flagged records out of {len(sales_df)} total)")

# %%
# ----------------- EXPORTING -----------------
sales_df.to_csv(os.path.join(output_folder, "sales_transactions.csv"), index=False)
gt_df.to_csv(os.path.join(output_folder, "ground_truth.csv"), index=False)
events_df.to_csv(os.path.join(output_folder, "poaching_events.csv"), index=False)

pd.DataFrame(STOCKISTS).to_csv(os.path.join(output_folder, "stockists_master.csv"), index=False)
pd.DataFrame(TERRITORIES).to_csv(os.path.join(output_folder, "territories_master.csv"), index=False)
pd.DataFrame(PRODUCTS).to_csv(os.path.join(output_folder, "products_master.csv"), index=False)

print("Saved all masters and transaction files to data/ folder.")

# %%
# ----------------- SANITY CHECKS -----------------
total_revenue = sales_df["total_value"].sum()

prod_breakdown = (sales_df.groupby("product_name")["total_value"].sum()
                .reset_index().sort_values("total_value", ascending=False))
prod_breakdown["pct_of_total"] = prod_breakdown["total_value"] / total_revenue
print("\nProduct wise revenue split:")
print(prod_breakdown.to_string(index=False))

fraud_stats = sales_df.groupby("is_diverted")["total_value"].sum()
print("\nGenuine vs Diverted Revenue:")
print(fraud_stats)

print(f"\nTotal Stockists: {len(STOCKISTS)}")
print(f"Total Territories (MH): {len(TERRITORIES)}")
print(f"Total Dumping Events Simulated: {len(events_df)}")
