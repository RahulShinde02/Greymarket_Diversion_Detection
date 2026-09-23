# Greymarket Diversion Detection 

This is a data analytics portfolio project that simulates a common supply chain issue: **product diversion (or grey market dumping)**. 

In many distribution-heavy industries like Cosmetics, Alcohol & Tobacoo, Automotive Parts and Tires, Electronics and pharmaceuticals, sales representatives might push bulk orders to hit month-end targets. Distributors then dump this excess stock into unauthorized territories, causing legitimate distributors in those victim territories to lose sales. 

While a production-grade system would require complex machine learning, geographic network modeling, and massive historical datasets, this project is a **simplified proof-of-concept**. It uses synthetic data generation and basic heuristic rules to demonstrate how one might approach identifying these anomalies.

## 🎯 Project Purpose

This project was built to practice and demonstrate:
- **Synthetic Data Generation:** Writing scripts to simulate realistic business transactions and injecting controlled anomalies (fraud scenarios).
- **Data Manipulation with Pandas:** Using window functions, group-bys, and temporal joins to analyze time-series sales data.
- **Heuristic Rule Design:** Building basic logic to flag suspicious patterns (e.g., matching a massive sales spike in one territory with an uncharacteristic drop in another).

## 🛠️ How It Works

The project is split into two main parts:

### 1. Data Generator (`Generate_data.py`)
Since real-world pharmaceutical sales data is confidential, this script generates a mock dataset. 
- It creates a baseline of normal daily sales for various stockists.
- It intentionally injects "diversion events" (anomalies) where a large bulk order is placed, followed by a simulated drop in sales in a neighboring territory 14 days later.
- It outputs transaction logs and a "ground truth" file so we can later check if our detection logic actually worked.

### 2. Detection Script (`GMP_Rank.py`)
This script tries to find the injected anomalies using basic rules. It looks for a "Two-Sided" pattern:
- **Source Anomaly:** Did a stockist order 3x their normal volume, specifically near the end of the month or on an off-cycle day?
- **Victim Corroboration:** Within a 14-day window, did another stockist in a different territory experience a sudden drop to less than 55% of their normal volume for the exact same product?

If both conditions are met, the script flags the transaction as high risk.

### 3. Interactive Visualization (Power BI)
To make the findings actionable for supply chain managers, the project includes a Power BI dashboard (`Power_BI_Dashboard/Sales_Dashboard_(Suspected Dumpings).pbix`).
- Ingests the flagged transactions and risk scores data.
- Allows users to visually investigate volume spikes, trace the relationship between source territories and victim territories, and filter by high-risk alerts.
- A PDF export of the dashboard is also included for quick previewing.

## 📂 Repository Structure

```
├── config.json                 # Basic parameters for the data simulation (days, random seed, etc.)
├── Generate_data.py            # Script to generate synthetic sales data
├── GMP_Rank.py                 # Script containing the heuristic rules to detect diversion
├── Power_BI_Dashboard/         # Contains the interactive Power BI dashboard (.pbix) and PDF export
├── pyproject.toml              # Dependencies (pandas, numpy)
└── data/                       # Folder where generated CSVs are stored
```

## 🚀 Running the Code

**Prerequisites:** Python 3.12+ and `uv` package manager (or just `pip` install pandas and numpy).

1. **Install dependencies:**
   ```bash
   uv sync
   ```
2. **Generate the synthetic data:**
   ```bash
   uv run python Generate_data.py
   ```
   *(This will populate the `data/` folder with CSVs).*

3. **Run the detection script:**
   ```bash
   uv run python GMP_Rank.py
   ```
   *(This will output a ranked list of suspicious transactions and evaluate how well the rules performed against the injected ground truth).*

## ⚠️ Limitations
- **Rule-Based Logic:** The detection engine relies on hardcoded thresholds (e.g., "3x volume spike", "55% drop"). In the real world, these thresholds would be too brittle and generate either too many false positives or miss subtle anomalies.
- **Synthetic Data:** Real sales data is much noisier and influenced by seasonality, competitor actions, and stockouts, which are mostly ignored in this simulation.
- **No Geographic Context:** A real system would use a distance matrix (measuring the distance between the source and victim territories) to assess if moving the physical stock is economically viable. This simple script just assumes any territory can dump into any other. 

## 💻 Requirements
- **Python 3.14+**
- **Pandas** (Data manipulation)
- **NumPy** (Random sampling and math)
