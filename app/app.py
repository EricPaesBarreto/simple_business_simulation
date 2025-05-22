# save as app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
import json
import os
from io import BytesIO

plt.ioff()

# ==== Session State Setup ====
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
if "df" not in st.session_state:
    st.session_state.df = None

# ==== Functions ====

def calculate_tax(income, avg_transaction_value, tax_brackets):
    if income <= tax_brackets['low']['threshold']:
        return income * tax_brackets['low']['rate']
    elif income <= tax_brackets['mid']['threshold']:
        return income * tax_brackets['mid']['rate']
    else:
        return income * tax_brackets['high']['rate']

def generate_dataset(start, end, seed, avg_transaction_value, initial_transactions, final_transactions,
                     costs_config, event_dips, seasonal_strength, initial_staff, inflation_trend,
                     tax_brackets, max_staff=30):
    np.random.seed(seed)
    data = []
    current = datetime(start[0], start[1], 1)
    end_dt = datetime(end[0], end[1], 1)

    total_months = (end_dt.year - current.year) * 12 + (end_dt.month - current.month) + 1
    staff_count = initial_staff

    for i in range(total_months):
        ym = (current.year, current.month)
        date_label = current.strftime('%Y/%m')

        trend_fraction = i / (total_months - 1)
        base_transactions = initial_transactions + trend_fraction * (final_transactions - initial_transactions)

        seasonal_multiplier = 1 + seasonal_strength * math.sin(2 * math.pi * (current.month - 1) / 12)
        dip = event_dips.get(ym, 1.0)
        staff_multiplier = 0.98 + 0.01 * staff_count

        transactions = int(np.random.normal(base_transactions * seasonal_multiplier * dip * staff_multiplier, 100))
        revenue = transactions * avg_transaction_value

        inflation_multiplier = inflation_trend[0] + trend_fraction * (inflation_trend[1] - inflation_trend[0])
        inflation_multiplier += np.random.normal(0, 0.005)

        base_salary = costs_config['staff'] / initial_staff
        low_salary = base_salary - 200
        mid_salary = base_salary
        high_salary = base_salary + 200

        num_low = staff_count // 3
        num_high = staff_count // 3
        num_mid = staff_count - num_low - num_high

        salaries = (
            num_low * low_salary +
            num_mid * mid_salary +
            num_high * high_salary
        ) * inflation_multiplier

        other_costs = {
            key: np.random.normal(val * (1 + 0.05 * math.sin(2 * math.pi * (current.month - 1) / 12)), val * 0.1)
            for key, val in costs_config.items() if key != 'staff'
        }

        tax = calculate_tax(revenue, avg_transaction_value, tax_brackets)
        total_costs = sum(other_costs.values()) + salaries + tax
        net_profit = revenue - total_costs

        if net_profit < 0 and staff_count > 1:
            staff_count -= 1
        elif net_profit > 2000 and staff_count < max_staff:
            staff_count += 1

        data.append({
            "date": date_label,
            "transactions": transactions,
            "revenue": revenue,
            **other_costs,
            "staff_costs": salaries,
            "staff_count": staff_count,
            "taxes": tax,
            "inflation": inflation_multiplier,
            "total_costs": total_costs,
            "net_profit": net_profit
        })

        current += relativedelta(months=1)

    return pd.DataFrame(data)

# ==== Streamlit UI ====

st.title("Cost & Profit Simulation Dashboard")

uploaded_json = st.sidebar.file_uploader("Load JSON Config", type=["json"])

json_data = {}
if uploaded_json:
    try:
        json_data = json.load(uploaded_json)
    except Exception as e:
        st.warning(f"Failed to load uploaded JSON file: {e}")
elif os.path.exists("initial_conditions/initial_conditions.json"):
    try:
        with open("initial_conditions/initial_conditions.json") as f:
            json_data = json.load(f)
    except Exception as e:
        st.warning(f"Failed to read initial_conditions.json: {e}")

with st.sidebar:
    st.header("Configuration")
    seed = st.number_input("Random Seed", value=json_data.get("seed", 42))
    start_year = st.number_input("Start Year", min_value=2000, max_value=2100, value=json_data.get("start_year", 2021))
    start_month = st.number_input("Start Month", min_value=1, max_value=12, value=json_data.get("start_month", 3))
    end_year = st.number_input("End Year", min_value=2000, max_value=2100, value=json_data.get("end_year", 2025))
    end_month = st.number_input("End Month", min_value=1, max_value=12, value=json_data.get("end_month", 5))
    avg_transaction_value = st.number_input("Average Transaction Value ($)", value=json_data.get("avg_transaction_value", 25))
    initial_transactions = st.number_input("Initial Monthly Transactions", value=json_data.get("initial_transactions", 7000))
    final_transactions = st.number_input("Final Monthly Transactions", value=json_data.get("final_transactions", 9100))
    seasonal_strength = st.slider("Seasonal Effect Strength", min_value=0.0, max_value=1.0, value=json_data.get("seasonal_strength", 0.2), step=0.01)
    initial_staff = st.number_input("Initial Staff", min_value=1, max_value=30, value=json_data.get("initial_staff", 5))

    st.subheader("Inflation Multiplier Trend (e.g. 1 = no change, 1.18 = 18% increase)")
    inflation_start = st.number_input("Initial Inflation Multiplier", value=json_data.get("inflation_start", 1.0))
    inflation_end = st.number_input("Final Inflation Multiplier", value=json_data.get("inflation_end", 1.18))

    st.subheader("Average Monthly Costs")
    upkeep = st.number_input("Upkeep", value=json_data.get("upkeep", 1000))
    staff = st.number_input("Staff (Base for Initial Staff)", value=json_data.get("staff", 5000))
    operating_fees = st.number_input("Operating Fees", value=json_data.get("operating_fees", 63000))
    insurance = st.number_input("Insurance", value=json_data.get("insurance", 15000))

    st.subheader("Event Dips")
    dip_str = st.text_area("Event Dips (e.g., 2020-04:0.5)", value="2020-03:0.7\n2020-04:0.6\n2021-01:0.8")

if st.button("Generate Simulation"):
    st.session_state.simulation_run = True

    costs_config = {
        "upkeep": upkeep,
        "staff": staff,
        "operating_fees": operating_fees,
        "insurance": insurance
    }

    event_dips = {}
    for line in dip_str.strip().splitlines():
        try:
            k, v = line.split(":")
            y, m = map(int, k.split("-"))
            event_dips[(y, m)] = float(v)
        except:
            st.warning(f"Invalid dip format: {line}")

    tax_brackets = json_data.get("tax_brackets", {
        "low": {"threshold": 2500000, "rate": 0.10},
        "mid": {"threshold": 7500000, "rate": 0.17},
        "high": {"rate": 0.21}
    })

    df = generate_dataset(
        (start_year, start_month), (end_year, end_month), seed,
        avg_transaction_value, initial_transactions, final_transactions,
        costs_config, event_dips, seasonal_strength, initial_staff,
        (inflation_start, inflation_end), tax_brackets
    )

    st.session_state.df = df

# === Show Output ===

if st.session_state.simulation_run and st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("Simulation Results")
    st.dataframe(df)

    selected_columns = st.multiselect("Select Additional Graph Columns", df.columns.tolist(), default=["transactions"])

    num_graphs = len(selected_columns) + 1
    fig, ax = plt.subplots(num_graphs, 1, figsize=(14, 5 + 3 * num_graphs))

    if num_graphs == 1:
        ax = [ax]

    x = range(len(df['date']))

    ax[0].fill_between(x, df['revenue'], label='Revenue', color='blue', alpha=0.3)
    ax[0].fill_between(x, df['total_costs'], label='Total Costs', color='yellow', alpha=0.3)
    ax[0].fill_between(x, df['net_profit'], label='Net Profit', color='green', alpha=0.3)
    ax[0].set_title("Revenue, Costs and Net Profit")
    ax[0].legend()
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(df['date'], rotation=45, fontsize=8)

    for i, col in enumerate(selected_columns):
        ax[i + 1].plot(x, df[col], label=col, color='purple')
        ax[i + 1].set_title(f"Monthly {col.title()}")
        ax[i + 1].legend()
        ax[i + 1].set_xticks(x)
        ax[i + 1].set_xticklabels(df['date'], rotation=45, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("📊 Download Graphs", data=buf.getvalue(), file_name="simulation_graphs.png", mime="image/png")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📅 Download CSV", data=csv, file_name="simulation.csv", mime='text/csv')

# === Save default config on first run ===

if not os.path.exists("initial_conditions"):
    os.makedirs("initial_conditions")

if not os.path.exists("initial_conditions/initial_conditions.json"):
    default_template = {
        "seed": 42,
        "start_year": 2021,
        "start_month": 3,
        "end_year": 2025,
        "end_month": 5,
        "avg_transaction_value": 25,
        "initial_transactions": 7000,
        "final_transactions": 9100,
        "seasonal_strength": 0.2,
        "initial_staff": 5,
        "upkeep": 1000,
        "staff": 5000,
        "operating_fees": 63000,
        "insurance": 15000,
        "inflation_start": 1.0,
        "inflation_end": 1.18,
        "tax_brackets": {
            "low": {"threshold": 2500000, "rate": 0.10},
            "mid": {"threshold": 7500000, "rate": 0.17},
            "high": {"rate": 0.21}
        }
    }
    with open("initial_conditions/initial_conditions.json", "w") as f:
        json.dump(default_template, f, indent=4)

# === Download current settings ===
current_settings = {
    "seed": seed,
    "start_year": start_year,
    "start_month": start_month,
    "end_year": end_year,
    "end_month": end_month,
    "avg_transaction_value": avg_transaction_value,
    "initial_transactions": initial_transactions,
    "final_transactions": final_transactions,
    "seasonal_strength": seasonal_strength,
    "initial_staff": initial_staff,
    "inflation_start": inflation_start,
    "inflation_end": inflation_end,
    "upkeep": upkeep,
    "staff": staff,
    "operating_fees": operating_fees,
    "insurance": insurance,
    "tax_brackets": json_data.get("tax_brackets", {
        "low": {"threshold": 2500000, "rate": 0.10},
        "mid": {"threshold": 7500000, "rate": 0.17},
        "high": {"rate": 0.21}
    })
}

json_str = json.dumps(current_settings, indent=4)
st.download_button(
    label="📥 Download Current Settings (JSON)",
    data=json_str,
    file_name="simulation_settings.json",
    mime="application/json"
)
