"""
app.py
------
AI-Powered Automated Data Analytics & Dashboard Web Application.

Run with:  streamlit run app.py

Workflow implemented:
Upload -> Understand -> Clean -> Analyze -> KPIs -> Charts -> Dashboard ->
Insights -> Data Story -> Ask Your Data -> Reports
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from modules.data_loader import load_uploaded_file, DataLoadError
from modules.data_profiler import profile_dataset, column_level_profile
from modules.data_cleaner import detect_issues, clean_dataset
from modules.quality_score import compute_quality_score
from modules.kpi_engine import generate_kpis, kpi_summary_dict
from modules.statistics import (
    descriptive_stats, outlier_summary, correlation_matrix, top_correlations,
    pareto_analysis, rfm_analysis,
)
from modules.forecasting import aggregate_timeseries, linear_trend_forecast, growth_rate, moving_average
from modules.storytelling import build_data_story
from modules.insight_engine import (
    top_category_insight, margin_insight, trend_insight, outlier_insight,
    correlation_insight, customer_concentration_insight, discount_profitability_insight,
)
from modules.ask_data import answer_question
from modules.report_generator import (
    dataframe_to_csv_bytes, dataframe_to_excel_bytes, build_html_report, build_insights_text_report,
)
from utils.column_detection import detect_column_types, detect_roles, summarize_roles
from utils.helpers import format_number, format_percent

# ---------------------------------------------------------------------------
# Page config & global style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    /* Tabs Styling - Fix Black on Dark Contrast */
    [data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: 2px solid rgba(148, 163, 184, 0.2) !important;
        padding-bottom: 4px !important;
    }
    [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 6px 14px !important;
        background-color: transparent !important;
    }
    [data-baseweb="tab"] *, [data-baseweb="tab"] div, [data-baseweb="tab"] p, [data-baseweb="tab"] span {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important; /* Visible light-slate gray for inactive tabs */
    }
    [data-baseweb="tab"][aria-selected="true"] *, 
    [data-baseweb="tab"][aria-selected="true"] div, 
    [data-baseweb="tab"][aria-selected="true"] p, 
    [data-baseweb="tab"][aria-selected="true"] span {
        color: #38bdf8 !important; /* Vibrant sky blue for active tab */
        font-weight: 700 !important;
    }

    /* Metric Cards - Theme Adaptive & Clean Contrast */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color, #ffffff) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
        color: var(--text-color, #475569) !important;
        opacity: 0.85;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: var(--text-color, #0f172a) !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Insight Box */
    .insight-box {
        background-color: rgba(37, 99, 235, 0.12) !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 14px 18px !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
        color: var(--text-color, #1e3a8a) !important;
    }
    .insight-box, .insight-box * {
        color: var(--text-color, #1e3a8a) !important;
    }

    /* Story Box */
    .story-box {
        background-color: var(--secondary-background-color, #ffffff) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    .story-box h4, .story-box h4 * {
        color: var(--text-color, #0f172a) !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        margin-bottom: 10px !important;
        margin-top: 0 !important;
    }
    .story-box, .story-box p, .story-box span, .story-box div, .story-box li {
        color: var(--text-color, #334155) !important;
        opacity: 0.9;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "raw_df": None, "cleaned_df": None, "file_name": None, "roles": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 AI Analytics Platform")
    st.caption("Upload a dataset to get started")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls", "tsv"])

    use_sample = st.button("▶ Use sample sales dataset", use_container_width=True)

    if use_sample:
        sample_path = Path(__file__).parent / "data" / "sample_sales_data.csv"
        if sample_path.exists():
            st.session_state.raw_df = pd.read_csv(sample_path)
            st.session_state.file_name = "sample_sales_data.csv"
        else:
            st.error("Sample dataset not found. Run data/generate_sample_data.py first.")

    if uploaded_file is not None:
        try:
            result = load_uploaded_file(uploaded_file)
            if result.is_excel and len(result.sheet_names) > 1:
                sheet_choice = st.selectbox("Select sheet", result.sheet_names)
            else:
                sheet_choice = result.sheet_names[0]
            st.session_state.raw_df = result.get(sheet_choice)
            st.session_state.file_name = uploaded_file.name
            st.session_state["file_size"] = result.file_size_bytes
            st.session_state["sheet_names"] = result.sheet_names
        except DataLoadError as e:
            st.error(str(e))

# ---------------------------------------------------------------------------
# Landing state — no data yet
# ---------------------------------------------------------------------------
if st.session_state.raw_df is None:
    st.title("📊 AI-Powered Automated Data Analytics Platform")
    st.markdown(
        "Upload an Excel or CSV file in the sidebar and this application will automatically "
        "**understand, clean, analyze**, and build a full **interactive dashboard** — complete with "
        "KPIs, charts, insights, a data story, and a natural-language Q&A interface."
    )
    st.info("👈 Upload a file, or click **Use sample sales dataset** in the sidebar to try it instantly.")
    col1, col2, col3, col4 = st.columns(4)
    for col, icon, text in zip(
        [col1, col2, col3, col4],
        ["🧹", "📈", "🧠", "💬"],
        ["Automatic cleaning", "Adaptive dashboards", "AI-generated insights", "Ask Your Data"],
    ):
        with col:
            st.markdown(f"### {icon}")
            st.write(text)
    st.stop()

# ---------------------------------------------------------------------------
# Data pipeline: understand -> clean -> analyze
# ---------------------------------------------------------------------------
raw_df = st.session_state.raw_df
file_name = st.session_state.file_name

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧹 Cleaning Options")
    remove_dupes = st.checkbox("Remove duplicate rows", value=True)
    fill_numeric_choice = st.selectbox("Fill missing numeric values with", ["median", "mean", "Don't fill"], index=0)
    fill_cat_choice = st.selectbox("Fill missing categorical values with", ["mode", "Unknown", "Don't fill"], index=0)
    convert_dates_opt = st.checkbox("Convert date-like columns to datetime", value=True)
    remove_empty_opt = st.checkbox("Remove fully empty columns", value=True)

fill_numeric = None if fill_numeric_choice == "Don't fill" else fill_numeric_choice
fill_categorical = None if fill_cat_choice == "Don't fill" else fill_cat_choice

cleaned_df, cleaning_log = clean_dataset(
    raw_df,
    remove_duplicates=remove_dupes,
    fill_numeric=fill_numeric,
    fill_categorical=fill_categorical,
    convert_dates=convert_dates_opt,
    remove_empty_columns=remove_empty_opt,
)
issues_report = detect_issues(raw_df)
profile = profile_dataset(cleaned_df)
column_types = detect_column_types(cleaned_df)
roles = detect_roles(cleaned_df, column_types)
quality_score, quality_notes = compute_quality_score(cleaned_df)

revenue_col = roles["revenue"][0] if roles["revenue"] else None
numeric_cols = [
    c for c, t in column_types.items() 
    if t == "numeric" and not str(c).lower().startswith("unnamed") and str(c).lower() not in ["id", "index", "_id", "unnamed: 0"]
]
categorical_cols = [
    c for c, t in column_types.items() 
    if t in ("categorical", "text") and not str(c).lower().startswith("unnamed") and str(c).lower() not in ["id", "index", "_id", "unnamed: 0"]
]

revenue_col = roles["revenue"][0] if roles["revenue"] else None
profit_col = roles["profit"][0] if roles["profit"] else None
quantity_col = roles["quantity"][0] if roles["quantity"] else None
date_col = roles["date"][0] if roles["date"] else None
category_col = (roles["category"] or roles["product"] or categorical_cols or [None])[0]
region_col = (roles["region"] or roles["geo"] or [c for c in categorical_cols if c != category_col] or [None])[0]
customer_col = roles["customer"][0] if roles["customer"] else None
product_col = (roles["product"] or [c for c in categorical_cols if c != category_col] or [None])[0]
discount_col = roles["discount"][0] if roles["discount"] else None

# ---------------------------------------------------------------------------
# Sidebar — dynamic filters
# ---------------------------------------------------------------------------
filtered_df = cleaned_df.copy()

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔎 Filters")

    if date_col and date_col in cleaned_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(cleaned_df[date_col]):
            cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors="coerce")
        min_d, max_d = cleaned_df[date_col].min(), cleaned_df[date_col].max()
        if pd.notna(min_d) and pd.notna(max_d) and hasattr(min_d, "date") and hasattr(max_d, "date"):
            date_range = st.date_input("Date range", value=(min_d.date(), max_d.date()),
                                        min_value=min_d.date(), max_value=max_d.date())
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = date_range
                filtered_df = filtered_df[
                    (filtered_df[date_col] >= pd.Timestamp(start)) &
                    (filtered_df[date_col] <= pd.Timestamp(end))
                ]

    filter_candidates = []
    for role_key in ["region", "geo", "category", "product"]:
        filter_candidates.extend(roles.get(role_key, []))
    filter_candidates = [c for c in dict.fromkeys(filter_candidates) if c in cleaned_df.columns][:5]

    for col in filter_candidates:
        options = sorted(cleaned_df[col].dropna().astype(str).unique().tolist())
        if 1 < len(options) <= 300:
            selected = st.multiselect(f"{col}", options, default=[])
            if selected:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    st.markdown("---")
    st.caption(f"Showing {len(filtered_df):,} of {len(cleaned_df):,} rows after filters")

if filtered_df.empty:
    st.warning("No rows match the current filters. Adjust filters in the sidebar.")
    st.stop()

kpis = generate_kpis(filtered_df, roles)

# ---------------------------------------------------------------------------
# Build the list of tabs dynamically based on what the dataset supports
# ---------------------------------------------------------------------------
tab_defs = [("🏠 Overview", "overview"), ("📊 Dashboard", "dashboard"),
            ("🔍 Data Exploration", "explore")]
if date_col and numeric_cols:
    tab_defs.append(("📈 Trends", "trends"))
if customer_col:
    tab_defs.append(("👥 Customer Analysis", "customer"))
if product_col:
    tab_defs.append(("📦 Product Analysis", "product"))
if region_col:
    tab_defs.append(("🌍 Regional Analysis", "regional"))
if len(numeric_cols) >= 1:
    tab_defs.append(("📉 Statistical Analysis", "stats"))
tab_defs.append(("💡 Insights", "insights"))
tab_defs.append(("📖 Data Story", "story"))
tab_defs.append(("🤖 Ask Your Data", "ask"))
tab_defs.append(("📥 Reports", "reports"))

tabs = st.tabs([t[0] for t in tab_defs])
tab_keys = [t[1] for t in tab_defs]


def get_tab(key):
    return tabs[tab_keys.index(key)]


# ---------------------------------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------------------------------
with get_tab("overview"):
    st.title("Dataset Overview")
    st.caption(f"File: **{file_name}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{profile['n_rows']:,}")
    c2.metric("Columns", f"{profile['n_cols']:,}")
    c3.metric("Numerical Columns", profile["numerical_columns"])
    c4.metric("Categorical Columns", profile["categorical_columns"])
    c5.metric("Date Columns", profile["date_columns"])

    c6, c7, c8 = st.columns(3)
    c6.metric("Missing Values", f"{profile['missing_pct']}%")
    c7.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")
    c8.metric("Data Quality Score", f"{quality_score}/100")

    st.markdown("#### Data Preview")
    st.dataframe(cleaned_df.head(50), use_container_width=True)

    st.markdown("#### Column Profile")
    st.dataframe(column_level_profile(cleaned_df), use_container_width=True)

    with st.expander("🧠 Detected column roles (semantic understanding)"):
        st.write(summarize_roles(roles))

    st.markdown("#### 🧹 Data Cleaning Report")
    for line in cleaning_log:
        st.markdown(f"- {line}")

    st.markdown("#### ✅ Data Quality Score Breakdown")
    st.progress(quality_score / 100)
    for note in quality_notes:
        st.markdown(f"- {note}")

# ---------------------------------------------------------------------------
# DASHBOARD TAB
# ---------------------------------------------------------------------------
with get_tab("dashboard"):
    title_guess = "Sales Performance Analytics" if revenue_col else "Data Performance Analytics"
    st.title(title_guess)
    st.caption("Interactive analysis of key metrics, trends, and segments — auto-generated from your data")

    if kpis:
        cols = st.columns(min(len(kpis), 6))
        for i, kpi in enumerate(kpis[:6]):
            with cols[i % len(cols)]:
                val = kpi["value"]
                display = f"{val:.1f}%" if kpi.get("is_percent") else format_number(val, is_currency=kpi.get("is_currency", False))
                st.metric(kpi["label"], display, help=kpi.get("help"))
        if len(kpis) > 6:
            cols2 = st.columns(min(len(kpis) - 6, 6))
            for i, kpi in enumerate(kpis[6:]):
                with cols2[i % len(cols2)]:
                    val = kpi["value"]
                    display = f"{val:.1f}%" if kpi.get("is_percent") else format_number(val, is_currency=kpi.get("is_currency", False))
                    st.metric(kpi["label"], display, help=kpi.get("help"))
    else:
        st.info("No numeric business metrics (revenue, profit, quantity...) were detected to build KPI cards.")

    st.markdown("---")
    from modules import chart_engine as ce
    import importlib
    importlib.reload(ce)

    charts_rendered = 0

    # Section 1: Line Chart & Bar Chart
    st.markdown("### 📈 Line Chart & Bar Chart (Trends & Breakdown)")
    row1 = st.columns(2)

    valid_numerics = [
        c for c in numeric_cols 
        if not str(c).lower().startswith("unnamed") and str(c).lower() not in ["id", "index", "_id", "unnamed: 0"]
    ]
    val_metric = revenue_col or profit_col or quantity_col or (valid_numerics[0] if valid_numerics else None)
    primary_cat = category_col or (categorical_cols[0] if categorical_cols else None)
    secondary_cat = region_col or (categorical_cols[1] if len(categorical_cols) > 1 else primary_cat)
    plotly_config = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

    if date_col and val_metric:
        with row1[0]:
            st.plotly_chart(
                ce.trend_line_chart(filtered_df, date_col, val_metric, title=f"📈 Line Chart: {val_metric} Trend Over Time"),
                use_container_width=True, key="proj_dashboard_trend_line", config=plotly_config
            )
            charts_rendered += 1
    elif primary_cat and val_metric:
        with row1[0]:
            st.plotly_chart(
                ce.category_bar_chart(filtered_df, primary_cat, val_metric, title=f"📊 Bar Chart: {val_metric} by {primary_cat}"),
                use_container_width=True, key="proj_dashboard_bar_sec1_left", config=plotly_config
            )
            charts_rendered += 1

    if primary_cat and val_metric:
        with row1[1]:
            st.plotly_chart(
                ce.category_bar_chart(filtered_df, primary_cat, val_metric, title=f"📊 Bar Chart: {val_metric} by {primary_cat}"),
                use_container_width=True, key="proj_dashboard_category_bar", config=plotly_config
            )
            charts_rendered += 1
    elif len(numeric_cols) >= 2:
        with row1[1]:
            st.plotly_chart(
                ce.scatter_chart(filtered_df, numeric_cols[0], numeric_cols[1], title=f"🔵 Scatter Plot: {numeric_cols[1]} vs {numeric_cols[0]}"),
                use_container_width=True, key="proj_dashboard_scatter_sec1", config=plotly_config
            )
            charts_rendered += 1

    # Section 2: Pie Chart / Donut Chart & Map Chart
    st.markdown("### 🥧 Pie Chart / Share & 🗺️ Map Chart")
    row2 = st.columns(2)

    cat_or_reg = primary_cat
    val_for_pie = val_metric
    pie_df = filtered_df
    if val_for_pie is None and cat_or_reg:
        val_for_pie = "Count"
        pie_df = filtered_df.copy()
        pie_df["Count"] = 1

    if cat_or_reg and val_for_pie:
        with row2[0]:
            pie_type = st.radio("Chart Type:", ["Pie Chart 🥧", "Donut Chart 🍩"], horizontal=True, key="proj_dash_pie_toggle")
            if "Pie Chart" in pie_type:
                st.plotly_chart(
                    ce.pie_chart(pie_df, cat_or_reg, val_for_pie, title=f"🥧 Pie Chart: {val_for_pie} Share by {cat_or_reg}"),
                    use_container_width=True, key="proj_dashboard_pie_chart", config=plotly_config
                )
            else:
                st.plotly_chart(
                    ce.donut_chart(pie_df, cat_or_reg, val_for_pie, title=f"🍩 Donut Chart: {val_for_pie} Share by {cat_or_reg}"),
                    use_container_width=True, key="proj_dashboard_donut_chart", config=plotly_config
                )
            charts_rendered += 1

    geo_col_for_map = next((c for c in roles.get("geo", []) if c in filtered_df.columns), None) or (roles.get("region") or [None])[0]
    with row2[1]:
        map_rendered = False
        if geo_col_for_map and val_for_pie:
            try:
                st.plotly_chart(
                    ce.geo_choropleth(filtered_df, geo_col_for_map, val_for_pie, title=f"🗺️ Map Chart: {val_for_pie} by {geo_col_for_map}"),
                    use_container_width=True, key="proj_dashboard_map_chart", config=plotly_config
                )
                charts_rendered += 1
                map_rendered = True
            except Exception:
                pass

        if not map_rendered:
            if secondary_cat and val_for_pie and secondary_cat != primary_cat:
                st.plotly_chart(
                    ce.category_bar_chart(filtered_df, secondary_cat, val_for_pie, title=f"📊 Secondary Breakdown: {val_for_pie} by {secondary_cat}"),
                    use_container_width=True, key="proj_dashboard_sec_cat_bar", config=plotly_config
                )
                charts_rendered += 1
            elif len(numeric_cols) >= 2:
                st.plotly_chart(
                    ce.scatter_chart(filtered_df, numeric_cols[0], numeric_cols[1], title=f"🔵 Scatter Plot: {numeric_cols[1]} vs {numeric_cols[0]}"),
                    use_container_width=True, key="proj_dashboard_scatter_r2", config=plotly_config
                )
                charts_rendered += 1
            elif primary_cat and val_for_pie:
                st.plotly_chart(
                    ce.donut_chart(pie_df, primary_cat, val_for_pie, title=f"🍩 Category Share Overview: {val_for_pie} by {primary_cat}"),
                    use_container_width=True, key="proj_dashboard_donut_alt", config=plotly_config
                )
                charts_rendered += 1

    # Section 3: Deep Dive Charts (Grouped Bar, Box Plot, Scatter Plot)
    st.markdown("### 📊 Additional Essential Dashboard Charts")
    row3 = st.columns(2)

    if revenue_col and profit_col and primary_cat:
        with row3[0]:
            st.plotly_chart(
                ce.grouped_bar_chart(filtered_df, primary_cat, [revenue_col, profit_col], title=f"📊 Grouped Bar Chart: {revenue_col} vs {profit_col} by {primary_cat}"),
                use_container_width=True, key="proj_dashboard_grouped_bar", config=plotly_config
            )
            charts_rendered += 1
    elif len(numeric_cols) >= 2:
        with row3[0]:
            st.plotly_chart(
                ce.scatter_chart(filtered_df, numeric_cols[0], numeric_cols[1], title=f"🔵 Scatter Plot: {numeric_cols[1]} vs {numeric_cols[0]}"),
                use_container_width=True, key="proj_dashboard_scatter_r3", config=plotly_config
            )
            charts_rendered += 1

    if primary_cat and val_metric:
        with row3[1]:
            st.plotly_chart(
                ce.box_chart(filtered_df, val_metric, primary_cat, title=f"📦 Box Plot: Spread of {val_metric} by {primary_cat}"),
                use_container_width=True, key="proj_dashboard_box_chart", config=plotly_config
            )
            charts_rendered += 1
    elif len(numeric_cols) >= 1:
        with row3[1]:
            st.plotly_chart(
                ce.histogram_chart(filtered_df, numeric_cols[0], title=f"📊 Distribution Histogram: {numeric_cols[0]}"),
                use_container_width=True, key="proj_dashboard_hist", config=plotly_config
            )
            charts_rendered += 1

    if charts_rendered == 0:
        st.info("Upload a dataset with at least one numeric column and one date/category column to see charts here.")

    if charts_rendered == 0:
        st.info("Upload a dataset with at least one numeric column and one date/category column to see charts here.")

# ---------------------------------------------------------------------------
# DATA EXPLORATION TAB
# ---------------------------------------------------------------------------
with get_tab("explore"):
    st.title("🔍 Data Exploration")
    st.markdown("#### What can we learn from this data?")

    questions = []
    if revenue_col:
        questions.append(f"What is the total {revenue_col.lower()}?")
    if category_col and revenue_col:
        questions.append(f"Which {category_col.lower()} generates the highest {revenue_col.lower()}?")
    if region_col and revenue_col:
        questions.append(f"Which {region_col.lower()} generates the most {revenue_col.lower()}?")
    if date_col and revenue_col:
        questions.append(f"How is {revenue_col.lower()} changing over time?")
    if customer_col:
        questions.append("Who are the most valuable customers?")
    if product_col:
        questions.append(f"Which {product_col.lower()}s sell the most?")
    if profit_col and category_col:
        questions.append(f"Which {category_col.lower()} has the highest profit margin?")
    if not questions:
        questions.append("Explore the raw data below and use filters to narrow it down.")
    for q in questions:
        st.markdown(f"- {q}")

    st.markdown("#### Filtered Data")
    st.dataframe(filtered_df, use_container_width=True, height=420)

    st.markdown("#### Explore a column")
    explore_col = st.selectbox("Choose a column", filtered_df.columns.tolist())
    ctype = column_types.get(explore_col, "unknown")
    if ctype == "numeric":
        st.plotly_chart(px.histogram(filtered_df, x=explore_col, nbins=30, template="plotly_white"),
                         use_container_width=True, key="explore_numeric_hist")
        st.dataframe(filtered_df[explore_col].describe().to_frame().T, use_container_width=True)
    else:
        counts = filtered_df[explore_col].value_counts().head(20)
        st.plotly_chart(px.bar(x=counts.values, y=counts.index.astype(str), orientation="h",
                                labels={"x": "Count", "y": explore_col}, template="plotly_white"),
                         use_container_width=True, key="explore_cat_bar")

# ---------------------------------------------------------------------------
# TRENDS TAB
# ---------------------------------------------------------------------------
if "trends" in tab_keys:
    with get_tab("trends"):
        st.title("📈 Trend Analysis")
        from modules import chart_engine as ce

        value_col = revenue_col or profit_col or quantity_col or numeric_cols[0]
        ts = aggregate_timeseries(filtered_df, date_col, value_col, freq="ME")
        ts["Moving Avg (3M)"] = moving_average(ts[value_col], window=3)

        st.plotly_chart(ce.trend_line_chart(filtered_df, date_col, value_col), use_container_width=True, key="trends_trend_line")

        g = growth_rate(ts, value_col)
        if g is not None:
            st.metric(f"Latest period-over-period growth ({value_col})", f"{g:+.1f}%")

        st.markdown("#### Forecast (Linear Trend Projection)")
        periods = st.slider("Months to forecast", 1, 12, 3)
        forecast_df = linear_trend_forecast(ts, date_col, value_col, periods=periods, freq="ME")
        if forecast_df is not None:
            st.plotly_chart(ce.multi_series_trend(forecast_df, date_col, value_col), use_container_width=True, key="trends_forecast_multi")
            st.caption("⚠ Forecast values are projections based on a simple linear trend and are clearly "
                       "distinguished from actual historical data. They should be treated as directional, not exact.")
        else:
            st.info("Not enough historical periods to build a reliable forecast (need at least 4).")

        st.markdown("#### Monthly Data Table")
        st.dataframe(ts, use_container_width=True)

# ---------------------------------------------------------------------------
# CUSTOMER ANALYSIS TAB
# ---------------------------------------------------------------------------
if "customer" in tab_keys:
    with get_tab("customer"):
        st.title("👥 Customer Analysis")
        from modules import chart_engine as ce

        n_customers = filtered_df[customer_col].nunique()
        st.metric("Number of Customers", f"{n_customers:,}")

        value_col = revenue_col or profit_col or quantity_col
        if value_col:
            st.plotly_chart(
                ce.category_bar_chart(filtered_df, customer_col, value_col, top_n=15,
                                       title=f"Top 15 Customers by {value_col}"),
                use_container_width=True, key="customer_top15_bar"
            )
            st.markdown(f"<div class='insight-box'>{customer_concentration_insight(filtered_df, customer_col, value_col)}</div>",
                        unsafe_allow_html=True)

        if roles.get("customer") and date_col and (revenue_col or profit_col):
            st.markdown("#### RFM Segmentation")
            monetary_col = revenue_col or profit_col
            rfm = rfm_analysis(filtered_df, customer_col, date_col, monetary_col)
            if not rfm.empty:
                st.plotly_chart(px.histogram(rfm, x="Segment", color="Segment", template="plotly_white",
                                              title="Customer Segments (RFM)"), use_container_width=True, key="customer_rfm_hist")
                st.dataframe(rfm.head(50), use_container_width=True)
            else:
                st.info("Not enough transaction-level detail to compute RFM segments.")

# ---------------------------------------------------------------------------
# PRODUCT ANALYSIS TAB
# ---------------------------------------------------------------------------
if "product" in tab_keys:
    with get_tab("product"):
        st.title("📦 Product Analysis")
        from modules import chart_engine as ce

        value_col = revenue_col or quantity_col
        if value_col:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(ce.category_bar_chart(filtered_df, product_col, value_col, top_n=10,
                                                        title=f"Top 10 Products by {value_col}"),
                                 use_container_width=True, key="product_top10_bar")
            with col2:
                bottom = filtered_df.groupby(product_col)[value_col].sum().sort_values().head(10).reset_index()
                st.plotly_chart(px.bar(bottom, x=value_col, y=product_col, orientation="h",
                                        template="plotly_white", title=f"Bottom 10 Products by {value_col}",
                                        color_discrete_sequence=["#EF4444"]),
                                 use_container_width=True, key="product_bottom10_bar")

        if profit_col:
            st.markdown("#### Highest / Lowest Profit Products")
            prof_grouped = filtered_df.groupby(product_col)[profit_col].sum().sort_values(ascending=False)
            col3, col4 = st.columns(2)
            col3.dataframe(prof_grouped.head(10).reset_index(), use_container_width=True)
            col4.dataframe(prof_grouped.tail(10).reset_index(), use_container_width=True)

            if revenue_col:
                st.markdown("#### Low-Margin Products (high sales, low profit)")
                merged = filtered_df.groupby(product_col).agg(
                    revenue=(revenue_col, "sum"), profit=(profit_col, "sum")
                )
                merged["margin_%"] = (merged["profit"] / merged["revenue"].replace(0, np.nan) * 100).round(1)
                low_margin = merged[merged["revenue"] > merged["revenue"].median()].sort_values("margin_%").head(10)
                st.dataframe(low_margin.reset_index(), use_container_width=True)

# ---------------------------------------------------------------------------
# REGIONAL ANALYSIS TAB
# ---------------------------------------------------------------------------
if "regional" in tab_keys:
    with get_tab("regional"):
        st.title("🌍 Regional Analysis")
        from modules import chart_engine as ce

        value_col = revenue_col or profit_col or quantity_col
        if value_col:
            st.plotly_chart(ce.category_bar_chart(filtered_df, region_col, value_col,
                                                    title=f"{value_col} by {region_col}"),
                             use_container_width=True, key="regional_bar_chart")

            geo_col_for_map = next((c for c in roles.get("geo", []) if c in filtered_df.columns), None)
            map_col = geo_col_for_map or region_col
            try:
                if map_col:
                    st.plotly_chart(ce.geo_choropleth(filtered_df, map_col, value_col),
                                     use_container_width=True, key="regional_geo_choropleth")
            except Exception:
                st.caption("Map visualization unavailable for this location column's naming format.")

            grouped = filtered_df.groupby(region_col)[value_col].sum().sort_values(ascending=False)
            if not grouped.empty:
                st.success(f"**Best performing:** {grouped.index[0]} — {format_number(grouped.iloc[0], is_currency=True)}")
                st.error(f"**Weakest performing:** {grouped.index[-1]} — {format_number(grouped.iloc[-1], is_currency=True)}")

            if profit_col and revenue_col:
                st.markdown("#### Revenue vs Profit by Region")
                comp = filtered_df.groupby(region_col).agg(Revenue=(revenue_col, "sum"), Profit=(profit_col, "sum")).reset_index()
                st.plotly_chart(px.bar(comp, x=region_col, y=["Revenue", "Profit"], barmode="group",
                                        template="plotly_white"), use_container_width=True, key="regional_rev_vs_prof_bar")

# ---------------------------------------------------------------------------
# STATISTICAL ANALYSIS TAB
# ---------------------------------------------------------------------------
if "stats" in tab_keys:
    with get_tab("stats"):
        st.title("📉 Statistical Analysis")
        from modules import chart_engine as ce

        st.markdown("#### Descriptive Statistics")
        st.dataframe(descriptive_stats(filtered_df, numeric_cols), use_container_width=True)

        st.markdown("#### Outlier Detection (IQR & Z-score)")
        st.dataframe(outlier_summary(filtered_df, numeric_cols), use_container_width=True)

        if len(numeric_cols) >= 2:
            st.markdown("#### Correlation Analysis")
            corr = correlation_matrix(filtered_df, numeric_cols)
            st.plotly_chart(ce.correlation_heatmap(corr), use_container_width=True, key="stats_corr_heatmap")
            for c1, c2, val in top_correlations(corr, top_n=3):
                st.markdown(f"<div class='insight-box'>{correlation_insight(c1, c2, val)}</div>",
                            unsafe_allow_html=True)

        if category_col and (revenue_col or quantity_col):
            st.markdown("#### Pareto (80/20) Analysis")
            value_col = revenue_col or quantity_col
            pareto = pareto_analysis(filtered_df, category_col, value_col)
            st.dataframe(pareto, use_container_width=True)
            n_80 = (pareto["Cumulative %"] <= 80).sum() + 1
            st.info(f"Approximately {n_80} out of {len(pareto)} {category_col.lower()}(ies) drive 80% of total {value_col.lower()}.")

# ---------------------------------------------------------------------------
# INSIGHTS TAB
# ---------------------------------------------------------------------------
with get_tab("insights"):
    st.title("💡 Automated Insights")
    st.caption("Every insight below is calculated directly from your data — nothing is fabricated.")

    insights: list[str] = []
    if category_col and revenue_col:
        insights.append(top_category_insight(filtered_df, category_col, revenue_col))
    if category_col and revenue_col and profit_col:
        insights.append(margin_insight(filtered_df, category_col, revenue_col, profit_col))
    if date_col and (revenue_col or profit_col):
        ts = aggregate_timeseries(filtered_df, date_col, revenue_col or profit_col, freq="ME")
        insights.append(trend_insight(ts, date_col, revenue_col or profit_col))
    if numeric_cols:
        top_outlier_col = numeric_cols[0]
        from modules.statistics import outliers_iqr
        n_out = len(outliers_iqr(filtered_df[top_outlier_col]))
        insights.append(outlier_insight(top_outlier_col, n_out, len(filtered_df)))
    if customer_col and (revenue_col or profit_col):
        insights.append(customer_concentration_insight(filtered_df, customer_col, revenue_col or profit_col))
    if discount_col and profit_col:
        di = discount_profitability_insight(filtered_df, discount_col, profit_col)
        if di:
            insights.append(di)
    if len(numeric_cols) >= 2:
        corr = correlation_matrix(filtered_df, numeric_cols)
        for c1, c2, val in top_correlations(corr, top_n=2):
            insights.append(correlation_insight(c1, c2, val))

    insights = [i for i in insights if i]
    st.session_state["latest_insights"] = insights

    if insights:
        for insight in insights:
            st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a dataset with clearer business columns (revenue, category, date, etc.) "
                "to generate richer insights.")

# ---------------------------------------------------------------------------
# DATA STORY TAB
# ---------------------------------------------------------------------------
with get_tab("story"):
    st.title("📖 Data Story")
    st.caption("Your dataset explained the way a business analyst would present it to management.")

    story = build_data_story(filtered_df, roles, kpis)
    st.session_state["latest_story"] = story

    section_titles = {
        "what_happened": "1️⃣ What happened?",
        "why": "2️⃣ Why did it happen?",
        "where": "3️⃣ Where did it happen?",
        "what_changed": "4️⃣ What changed?",
        "concerns": "5️⃣ What is concerning?",
        "recommendations": "6️⃣ What should we do?",
    }
    for key, title in section_titles.items():
        st.markdown(f"<div class='story-box'><h4>{title}</h4><p>{story.get(key, '').replace(chr(10), '<br>')}</p></div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ASK YOUR DATA TAB
# ---------------------------------------------------------------------------
with get_tab("ask"):
    st.title("🤖 Ask Your Data")
    st.caption("Ask a question in plain English. Calculations always run on the real data first.")

    example_qs = []
    if region_col and profit_col:
        example_qs.append(f"Which {region_col.lower()} generated the highest profit?")
    if product_col:
        example_qs.append("Show me the top 10 products.")
    if category_col and profit_col:
        example_qs.append(f"Which {category_col.lower()} has the highest profit margin?")
    if customer_col:
        example_qs.append("Who are the top customers?")
    if region_col:
        example_qs.append(f"Which {region_col.lower()} is performing poorly?")
    if example_qs:
        st.caption("Try: " + " · ".join(f"*{q}*" for q in example_qs[:4]))

    user_question = st.text_input("Your question", placeholder="e.g. Which region generated the highest profit?")
    if st.button("Ask", type="primary") and user_question.strip():
        answer = answer_question(user_question, filtered_df, roles)
        st.session_state.chat_history.append((user_question, answer))

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"<div class='insight-box'>{a}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# REPORTS TAB
# ---------------------------------------------------------------------------
with get_tab("reports"):
    st.title("📥 Download Reports")

    story_for_report = st.session_state.get("latest_story") or build_data_story(filtered_df, roles, kpis)
    insights_for_report = st.session_state.get("latest_insights") or []

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Cleaned Dataset")
        st.download_button("⬇ Download Cleaned Dataset (CSV)", dataframe_to_csv_bytes(cleaned_df),
                            file_name="cleaned_dataset.csv", mime="text/csv", use_container_width=True)
        st.download_button("⬇ Download Cleaned Dataset (Excel)", dataframe_to_excel_bytes(cleaned_df),
                            file_name="cleaned_dataset.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)

        st.markdown("#### Dashboard Data (filtered)")
        st.download_button("⬇ Download Filtered Data (CSV)", dataframe_to_csv_bytes(filtered_df),
                            file_name="dashboard_data.csv", mime="text/csv", use_container_width=True)

    with col2:
        st.markdown("#### Analysis Report (HTML)")
        html_report = build_html_report(file_name, profile, kpis, quality_score, quality_notes,
                                         story_for_report, insights_for_report)
        st.download_button("⬇ Download Analysis Report (HTML)", html_report.encode("utf-8"),
                            file_name="analysis_report.html", mime="text/html", use_container_width=True)

        st.markdown("#### Insights Report (TXT)")
        txt_report = build_insights_text_report(file_name, profile, kpis, quality_score, quality_notes,
                                                  story_for_report, insights_for_report)
        st.download_button("⬇ Download Insights Report (TXT)", txt_report.encode("utf-8"),
                            file_name="insights_report.txt", mime="text/plain", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Report Preview")
    with st.expander("Preview HTML report"):
        st.components.v1.html(html_report, height=500, scrolling=True)
