"""
storytelling.py
----------------
Builds the "Data Story" narrative: What happened / Why / Where / What
changed / What is concerning / What should we do -- structured like a
business analyst presenting findings to management. Every sentence is
derived from real calculated values (never fabricated).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from utils.helpers import format_number, safe_div, pct_change
from modules.forecasting import aggregate_timeseries, growth_rate


def _cap_first(text: str) -> str:
    """Capitalize only the first character, leaving the rest (including
    Markdown bold markers and proper nouns) untouched -- unlike str.capitalize()
    which incorrectly lowercases everything after the first letter."""
    return text[0].upper() + text[1:] if text else text


def _first(cols: List[str] | None) -> Optional[str]:
    return cols[0] if cols else None


def build_data_story(df: pd.DataFrame, roles: Dict[str, List[str]], kpis: List[Dict]) -> Dict[str, str]:
    story: Dict[str, str] = {}
    kpi_map = {k["label"]: k["value"] for k in kpis}

    revenue_col = _first(roles.get("revenue"))
    profit_col = _first(roles.get("profit"))
    date_col = _first(roles.get("date"))
    category_col = _first(roles.get("category")) or _first(roles.get("product")) or _first(roles.get("region"))
    region_col = _first(roles.get("region")) or _first(roles.get("geo"))
    discount_col = _first(roles.get("discount"))

    # ---- 1. What happened? -------------------------------------------------
    parts = []
    if "Total Revenue" in kpi_map:
        parts.append(f"total revenue reached {format_number(kpi_map['Total Revenue'], is_currency=True)}")
    if "Total Profit" in kpi_map:
        parts.append(f"total profit stood at {format_number(kpi_map['Total Profit'], is_currency=True)}")
    if "Total Orders" in kpi_map:
        parts.append(f"{format_number(kpi_map['Total Orders'])} orders were recorded")
    story["what_happened"] = (
        ("Across the dataset, " + ", ".join(parts) + ".") if parts
        else f"The dataset contains {len(df):,} records across {len(df.columns)} columns."
    )

    # ---- 2. Why did it happen? ----------------------------------------------
    why_bits = []
    if category_col and revenue_col:
        grouped = df.groupby(category_col)[revenue_col].sum().sort_values(ascending=False)
        if not grouped.empty:
            share = safe_div(grouped.iloc[0], grouped.sum()) * 100
            why_bits.append(
                f"**{grouped.index[0]}** was the single largest driver, contributing {share:.1f}% of "
                f"{revenue_col.lower()}"
            )
    if discount_col and profit_col and discount_col in df.columns and profit_col in df.columns:
        corr = df[[discount_col, profit_col]].corr().iloc[0, 1]
        if pd.notna(corr) and corr < -0.2:
            why_bits.append(
                f"increased discounting shows a negative relationship with profit (correlation {corr:.2f}), "
                f"suggesting margin pressure from promotions"
            )
    story["why"] = (_cap_first(". ".join(why_bits)) + "." if why_bits
                     else "No single dominant driver could be isolated from the available columns.")

    # ---- 3. Where did it happen? --------------------------------------------
    where_bits = []
    if region_col and revenue_col:
        grouped = df.groupby(region_col)[revenue_col].sum().sort_values(ascending=False)
        if not grouped.empty:
            top, bottom = grouped.index[0], grouped.index[-1]
            where_bits.append(
                f"**{top}** was the strongest-performing {region_col.lower()}, while **{bottom}** "
                f"lagged behind"
            )
    cust_col = _first(roles.get("customer"))
    if cust_col and revenue_col:
        grouped = df.groupby(cust_col)[revenue_col].sum().sort_values(ascending=False)
        if len(grouped) >= 10:
            share = safe_div(grouped.head(10).sum(), grouped.sum()) * 100
            where_bits.append(f"the top 10 customers accounted for {share:.1f}% of revenue")
    story["where"] = (_cap_first(". ".join(where_bits)) + "." if where_bits
                       else "The dataset does not contain enough geographic or customer detail to localize performance.")

    # ---- 4. What changed? -----------------------------------------------------
    change_bits = []
    if date_col and revenue_col:
        ts = aggregate_timeseries(df, date_col, revenue_col, freq="ME")
        g = growth_rate(ts, revenue_col)
        if g is not None:
            direction = "grew" if g >= 0 else "declined"
            change_bits.append(f"{revenue_col} {direction} {abs(g):.1f}% in the most recent period versus the one before it")
        if len(ts) >= 2:
            overall = pct_change(ts[revenue_col].iloc[0], ts[revenue_col].iloc[-1])
            change_bits.append(
                f"over the full time range, {revenue_col.lower()} moved {overall:+.1f}% from "
                f"{ts[date_col].iloc[0].strftime('%b %Y')} to {ts[date_col].iloc[-1].strftime('%b %Y')}"
            )
    story["what_changed"] = (_cap_first(". ".join(change_bits)) + "." if change_bits
                              else "No date column was detected, so trend-over-time analysis isn't available.")

    # ---- 5. What is concerning? ------------------------------------------------
    concern_bits = []
    if category_col and revenue_col and profit_col:
        grouped = df.groupby(category_col).agg(rev=(revenue_col, "sum"), prof=(profit_col, "sum"))
        grouped["margin"] = grouped["prof"] / grouped["rev"].replace(0, pd.NA) * 100
        low_margin = grouped[grouped["margin"] < 5].dropna()
        if not low_margin.empty:
            concern_bits.append(
                f"{len(low_margin)} {category_col.lower()}(ies), including **{low_margin['margin'].idxmin()}**, "
                f"operate at a profit margin below 5%"
            )
        losses = grouped[grouped["prof"] < 0]
        if not losses.empty:
            concern_bits.append(f"{len(losses)} {category_col.lower()}(ies) are currently operating at a net loss")
    if discount_col and discount_col in df.columns:
        max_discount = df[discount_col].max()
        if pd.notna(max_discount) and max_discount > (df[discount_col].mean() * 3 if df[discount_col].mean() else 0):
            concern_bits.append("some transactions carry unusually high discount levels compared to the average")
    story["concerns"] = (_cap_first(". ".join(concern_bits)) + "." if concern_bits
                          else "No major red flags were identified from the available metrics.")

    # ---- 6. What should we do? ------------------------------------------------
    recs = []
    if concern_bits:
        recs.append("Review pricing and discount policy in the lowest-margin categories to protect profitability.")
    if where_bits:
        recs.append("Investigate what the top-performing region/segment is doing well and replicate it elsewhere.")
    if change_bits and any("declined" in b for b in change_bits):
        recs.append("Prioritize a root-cause review of the recent decline before the next reporting period.")
    if not recs:
        recs.append("Continue monitoring key metrics; no urgent corrective action is indicated by this dataset.")
    story["recommendations"] = " ".join(f"- {r}" for r in recs)

    return story
