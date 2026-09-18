"""
insight_engine.py
------------------
Generates plain-English insights strictly derived from calculated values.
No insight text in this module is ever fabricated -- every sentence is
built by formatting real numbers computed from the DataFrame, so the
narrative always matches the underlying data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from utils.helpers import format_number, format_percent, safe_div, pct_change


def top_category_insight(df: pd.DataFrame, category_col: str, value_col: str,
                          currency: bool = True) -> str:
    grouped = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    if grouped.empty:
        return ""
    top_name, top_value = grouped.index[0], grouped.iloc[0]
    total = grouped.sum()
    share = safe_div(top_value, total) * 100
    return (
        f"**{top_name}** leads in {category_col.lower()} with "
        f"{format_number(top_value, is_currency=currency)} of {value_col.lower()}, "
        f"contributing approximately {share:.1f}% of the total."
    )


def margin_insight(df: pd.DataFrame, category_col: str, revenue_col: str, profit_col: str) -> str:
    grouped = df.groupby(category_col).agg(rev=(revenue_col, "sum"), prof=(profit_col, "sum"))
    if grouped.empty:
        return ""
    grouped["margin"] = grouped["prof"] / grouped["rev"].replace(0, pd.NA) * 100
    grouped = grouped.dropna(subset=["margin"])
    if grouped.empty:
        return ""
    top_rev = grouped["rev"].idxmax()
    margin_val = grouped.loc[top_rev, "margin"]
    if margin_val < 10:
        return (
            f"**{top_rev}** generates the highest revenue but has a relatively low profit margin of "
            f"{margin_val:.1f}%, indicating that high sales volume is not translating into proportional "
            f"profitability."
        )
    return (
        f"**{top_rev}** generates the highest revenue with a healthy profit margin of {margin_val:.1f}%."
    )


def trend_insight(ts: pd.DataFrame, date_col: str, value_col: str) -> str:
    if len(ts) < 2:
        return ""
    first, last = ts[value_col].iloc[0], ts[value_col].iloc[-1]
    change = pct_change(first, last)
    direction = "increased" if change >= 0 else "decreased"
    period_start = ts[date_col].iloc[0].strftime("%b %Y")
    period_end = ts[date_col].iloc[-1].strftime("%b %Y")
    return (
        f"{value_col} {direction} by {abs(change):.1f}% between {period_start} and {period_end}."
    )


def outlier_insight(col: str, n_outliers: int, n_total: int) -> str:
    if n_outliers == 0:
        return f"No significant outliers detected in **{col}**."
    pct = safe_div(n_outliers, n_total) * 100
    return (
        f"**{col}** contains {n_outliers:,} outlier value(s) ({pct:.1f}% of records) that fall well "
        f"outside the typical range and may warrant closer review."
    )


def correlation_insight(col1: str, col2: str, corr_value: float) -> str:
    strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.4 else "weak"
    direction = "positive" if corr_value > 0 else "negative"
    return (
        f"There is a {strength} {direction} correlation ({corr_value:.2f}) between **{col1}** and **{col2}**."
    )


def customer_concentration_insight(df: pd.DataFrame, customer_col: str, value_col: str,
                                    top_n: int = 10) -> str:
    grouped = df.groupby(customer_col)[value_col].sum().sort_values(ascending=False)
    if grouped.empty:
        return ""
    total = grouped.sum()
    top_share = safe_div(grouped.head(top_n).sum(), total) * 100
    return (
        f"The top {top_n} customers contribute {top_share:.1f}% of total {value_col.lower()}, "
        f"out of {grouped.shape[0]:,} customers overall."
    )


def discount_profitability_insight(df: pd.DataFrame, discount_col: str, profit_col: str) -> Optional[str]:
    if discount_col not in df.columns or profit_col not in df.columns:
        return None
    corr = df[[discount_col, profit_col]].corr().iloc[0, 1]
    if pd.isna(corr):
        return None
    if corr < -0.2:
        return (
            f"Higher discounting is associated with lower profit (correlation of {corr:.2f} between "
            f"{discount_col} and {profit_col}), suggesting discount levels may be eroding margins."
        )
    return None
