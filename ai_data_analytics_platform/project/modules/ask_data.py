"""
ask_data.py
-----------
"Ask Your Data" natural-language query interface.

Calculations are ALWAYS performed with pandas first. If an LLM API key is
configured (see modules/llm_client.py), the LLM is only used to rephrase
the already-computed numeric answer in a more natural sentence -- it is
never allowed to invent numbers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from utils.helpers import format_number, safe_div


def _first(cols: List[str]) -> Optional[str]:
    return cols[0] if cols else None


def answer_question(query: str, df: pd.DataFrame, roles: Dict[str, List[str]]) -> str:
    """
    Rule-based question answering. Matches common analytical question
    patterns and computes the answer directly from the DataFrame.
    """
    q = query.lower().strip()

    revenue_col = _first(roles.get("revenue", []))
    profit_col = _first(roles.get("profit", []))
    quantity_col = _first(roles.get("quantity", []))
    date_col = _first(roles.get("date", []))
    region_col = _first(roles.get("region") or roles.get("geo") or [])
    category_col = _first(roles.get("category") or roles.get("product") or [])
    customer_col = _first(roles.get("customer", []))
    product_col = _first(roles.get("product", []))
    discount_col = _first(roles.get("discount", []))

    metric_col = revenue_col or profit_col or quantity_col

    # --- "top N products/customers/categories" ---------------------------------
    m = re.search(r"top\s*(\d+)?", q)
    if m and any(w in q for w in ["top", "best", "highest"]):
        n = int(m.group(1)) if m.group(1) else 10
        target_col = None
        if "product" in q and product_col:
            target_col = product_col
        elif "customer" in q and customer_col:
            target_col = customer_col
        elif "region" in q and region_col:
            target_col = region_col
        elif "categor" in q and category_col:
            target_col = category_col
        if target_col and metric_col:
            grouped = df.groupby(target_col)[metric_col].sum().sort_values(ascending=False).head(n)
            lines = [f"{i+1}. {name} — {format_number(val, is_currency=(metric_col==revenue_col or metric_col==profit_col))}"
                     for i, (name, val) in enumerate(grouped.items())]
            return f"Top {n} by {metric_col} ({target_col}):\n" + "\n".join(lines)

    # --- "which region/category/product generated highest profit/revenue" -----
    if any(w in q for w in ["which region", "which country", "which city"]) and region_col:
        target_metric = profit_col if "profit" in q else (revenue_col if "revenue" in q or "sales" in q else metric_col)
        if target_metric:
            grouped = df.groupby(region_col)[target_metric].sum().sort_values(ascending=False)
            if not grouped.empty:
                name, val = grouped.index[0], grouped.iloc[0]
                share = safe_div(val, grouped.sum()) * 100
                return (f"**{name}** generated the highest {target_metric.lower()} of "
                        f"{format_number(val, is_currency=True)}, contributing {share:.1f}% of the total.")

    if any(w in q for w in ["which categor", "which product"]) and category_col:
        target_metric = profit_col if "profit" in q else (revenue_col if "revenue" in q or "sales" in q else metric_col)
        if target_metric:
            grouped = df.groupby(category_col)[target_metric].sum().sort_values(ascending=False)
            if not grouped.empty:
                name, val = grouped.index[0], grouped.iloc[0]
                return (f"**{name}** has the highest {target_metric.lower()} at "
                        f"{format_number(val, is_currency=True)}.")

    # --- profit margin by category ---------------------------------------------
    if "margin" in q and category_col and revenue_col and profit_col:
        grouped = df.groupby(category_col).agg(rev=(revenue_col, "sum"), prof=(profit_col, "sum"))
        grouped["margin"] = grouped["prof"] / grouped["rev"].replace(0, pd.NA) * 100
        grouped = grouped.dropna(subset=["margin"]).sort_values("margin", ascending=False)
        if not grouped.empty:
            name = grouped.index[0]
            return f"**{name}** has the highest profit margin at {grouped.loc[name,'margin']:.1f}%."

    # --- "who are the top customers" --------------------------------------------
    if "customer" in q and customer_col and metric_col:
        grouped = df.groupby(customer_col)[metric_col].sum().sort_values(ascending=False).head(5)
        lines = [f"{name} — {format_number(val, is_currency=True)}" for name, val in grouped.items()]
        return "Top customers:\n" + "\n".join(lines)

    # --- underperforming / worst-performing --------------------------------------
    if any(w in q for w in ["worst", "poor", "underperform", "lowest", "declin", "risk"]):
        target_col = region_col or category_col or product_col
        if target_col and metric_col:
            grouped = df.groupby(target_col)[metric_col].sum().sort_values(ascending=True).head(5)
            lines = [f"{name} — {format_number(val, is_currency=True)}" for name, val in grouped.items()]
            return f"Lowest-performing {target_col.lower()}(ies) by {metric_col}:\n" + "\n".join(lines)

    # --- total / sum questions -----------------------------------------------------
    if any(w in q for w in ["total revenue", "total sales"]) and revenue_col:
        return f"Total {revenue_col} is {format_number(df[revenue_col].sum(), is_currency=True)}."
    if "total profit" in q and profit_col:
        return f"Total {profit_col} is {format_number(df[profit_col].sum(), is_currency=True)}."
    if any(w in q for w in ["total quantity", "units sold"]) and quantity_col:
        return f"Total {quantity_col} is {format_number(df[quantity_col].sum())}."
    if "average discount" in q and discount_col:
        return f"Average {discount_col} is {format_number(df[discount_col].mean())}."
    if any(w in q for w in ["how many customer", "number of customer"]) and customer_col:
        return f"There are {df[customer_col].nunique():,} unique customers in this dataset."
    if any(w in q for w in ["how many order", "number of order"]):
        order_id = _first([c for c in roles.get("id", []) if "order" in c.lower()])
        n = df[order_id].nunique() if order_id else len(df)
        return f"There are {n:,} orders in this dataset."

    # --- trend / seasonality --------------------------------------------------
    if any(w in q for w in ["trend", "seasonal", "growth", "month with the highest"]) and date_col and metric_col:
        ts = df[[date_col, metric_col]].dropna().set_index(date_col).resample("ME")[metric_col].sum()
        if not ts.empty:
            best_month = ts.idxmax()
            return (f"The highest-performing month was **{best_month.strftime('%B %Y')}** with "
                    f"{format_number(ts.max(), is_currency=True)} in {metric_col.lower()}.")

    return (
        "I couldn't map that question to a specific calculation with the columns available. "
        "Try asking about totals, top/bottom performers, margins, trends, or customer counts — "
        "for example: \"Which region generated the highest profit?\" or \"Show me the top 10 products.\""
    )
