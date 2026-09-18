"""
kpi_engine.py
-------------
Automatically detects and calculates business KPIs from the dataset,
based on the semantic roles identified by utils.column_detection.

KPIs are only generated when the underlying data actually supports them
("Only show KPIs for which relevant data exists").
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from utils.helpers import safe_div


def _primary(cols: List[str]):
    return cols[0] if cols else None


def generate_kpis(df: pd.DataFrame, roles: Dict[str, List[str]]) -> List[Dict]:
    """
    Return a list of KPI dicts: {"label", "value", "is_currency", "help"}.
    Order is meaningful (used directly for KPI-card layout).
    """
    kpis: List[Dict] = []

    revenue_col = _primary(roles.get("revenue", []))
    profit_col = _primary(roles.get("profit", []))
    cost_col = _primary(roles.get("cost", []))
    quantity_col = _primary(roles.get("quantity", []))
    discount_col = _primary(roles.get("discount", []))
    price_col = _primary(roles.get("price", []))
    customer_cols = roles.get("customer", [])
    product_cols = roles.get("product", [])
    id_cols = roles.get("id", [])

    total_revenue = None
    if revenue_col:
        total_revenue = float(df[revenue_col].sum())
        kpis.append({"label": "Total Revenue", "value": total_revenue, "is_currency": True,
                      "help": f"Sum of {revenue_col}"})

    total_profit = None
    if profit_col:
        total_profit = float(df[profit_col].sum())
        kpis.append({"label": "Total Profit", "value": total_profit, "is_currency": True,
                      "help": f"Sum of {profit_col}"})

    if total_revenue is not None and total_profit is not None:
        margin = safe_div(total_profit, total_revenue) * 100
        kpis.append({"label": "Profit Margin", "value": margin, "is_currency": False,
                      "is_percent": True, "help": "Total Profit / Total Revenue"})

    if cost_col:
        total_cost = float(df[cost_col].sum())
        kpis.append({"label": "Total Cost", "value": total_cost, "is_currency": True,
                      "help": f"Sum of {cost_col}"})

    if quantity_col:
        total_qty = float(df[quantity_col].sum())
        kpis.append({"label": "Total Quantity Sold", "value": total_qty, "is_currency": False,
                      "help": f"Sum of {quantity_col}"})

    # Number of orders: use an ID column that looks like an order/transaction id,
    # falling back to row count.
    order_id_col = next((c for c in id_cols if "order" in c.lower() or "transaction" in c.lower()), None)
    n_orders = df[order_id_col].nunique() if order_id_col else len(df)
    kpis.append({"label": "Total Orders", "value": float(n_orders), "is_currency": False,
                  "help": f"Distinct {order_id_col}" if order_id_col else "Row count"})

    if revenue_col:
        aov = safe_div(total_revenue, n_orders)
        kpis.append({"label": "Average Order Value", "value": aov, "is_currency": True,
                      "help": "Total Revenue / Total Orders"})

    if customer_cols:
        cust_col = _primary([c for c in customer_cols if "id" in c.lower()] or customer_cols)
        n_customers = int(df[cust_col].nunique(dropna=True))
        kpis.append({"label": "Number of Customers", "value": float(n_customers), "is_currency": False,
                      "help": f"Distinct {cust_col}"})

    if product_cols:
        prod_col = _primary(product_cols)
        n_products = int(df[prod_col].nunique(dropna=True))
        kpis.append({"label": "Number of Products", "value": float(n_products), "is_currency": False,
                      "help": f"Distinct {prod_col}"})

    if discount_col:
        avg_discount = float(df[discount_col].mean())
        # Heuristic: if values look like fractions (<=1) treat as %, else as raw average
        display_val = avg_discount * 100 if df[discount_col].max() <= 1.5 else avg_discount
        kpis.append({"label": "Average Discount", "value": display_val, "is_currency": False,
                      "is_percent": df[discount_col].max() <= 1.5, "help": f"Mean of {discount_col}"})

    if price_col:
        avg_price = float(df[price_col].mean())
        kpis.append({"label": "Average Price", "value": avg_price, "is_currency": True,
                      "help": f"Mean of {price_col}"})

    return kpis


def kpi_summary_dict(kpis: List[Dict]) -> Dict[str, float]:
    """Convenience: map KPI label -> raw numeric value (used by storytelling/ask_data)."""
    return {k["label"]: k["value"] for k in kpis}
