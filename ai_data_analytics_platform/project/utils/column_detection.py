"""
column_detection.py
--------------------
Intelligent column-detection utilities.

This module inspects a pandas DataFrame and figures out, using a
combination of column-name keyword matching, dtype inspection, and
statistical properties (unique-value ratio, value ranges, etc.), what
each column probably *means* from a business-analytics point of view.

The output of `detect_roles()` is a dictionary that every other module
in the analytical engine (kpi_engine, chart_engine, insight_engine,
storytelling, ask_data, ...) relies on so that the whole application
adapts itself to whatever dataset is uploaded instead of assuming a
fixed schema.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Keyword dictionaries used for semantic matching. These are intentionally
# broad -- we match substrings (case-insensitive, punctuation/underscore
# insensitive) rather than exact column names so the engine generalises to
# datasets it has never seen before.
# ---------------------------------------------------------------------------

REVENUE_KEYWORDS = [
    "revenue", "sales", "amount", "income", "total sales", "net sales",
    "turnover", "value", "gmv", "earnings amount",
]
PROFIT_KEYWORDS = [
    "profit", "margin", "net profit", "gross profit", "earnings", "net income",
]
COST_KEYWORDS = ["cost", "expense", "expenditure", "cogs"]
QUANTITY_KEYWORDS = ["quantity", "qty", "units", "count", "volume"]
DISCOUNT_KEYWORDS = ["discount", "promo", "rebate"]
PRICE_KEYWORDS = ["price", "unit price", "rate"]
DATE_KEYWORDS = [
    "date", "order date", "transaction date", "created", "created_at",
    "timestamp", "month", "year", "period",
]
CUSTOMER_KEYWORDS = ["customer", "client", "buyer", "user", "account"]
PRODUCT_KEYWORDS = ["product", "item", "sku", "article"]
CATEGORY_KEYWORDS = ["category", "segment", "type", "class", "department", "sub-category", "subcategory"]
REGION_KEYWORDS = ["region", "zone", "territory", "area"]
GEO_KEYWORDS = ["country", "state", "city", "province", "location"]
ID_KEYWORDS = ["id", "code", "number", "no.", "no_", "_no", "identifier"]
EMPLOYEE_KEYWORDS = ["employee", "staff", "rep", "salesperson", "agent"]
SALARY_KEYWORDS = ["salary", "wage", "compensation", "pay"]
GENDER_KEYWORDS = ["gender", "sex"]


def _norm(col: str) -> str:
    """Normalise a column name for keyword matching."""
    return re.sub(r"[^a-z0-9 ]", " ", str(col).lower()).strip()


def _matches_any(col_norm: str, keywords: List[str]) -> bool:
    return any(kw in col_norm for kw in keywords)


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify every column into one of: 'numeric', 'categorical',
    'date', 'id', 'boolean', 'text'.

    This is a structural classification (based on dtype and cardinality),
    separate from the semantic role classification done in detect_roles().
    """
    types: Dict[str, str] = {}
    n_rows = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        col_norm = _norm(col)

        # Try to parse as datetime if it looks date-like and isn't numeric
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "date"
            continue

        if pd.api.types.is_bool_dtype(series):
            types[col] = "boolean"
            continue

        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique(dropna=True) / n_rows
            # High-cardinality integer columns whose names look like IDs,
            # or unnamed/index columns from CSV exports, are treated as identifiers.
            if (_matches_any(col_norm, ID_KEYWORDS) and unique_ratio > 0.5) or col_norm.startswith("unnamed") or col_norm in ("index", "id", "_id"):
                types[col] = "id"
            else:
                types[col] = "numeric"
            continue

        # Object / string columns: try a date parse first
        if _matches_any(col_norm, DATE_KEYWORDS):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.6:
                types[col] = "date"
                continue

        unique_ratio = series.nunique(dropna=True) / n_rows
        if _matches_any(col_norm, ID_KEYWORDS) and unique_ratio > 0.5:
            types[col] = "id"
        elif unique_ratio > 0.9 and n_rows > 20:
            # Almost every value is unique -> likely free text or an ID
            types[col] = "id" if series.astype(str).str.len().mean() < 20 else "text"
        else:
            types[col] = "categorical"

    return types


def detect_roles(df: pd.DataFrame, column_types: Dict[str, str] | None = None) -> Dict[str, List[str]]:
    """
    Determine the *semantic business role* of each column.

    Returns a dictionary such as:
        {
            "revenue": ["Sales"],
            "profit": ["Profit"],
            "cost": [],
            "quantity": ["Quantity"],
            "discount": ["Discount"],
            "date": ["Order_Date"],
            "customer": ["Customer_ID", "Customer_Name"],
            "product": ["Product"],
            "category": ["Category"],
            "region": ["Region"],
            "geo": ["City"],
            "id": ["Order_ID"],
            "numeric_other": [...],
            "categorical_other": [...],
        }
    """
    if column_types is None:
        column_types = detect_column_types(df)

    roles: Dict[str, List[str]] = {
        "revenue": [], "profit": [], "cost": [], "quantity": [],
        "discount": [], "price": [], "date": [], "customer": [],
        "product": [], "category": [], "region": [], "geo": [],
        "id": [], "employee": [], "salary": [], "gender": [],
        "numeric_other": [], "categorical_other": [],
    }

    for col, ctype in column_types.items():
        col_norm = _norm(col)

        if ctype == "date":
            roles["date"].append(col)
            continue

        if ctype == "id":
            roles["id"].append(col)
            if _matches_any(col_norm, CUSTOMER_KEYWORDS):
                roles["customer"].append(col)
            continue

        if ctype == "numeric":
            if _matches_any(col_norm, PROFIT_KEYWORDS):
                roles["profit"].append(col)
            elif _matches_any(col_norm, COST_KEYWORDS):
                roles["cost"].append(col)
            elif _matches_any(col_norm, DISCOUNT_KEYWORDS):
                roles["discount"].append(col)
            elif _matches_any(col_norm, PRICE_KEYWORDS):
                roles["price"].append(col)
            elif _matches_any(col_norm, QUANTITY_KEYWORDS):
                roles["quantity"].append(col)
            elif _matches_any(col_norm, SALARY_KEYWORDS):
                roles["salary"].append(col)
            elif _matches_any(col_norm, REVENUE_KEYWORDS):
                roles["revenue"].append(col)
            else:
                roles["numeric_other"].append(col)
            continue

        if ctype in ("categorical", "text", "boolean"):
            if _matches_any(col_norm, GEO_KEYWORDS):
                roles["geo"].append(col)
            elif _matches_any(col_norm, REGION_KEYWORDS):
                roles["region"].append(col)
            elif _matches_any(col_norm, CUSTOMER_KEYWORDS):
                roles["customer"].append(col)
            elif _matches_any(col_norm, PRODUCT_KEYWORDS):
                roles["product"].append(col)
            elif _matches_any(col_norm, EMPLOYEE_KEYWORDS):
                roles["employee"].append(col)
            elif _matches_any(col_norm, GENDER_KEYWORDS):
                roles["gender"].append(col)
            elif _matches_any(col_norm, CATEGORY_KEYWORDS):
                roles["category"].append(col)
            else:
                roles["categorical_other"].append(col)
            continue

    return roles


def summarize_roles(roles: Dict[str, List[str]]) -> str:
    """Human-readable one-line summary of detected roles, used in the UI."""
    parts = []
    for role, cols in roles.items():
        if cols:
            parts.append(f"{role}: {', '.join(cols)}")
    return " | ".join(parts) if parts else "No clear business columns detected."
