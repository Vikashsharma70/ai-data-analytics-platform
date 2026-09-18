"""
helpers.py
----------
Small, generic helper functions shared across modules: number formatting,
safe division, percentage formatting, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def format_number(value: float, is_currency: bool = False, currency_symbol: str = "$") -> str:
    """Format a number using K / M / B suffixes for readability in KPI cards."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"

    prefix = currency_symbol if is_currency else ""
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000:
        return f"{sign}{prefix}{abs_value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{sign}{prefix}{abs_value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{sign}{prefix}{abs_value / 1_000:.2f}K"
    if isinstance(value, (int, np.integer)) or float(value).is_integer():
        return f"{sign}{prefix}{abs_value:,.0f}"
    return f"{sign}{prefix}{abs_value:,.2f}"


def format_percent(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}%"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator in (0, None) or (isinstance(denominator, float) and np.isnan(denominator)):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def pct_change(old: float, new: float) -> float:
    """Percentage change from old to new. Returns 0 if old is 0/NaN."""
    return safe_div(new - old, abs(old), default=0.0) * 100


def truncate_label(label: str, max_len: int = 22) -> str:
    label = str(label)
    return label if len(label) <= max_len else label[: max_len - 1] + "…"
