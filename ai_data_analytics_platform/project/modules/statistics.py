"""
statistics.py
-------------
Statistical analysis helpers: descriptive statistics, outlier detection
(IQR and Z-score), correlation analysis, and Pareto (80/20) analysis.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def descriptive_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    stats = df[numeric_cols].describe().T
    stats["variance"] = df[numeric_cols].var()
    stats["skew"] = df[numeric_cols].skew()
    stats = stats.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std Dev", "min": "Min",
        "25%": "P25", "50%": "Median", "75%": "P75", "max": "Max",
        "variance": "Variance", "skew": "Skewness",
    })
    return stats.round(3)


def outliers_iqr(series: pd.Series) -> pd.Series:
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series([], dtype=series.dtype)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]


def outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    s = series.dropna()
    if s.std(ddof=0) == 0 or len(s) < 3:
        return pd.Series([], dtype=series.dtype)
    z = (s - s.mean()) / s.std(ddof=0)
    return series[z.abs() > threshold]


def outlier_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        iqr_out = outliers_iqr(df[col])
        z_out = outliers_zscore(df[col])
        rows.append({
            "Column": col,
            "IQR Outliers": len(iqr_out),
            "Z-score Outliers (|z|>3)": len(z_out),
            "IQR Outlier %": round(len(iqr_out) / max(len(df), 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr().round(3)


def top_correlations(corr: pd.DataFrame, top_n: int = 5) -> List[Tuple[str, str, float]]:
    if corr.empty:
        return []
    pairs = []
    cols = corr.columns
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = corr.loc[c1, c2]
            if pd.notna(val):
                pairs.append((c1, c2, float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_n]


def pareto_analysis(df: pd.DataFrame, category_col: str, value_col: str) -> pd.DataFrame:
    """80/20 analysis: cumulative contribution of categories to total value."""
    grouped = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    total = grouped.sum()
    cum_pct = (grouped.cumsum() / total * 100) if total else grouped.cumsum()
    result = pd.DataFrame({
        category_col: grouped.index,
        value_col: grouped.values,
        "Cumulative %": cum_pct.values.round(2),
    })
    return result


def rfm_analysis(df: pd.DataFrame, customer_col: str, date_col: str, monetary_col: str) -> pd.DataFrame:
    """Simple Recency, Frequency, Monetary segmentation."""
    work = df[[customer_col, date_col, monetary_col]].dropna()
    if work.empty:
        return pd.DataFrame()
    snapshot_date = work[date_col].max() + pd.Timedelta(days=1)
    rfm = work.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(date_col, "count"),
        Monetary=(monetary_col, "sum"),
    ).reset_index()

    def _score(series: pd.Series, ascending: bool) -> pd.Series:
        try:
            return pd.qcut(series.rank(method="first"), 4, labels=[1, 2, 3, 4] if ascending else [4, 3, 2, 1]).astype(int)
        except ValueError:
            return pd.Series([2] * len(series), index=series.index)

    rfm["R_Score"] = _score(rfm["Recency"], ascending=False)
    rfm["F_Score"] = _score(rfm["Frequency"], ascending=True)
    rfm["M_Score"] = _score(rfm["Monetary"], ascending=True)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def segment(score):
        if score >= 10:
            return "Champions"
        if score >= 8:
            return "Loyal Customers"
        if score >= 6:
            return "Potential Loyalists"
        if score >= 4:
            return "At Risk"
        return "Lost / Low Value"

    rfm["Segment"] = rfm["RFM_Score"].apply(segment)
    return rfm.sort_values("RFM_Score", ascending=False)
