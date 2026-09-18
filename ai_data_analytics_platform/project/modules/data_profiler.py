"""
data_profiler.py
-----------------
Automatic data-understanding / profiling module. Produces the "Dataset
Overview" numbers shown right after upload, and a per-column profile used
throughout the rest of the app.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from utils.column_detection import detect_column_types


def profile_dataset(df: pd.DataFrame) -> Dict:
    """Return a dictionary of high-level dataset statistics."""
    n_rows, n_cols = df.shape
    column_types = detect_column_types(df)

    type_counts = {"numeric": 0, "categorical": 0, "date": 0, "id": 0, "boolean": 0, "text": 0}
    for t in column_types.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    total_cells = max(n_rows * n_cols, 1)
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round(missing_cells / total_cells * 100, 2)

    duplicate_rows = int(df.duplicated().sum())

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numerical_columns": type_counts["numeric"],
        "categorical_columns": type_counts["categorical"],
        "date_columns": type_counts["date"],
        "id_columns": type_counts["id"],
        "boolean_columns": type_counts["boolean"],
        "text_columns": type_counts["text"],
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(safe_pct(duplicate_rows, n_rows), 2),
        "column_types": column_types,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
    }


def safe_pct(part, whole) -> float:
    return (part / whole * 100) if whole else 0.0


def column_level_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column profiling table (dtype, missing %, unique, etc.)."""
    column_types = detect_column_types(df)
    rows: List[Dict] = []
    n = len(df)

    for col in df.columns:
        series = df[col]
        missing = series.isna().sum()
        row = {
            "Column": col,
            "Detected Type": column_types.get(col, "unknown"),
            "Pandas dtype": str(series.dtype),
            "Missing": int(missing),
            "Missing %": round(safe_pct(missing, n), 2),
            "Unique Values": int(series.nunique(dropna=True)),
        }
        if column_types.get(col) == "numeric":
            desc = series.describe()
            row.update({
                "Min": round(desc.get("min", np.nan), 2) if n else np.nan,
                "Max": round(desc.get("max", np.nan), 2) if n else np.nan,
                "Mean": round(desc.get("mean", np.nan), 2) if n else np.nan,
            })
        rows.append(row)

    return pd.DataFrame(rows)


def detect_constant_and_empty_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    constant_cols = [
        c for c in df.columns
        if c not in empty_cols and df[c].nunique(dropna=True) <= 1
    ]
    return {"empty_columns": empty_cols, "constant_columns": constant_cols}


def detect_high_cardinality_columns(df: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    column_types = detect_column_types(df)
    n = max(len(df), 1)
    result = []
    for col, ctype in column_types.items():
        if ctype == "categorical":
            ratio = df[col].nunique(dropna=True) / n
            if ratio > threshold:
                result.append(col)
    return result
