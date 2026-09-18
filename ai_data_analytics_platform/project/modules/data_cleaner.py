"""
data_cleaner.py
----------------
Automated data-cleaning module. Never mutates the original DataFrame --
always works on (and returns) a copy, together with a human-readable
report describing exactly what was changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.column_detection import detect_column_types
from modules.data_profiler import detect_constant_and_empty_columns, detect_high_cardinality_columns


@dataclass
class CleaningReport:
    actions: List[str] = field(default_factory=list)
    duplicates_found: int = 0
    missing_found: int = 0
    columns_converted_to_date: List[str] = field(default_factory=list)
    numeric_outliers_found: int = 0
    empty_columns_removed: List[str] = field(default_factory=list)
    constant_columns_found: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    invalid_dates_found: int = 0

    def as_list(self) -> List[str]:
        return self.actions


def detect_issues(df: pd.DataFrame) -> CleaningReport:
    """Detect (but do not fix) data-quality issues, for the report shown to the user."""
    report = CleaningReport()
    column_types = detect_column_types(df)

    report.duplicates_found = int(df.duplicated().sum())
    report.missing_found = int(df.isna().sum().sum())

    struct = detect_constant_and_empty_columns(df)
    report.empty_columns_removed = struct["empty_columns"]
    report.constant_columns_found = struct["constant_columns"]
    report.high_cardinality_columns = detect_high_cardinality_columns(df)

    # Outlier count across numeric columns using IQR
    outlier_total = 0
    for col, ctype in column_types.items():
        if ctype == "numeric":
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_total += int(((series < lower) | (series > upper)).sum())
    report.numeric_outliers_found = outlier_total

    # Candidate date columns that failed to parse cleanly
    invalid_dates = 0
    for col, ctype in column_types.items():
        if ctype == "categorical" and "date" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            invalid_dates += int(parsed.isna().sum() - df[col].isna().sum())
    report.invalid_dates_found = max(invalid_dates, 0)

    report.actions = [
        f"{report.duplicates_found:,} duplicate rows detected",
        f"{report.missing_found:,} missing values detected",
        f"{report.numeric_outliers_found:,} numerical outliers detected (IQR method)",
        f"{len(report.empty_columns_removed)} empty column(s) detected",
        f"{len(report.constant_columns_found)} constant column(s) detected",
        f"{len(report.high_cardinality_columns)} high-cardinality categorical column(s) detected",
        f"{report.invalid_dates_found:,} invalid date value(s) detected",
    ]
    return report


def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    fill_numeric: Optional[str] = "median",   # "mean", "median", or None
    fill_categorical: Optional[str] = "mode",  # "mode", "Unknown", or None
    convert_dates: bool = True,
    remove_empty_columns: bool = True,
    remove_constant_columns: bool = False,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Apply cleaning operations to a COPY of df. Returns (cleaned_df, change_log).
    """
    cleaned = df.copy(deep=True)
    log: List[str] = []
    column_types = detect_column_types(cleaned)

    # 1. Remove empty columns
    if remove_empty_columns:
        empty_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
        if empty_cols:
            cleaned = cleaned.drop(columns=empty_cols)
            log.append(f"✓ Removed {len(empty_cols)} empty column(s): {', '.join(empty_cols)}")

    # 2. Remove constant columns (optional, off by default so we don't destroy useful flags)
    if remove_constant_columns:
        const_cols = [c for c in cleaned.columns if cleaned[c].nunique(dropna=True) <= 1]
        if const_cols:
            cleaned = cleaned.drop(columns=const_cols)
            log.append(f"✓ Removed {len(const_cols)} constant column(s): {', '.join(const_cols)}")

    # 3. Remove duplicates
    if remove_duplicates:
        n_before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        n_removed = n_before - len(cleaned)
        if n_removed:
            log.append(f"✓ {n_removed:,} duplicate row(s) removed")

    # 4. Convert date-like columns to real datetimes
    if convert_dates:
        converted = []
        for col, ctype in column_types.items():
            if col not in cleaned.columns:
                continue
            if ctype == "date" and not pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce", format="mixed")
                converted.append(col)
        if converted:
            log.append(f"✓ {len(converted)} column(s) converted to datetime: {', '.join(converted)}")

    # 5. Fill missing numeric values
    if fill_numeric:
        filled_cols = []
        for col in cleaned.columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]) and cleaned[col].isna().any():
                fill_value = cleaned[col].median() if fill_numeric == "median" else cleaned[col].mean()
                cleaned[col] = cleaned[col].fillna(fill_value)
                filled_cols.append(col)
        if filled_cols:
            log.append(f"✓ Filled missing numeric values in {len(filled_cols)} column(s) using {fill_numeric}")

    # 6. Fill missing categorical values
    if fill_categorical:
        filled_cols = []
        for col, ctype in column_types.items():
            if col not in cleaned.columns:
                continue
            if ctype == "categorical" and cleaned[col].isna().any():
                if fill_categorical == "mode" and not cleaned[col].mode().empty:
                    fill_value = cleaned[col].mode().iloc[0]
                else:
                    fill_value = "Unknown"
                cleaned[col] = cleaned[col].fillna(fill_value)
                filled_cols.append(col)
        if filled_cols:
            log.append(f"✓ Filled missing categorical values in {len(filled_cols)} column(s) "
                        f"using '{fill_categorical}'")

    if not log:
        log.append("✓ No cleaning actions were necessary — dataset already looks clean.")

    return cleaned, log
