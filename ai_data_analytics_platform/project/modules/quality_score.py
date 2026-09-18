"""
quality_score.py
-----------------
Computes an overall Data Quality Score (0-100) plus a breakdown explaining
what drove the score up or down.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from modules.data_profiler import detect_constant_and_empty_columns
from modules.data_cleaner import detect_issues


def compute_quality_score(df: pd.DataFrame) -> Tuple[int, List[str]]:
    n_rows, n_cols = df.shape
    notes: List[str] = []
    score = 100.0

    issues = detect_issues(df)
    struct = detect_constant_and_empty_columns(df)

    # Missing values penalty (up to 30 points)
    total_cells = max(n_rows * n_cols, 1)
    missing_pct = issues.missing_found / total_cells * 100
    missing_penalty = min(30, missing_pct * 1.5)
    score -= missing_penalty
    if missing_pct < 1:
        notes.append("✓ Excellent completeness — very few missing values")
    elif missing_pct < 5:
        notes.append("⚠ Some missing values present")
    else:
        notes.append(f"⚠ Significant missing data ({missing_pct:.1f}% of all cells)")

    # Duplicate rows penalty (up to 20 points)
    dup_pct = (issues.duplicates_found / n_rows * 100) if n_rows else 0
    dup_penalty = min(20, dup_pct * 2)
    score -= dup_penalty
    if dup_pct < 1:
        notes.append("✓ Low duplicate rate")
    else:
        notes.append(f"⚠ {dup_pct:.1f}% of rows are duplicates")

    # Outliers penalty (up to 15 points)
    numeric_cells = 0
    from utils.column_detection import detect_column_types
    types = detect_column_types(df)
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    numeric_cells = max(n_rows * max(len(numeric_cols), 1), 1)
    outlier_pct = issues.numeric_outliers_found / numeric_cells * 100
    outlier_penalty = min(15, outlier_pct * 1.2)
    score -= outlier_penalty
    if outlier_pct > 5:
        notes.append("⚠ Several extreme outliers detected in numeric columns")
    else:
        notes.append("✓ Outlier levels are within a normal range")

    # Empty / constant columns penalty (up to 15 points)
    struct_penalty = min(15, (len(struct["empty_columns"]) * 5 + len(struct["constant_columns"]) * 2))
    score -= struct_penalty
    if struct["empty_columns"]:
        notes.append(f"⚠ {len(struct['empty_columns'])} completely empty column(s) found")
    if struct["constant_columns"]:
        notes.append(f"⚠ {len(struct['constant_columns'])} constant-value column(s) found (little analytical value)")

    # Data type consistency penalty (up to 10 points) — invalid dates as a proxy
    if issues.invalid_dates_found > 0:
        score -= min(10, issues.invalid_dates_found / max(n_rows, 1) * 100)
        notes.append(f"⚠ {issues.invalid_dates_found} invalid/unparseable date value(s)")
    else:
        notes.append("✓ Date columns parsed cleanly")

    # Row-count / usability floor (up to 10 points)
    if n_rows < 10:
        score -= 10
        notes.append("⚠ Very few rows — statistical conclusions may not be reliable")

    score = max(0, min(100, round(score)))
    return int(score), notes
