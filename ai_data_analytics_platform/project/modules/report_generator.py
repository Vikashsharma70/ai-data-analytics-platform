"""
report_generator.py
--------------------
Assembles the downloadable Analysis Report (HTML) and Insights Report
(TXT) from the profile, KPIs, data story, and insights already computed
elsewhere in the app. Also provides helpers for cleaned-dataset/dashboard
CSV exports.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List

import pandas as pd

from utils.helpers import format_number


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buffer.getvalue()


def build_insights_text_report(
    file_name: str,
    profile: Dict,
    kpis: List[Dict],
    quality_score: int,
    quality_notes: List[str],
    story: Dict[str, str],
    insights: List[str],
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("AUTOMATED DATA ANALYTICS — INSIGHTS REPORT")
    lines.append("=" * 70)
    lines.append(f"Source file: {file_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("-- DATASET OVERVIEW --")
    lines.append(f"Rows: {profile['n_rows']:,}   Columns: {profile['n_cols']:,}")
    lines.append(f"Numerical columns: {profile['numerical_columns']}   "
                  f"Categorical columns: {profile['categorical_columns']}   "
                  f"Date columns: {profile['date_columns']}")
    lines.append(f"Missing values: {profile['missing_pct']}%   Duplicate rows: {profile['duplicate_rows']:,}")
    lines.append("")
    lines.append(f"-- DATA QUALITY SCORE: {quality_score}/100 --")
    for note in quality_notes:
        lines.append(f"  {note}")
    lines.append("")
    lines.append("-- KEY PERFORMANCE INDICATORS --")
    for kpi in kpis:
        val = kpi["value"]
        display = (f"{val:.1f}%" if kpi.get("is_percent") else
                   format_number(val, is_currency=kpi.get("is_currency", False)))
        lines.append(f"  {kpi['label']}: {display}")
    lines.append("")
    lines.append("-- DATA STORY --")
    section_titles = {
        "what_happened": "1. What happened?",
        "why": "2. Why did it happen?",
        "where": "3. Where did it happen?",
        "what_changed": "4. What changed?",
        "concerns": "5. What is concerning?",
        "recommendations": "6. What should we do?",
    }
    for key, title in section_titles.items():
        if key in story:
            lines.append(f"\n{title}")
            lines.append(story[key])
    lines.append("")
    lines.append("-- ADDITIONAL INSIGHTS --")
    for i, insight in enumerate(insights, 1):
        lines.append(f"  {i}. {insight}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("End of report — generated automatically by the AI-Powered Data Analytics Platform")
    return "\n".join(lines)


def build_html_report(
    file_name: str,
    profile: Dict,
    kpis: List[Dict],
    quality_score: int,
    quality_notes: List[str],
    story: Dict[str, str],
    insights: List[str],
) -> str:
    def _kpi_display(k):
        val = k["value"]
        if k.get("is_percent"):
            return f"{val:.1f}%"
        return format_number(val, is_currency=k.get("is_currency", False))

    kpi_rows = "".join(
        f"<tr><td>{k['label']}</td><td>{_kpi_display(k)}</td></tr>"
        for k in kpis
    )
    quality_list = "".join(f"<li>{n}</li>" for n in quality_notes)
    insight_list = "".join(f"<li>{i}</li>" for i in insights)

    section_titles = {
        "what_happened": "What happened?",
        "why": "Why did it happen?",
        "where": "Where did it happen?",
        "what_changed": "What changed?",
        "concerns": "What is concerning?",
        "recommendations": "What should we do?",
    }
    story_html = ""
    for key, title in section_titles.items():
        if key in story:
            body = story[key].replace("- ", "<br>• ")
            story_html += f"<h3>{title}</h3><p>{body}</p>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Analytics Report — {file_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2937; background:#f9fafb;}}
  h1 {{ color: #1e3a8a; }}
  h2 {{ color: #2563eb; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; margin-top: 36px;}}
  h3 {{ color: #1f2937; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background:#fff;}}
  th, td {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
  th {{ background-color: #eff6ff; }}
  .meta {{ color: #6b7280; font-size: 14px; }}
  .score {{ font-size: 32px; font-weight: bold; color: #2563eb; }}
  ul {{ background: #fff; padding: 16px 32px; border-radius: 6px; }}
</style>
</head>
<body>
  <h1>📊 Automated Data Analytics Report</h1>
  <p class="meta">Source file: <b>{file_name}</b> &nbsp;|&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

  <h2>Dataset Overview</h2>
  <table>
    <tr><th>Rows</th><td>{profile['n_rows']:,}</td></tr>
    <tr><th>Columns</th><td>{profile['n_cols']:,}</td></tr>
    <tr><th>Numerical Columns</th><td>{profile['numerical_columns']}</td></tr>
    <tr><th>Categorical Columns</th><td>{profile['categorical_columns']}</td></tr>
    <tr><th>Date Columns</th><td>{profile['date_columns']}</td></tr>
    <tr><th>Missing Values</th><td>{profile['missing_pct']}%</td></tr>
    <tr><th>Duplicate Rows</th><td>{profile['duplicate_rows']:,}</td></tr>
  </table>

  <h2>Data Quality Score</h2>
  <p class="score">{quality_score}/100</p>
  <ul>{quality_list}</ul>

  <h2>Key Performance Indicators</h2>
  <table><tr><th>KPI</th><th>Value</th></tr>{kpi_rows}</table>

  <h2>Data Story</h2>
  {story_html}

  <h2>Key Insights</h2>
  <ul>{insight_list}</ul>

  <p class="meta">Generated automatically by the AI-Powered Automated Data Analytics Platform.</p>
</body>
</html>
"""
    return html
