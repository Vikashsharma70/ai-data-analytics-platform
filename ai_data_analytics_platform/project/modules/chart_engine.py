"""
chart_engine.py
----------------
Builds Plotly figures using an "intelligent chart selection" approach:
the right chart type is chosen automatically based on the combination of
column types being visualised (time+numeric -> line, category+numeric ->
bar, two numerics -> scatter, distribution -> histogram/box,
part-to-whole -> donut, geography -> choropleth map, correlation ->
heatmap).

A consistent, professional colour theme is used across every chart so the
dashboard feels like a single cohesive BI product rather than a pile of
default matplotlib/plotly styles.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PRIMARY = "#2563EB"
PALETTE = px.colors.qualitative.Bold
TEMPLATE = "plotly_white"


import re


def _base_layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    clean_title = re.sub(r"Unnamed:\s*\d+\s*", "", title).strip()
    clean_title = re.sub(r"\s+", " ", clean_title)

    fig.update_layout(
        title=dict(
            text=clean_title,
            font=dict(size=15, family="Segoe UI, sans-serif"),
            x=0.01,
            y=0.96,
            xanchor="left",
            yanchor="top",
        ),
        template=TEMPLATE,
        height=height,
        margin=dict(t=80, l=45, r=45, b=50),
        font=dict(family="Segoe UI, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
    )
    return fig


def trend_line_chart(df: pd.DataFrame, date_col: str, value_col: str, agg: str = "sum",
                      title: Optional[str] = None) -> go.Figure:
    ts = df[[date_col, value_col]].dropna()
    ts = ts.set_index(date_col).resample("ME")[value_col].agg(agg).reset_index()
    fig = px.area(ts, x=date_col, y=value_col, color_discrete_sequence=[PRIMARY])
    fig.update_traces(line=dict(width=2.5), fillcolor="rgba(37, 99, 235, 0.12)")
    return _base_layout(fig, title or f"{value_col} Trend Over Time")


def multi_series_trend(ts: pd.DataFrame, date_col: str, value_col: str, type_col: str = "Type",
                        title: Optional[str] = None) -> go.Figure:
    """Used for forecast charts: solid line for Actual, dashed for Forecast."""
    fig = go.Figure()
    for label, dash in [("Actual", "solid"), ("Forecast", "dash")]:
        subset = ts[ts[type_col] == label]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset[date_col], y=subset[value_col], mode="lines+markers",
            name=label, line=dict(dash=dash, width=2.5,
                                   color=PRIMARY if label == "Actual" else "#F59E0B"),
        ))
    return _base_layout(fig, title or f"{value_col} Forecast")


def category_bar_chart(df: pd.DataFrame, category_col: str, value_col: str, agg: str = "sum",
                        top_n: int = 10, horizontal: bool = True,
                        title: Optional[str] = None) -> go.Figure:
    grouped = df.groupby(category_col)[value_col].agg(agg).sort_values(ascending=False).head(top_n)
    grouped = grouped.reset_index()
    if horizontal:
        fig = px.bar(grouped.sort_values(value_col), x=value_col, y=category_col, orientation="h",
                      color=value_col, color_continuous_scale="Blues", text_auto=".2s")
    else:
        fig = px.bar(grouped, x=category_col, y=value_col, color=value_col,
                      color_continuous_scale="Blues", text_auto=".2s")
    fig.update_layout(coloraxis_showscale=False)
    return _base_layout(fig, title or f"{value_col} by {category_col}")


def scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None,
                   title: Optional[str] = None) -> go.Figure:
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, opacity=0.7,
                      color_discrete_sequence=PALETTE, trendline="ols" if color_col is None else None)
    return _base_layout(fig, title or f"{y_col} vs {x_col}")


def histogram_chart(df: pd.DataFrame, col: str, bins: int = 30, title: Optional[str] = None) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=bins, color_discrete_sequence=[PRIMARY])
    return _base_layout(fig, title or f"Distribution of {col}")


def box_chart(df: pd.DataFrame, value_col: str, category_col: Optional[str] = None,
              title: Optional[str] = None) -> go.Figure:
    fig = px.box(df, x=category_col, y=value_col, color=category_col,
                 color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False)
    return _base_layout(fig, title or f"Spread of {value_col}" + (f" by {category_col}" if category_col else ""))


def donut_chart(df: pd.DataFrame, category_col: str, value_col: str, top_n: int = 8,
                 title: Optional[str] = None) -> go.Figure:
    grouped = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    if len(grouped) > top_n:
        top = grouped.head(top_n)
        other = pd.Series({"Other": grouped.iloc[top_n:].sum()})
        grouped = pd.concat([top, other])
    fig = px.pie(values=grouped.values, names=grouped.index, hole=0.5,
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="percent+label", textposition="outside")
    fig = _base_layout(fig, title or f"{value_col} Share by {category_col}")
    fig.update_layout(
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(t=75, l=20, r=130, b=30),
    )
    return fig


def pie_chart(df: pd.DataFrame, category_col: str, value_col: str, top_n: int = 8,
              title: Optional[str] = None) -> go.Figure:
    grouped = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    if len(grouped) > top_n:
        top = grouped.head(top_n)
        other = pd.Series({"Other": grouped.iloc[top_n:].sum()})
        grouped = pd.concat([top, other])
    fig = px.pie(values=grouped.values, names=grouped.index, hole=0.0,
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="percent+label", textposition="outside")
    fig = _base_layout(fig, title or f"{value_col} Pie Chart by {category_col}")
    fig.update_layout(
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(t=75, l=20, r=130, b=30),
    )
    return fig


def grouped_bar_chart(df: pd.DataFrame, category_col: str, value_cols: List[str],
                      title: Optional[str] = None) -> go.Figure:
    grouped = df.groupby(category_col)[value_cols].sum().reset_index()
    fig = px.bar(grouped, x=category_col, y=value_cols, barmode="group",
                 color_discrete_sequence=PALETTE)
    return _base_layout(fig, title or f"{' & '.join(value_cols)} by {category_col}")


def correlation_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     aspect="auto")
    return _base_layout(fig, title, height=450)


def geo_choropleth(df: pd.DataFrame, geo_col: str, value_col: str,
                    location_mode: str = "country names", title: Optional[str] = None) -> go.Figure:
    grouped = df.groupby(geo_col)[value_col].sum().reset_index()
    fig = px.choropleth(grouped, locations=geo_col, locationmode=location_mode,
                         color=value_col, color_continuous_scale="Blues")
    return _base_layout(fig, title or f"{value_col} by {geo_col}", height=460)


def kpi_gauge(value: float, max_value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": PRIMARY},
            "steps": [
                {"range": [0, max_value * 0.5], "color": "#FEE2E2"},
                {"range": [max_value * 0.5, max_value * 0.8], "color": "#FEF3C7"},
                {"range": [max_value * 0.8, max_value], "color": "#DCFCE7"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20), template=TEMPLATE)
    return fig
