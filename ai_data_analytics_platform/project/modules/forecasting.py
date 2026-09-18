"""
forecasting.py
---------------
Lightweight time-series forecasting: moving average and a linear-trend
projection. No heavy dependencies required (numpy least-squares is used
for the trend line); forecasts are always clearly labelled as predictions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def aggregate_timeseries(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "ME") -> pd.DataFrame:
    work = df[[date_col, value_col]].dropna()
    work = work.set_index(date_col).resample(freq)[value_col].sum().reset_index()
    return work


def moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def linear_trend_forecast(ts: pd.DataFrame, date_col: str, value_col: str,
                           periods: int = 3, freq: str = "ME") -> Optional[pd.DataFrame]:
    """
    Fit a simple linear trend to the historical aggregated time series and
    project `periods` steps into the future. Returns a DataFrame with
    columns [date_col, value_col, 'Type'] where Type is 'Actual' or 'Forecast',
    or None if there isn't enough data (< 4 points) to fit a trend.
    """
    if len(ts) < 4:
        return None

    x = np.arange(len(ts))
    y = ts[value_col].values
    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = coeffs[0], coeffs[1]

    future_x = np.arange(len(ts), len(ts) + periods)
    future_y = slope * future_x + intercept
    future_y = np.maximum(future_y, 0)  # avoid negative forecasted revenue

    last_date = ts[date_col].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

    actual = ts.copy()
    actual["Type"] = "Actual"

    forecast = pd.DataFrame({date_col: future_dates, value_col: future_y})
    forecast["Type"] = "Forecast"

    combined = pd.concat([actual, forecast], ignore_index=True)
    return combined


def growth_rate(ts: pd.DataFrame, value_col: str) -> Optional[float]:
    """Period-over-period growth rate (%) between the last two periods."""
    if len(ts) < 2:
        return None
    prev, curr = ts[value_col].iloc[-2], ts[value_col].iloc[-1]
    if prev == 0:
        return None
    return round((curr - prev) / abs(prev) * 100, 2)
