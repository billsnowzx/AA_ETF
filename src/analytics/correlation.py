"""Covariance and correlation utilities."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from src.analytics.returns import simple_returns


def build_adjusted_close_matrix(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine cleaned per-ticker frames into an adjusted-close price matrix."""
    series_map: dict[str, pd.Series] = {}

    for ticker, frame in frames.items():
        if "adj_close" not in frame.columns:
            raise ValueError(f"Ticker '{ticker}' is missing required column 'adj_close'.")
        series = frame["adj_close"].copy()
        series.name = ticker
        series_map[ticker] = series

    if not series_map:
        return pd.DataFrame()

    return pd.concat(series_map.values(), axis=1)


def return_matrix_from_prices(
    prices: pd.DataFrame,
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    """Convert an adjusted-close matrix into a simple return matrix."""
    returns = simple_returns(prices)
    if drop_all_nan_rows:
        returns = returns.dropna(how="all")
    return returns


def covariance_matrix(
    returns: pd.DataFrame,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the sample covariance matrix for a return matrix."""
    return returns.cov(min_periods=min_periods)


def correlation_matrix(
    returns: pd.DataFrame,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the sample correlation matrix for a return matrix."""
    return returns.corr(min_periods=min_periods)


def rolling_correlation(
    returns: pd.DataFrame,
    left: str,
    right: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute rolling correlation between two assets in a return matrix."""
    if left not in returns.columns or right not in returns.columns:
        missing = [column for column in [left, right] if column not in returns.columns]
        raise ValueError(f"Missing columns for rolling correlation: {missing}")

    min_periods = window if min_periods is None else min_periods
    return returns[left].rolling(window=window, min_periods=min_periods).corr(returns[right])
