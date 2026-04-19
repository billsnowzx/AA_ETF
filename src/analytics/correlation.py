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


def matrix_to_long_table(
    matrix: pd.DataFrame,
    value_name: str,
    include_diagonal: bool = True,
) -> pd.DataFrame:
    """Flatten a symmetric matrix into an auditable upper-triangle table."""
    if matrix.empty:
        return pd.DataFrame(columns=["left", "right", value_name])

    rows: list[dict[str, float | str]] = []
    columns = list(matrix.columns)

    for left_index, left in enumerate(columns):
        if left not in matrix.index:
            raise ValueError(f"Matrix index is missing label '{left}'.")

        for right_index, right in enumerate(columns):
            if right not in matrix.index:
                raise ValueError(f"Matrix index is missing label '{right}'.")
            if right_index < left_index:
                continue
            if not include_diagonal and left_index == right_index:
                continue

            rows.append(
                {
                    "left": left,
                    "right": right,
                    value_name: float(matrix.loc[left, right]),
                }
            )

    return pd.DataFrame(rows)


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
