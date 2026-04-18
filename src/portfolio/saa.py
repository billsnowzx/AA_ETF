"""Static strategic allocation helpers."""

from __future__ import annotations

import pandas as pd

from src.analytics.attribution import benchmark_comparison
from src.analytics.returns import cumulative_return_series
from src.analytics.risk import risk_summary
from src.portfolio.weights import normalize_weights


def align_weights_to_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
) -> pd.Series:
    """Align and normalize portfolio weights to the asset return matrix columns."""
    normalized = normalize_weights(weights)
    missing_columns = [ticker for ticker in normalized.index if ticker not in asset_returns.columns]
    if missing_columns:
        raise ValueError(f"Asset return matrix is missing columns required by weights: {missing_columns}")

    return normalized.reindex(asset_returns.columns, fill_value=0.0)


def static_weight_return_contributions(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
) -> pd.DataFrame:
    """Compute per-asset return contributions under static target weights."""
    aligned_weights = align_weights_to_returns(asset_returns, weights)
    return asset_returns.mul(aligned_weights, axis="columns")


def static_weight_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
) -> pd.Series:
    """Compute portfolio returns under a static target-weight approximation."""
    contributions = static_weight_return_contributions(asset_returns, weights)
    return contributions.sum(axis=1).rename("portfolio_return")


def static_weight_portfolio_nav(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    starting_value: float = 1.0,
) -> pd.Series:
    """Compute a portfolio NAV series from static-weight returns."""
    portfolio_returns = static_weight_portfolio_returns(asset_returns, weights)
    nav = cumulative_return_series(portfolio_returns, starting_value=starting_value)
    nav.name = "portfolio_nav"
    return nav


def static_weight_portfolio_summary(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> pd.Series:
    """Build a compact risk and benchmark summary for a static-weight portfolio."""
    portfolio_returns = static_weight_portfolio_returns(asset_returns, weights)
    summary = risk_summary(portfolio_returns, periods_per_year=periods_per_year)

    if benchmark_returns is None:
        return summary

    comparison = benchmark_comparison(
        portfolio_returns,
        benchmark_returns,
        periods_per_year=periods_per_year,
    )
    return pd.concat([summary, comparison])
