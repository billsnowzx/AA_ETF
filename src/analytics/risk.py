"""Basic risk-adjusted performance metrics."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from src.analytics.drawdown import max_drawdown_from_returns
from src.analytics.returns import annualized_return, annualized_volatility

PandasLike = Union[pd.Series, pd.DataFrame]


def downside_volatility(
    returns: PandasLike,
    periods_per_year: int = 252,
    minimum_acceptable_return: float = 0.0,
) -> float | pd.Series:
    """Compute annualized downside volatility from a simple return series."""
    downside = (returns - minimum_acceptable_return).clip(upper=0.0)
    return downside.std(ddof=1) * np.sqrt(periods_per_year)


def _safe_ratio(
    numerator: float | pd.Series,
    denominator: float | pd.Series,
) -> float | pd.Series:
    """Divide two aligned objects while returning NaN for zero denominators."""
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0.0, np.nan)
        if isinstance(numerator, pd.Series):
            return numerator / denominator
        return pd.Series(numerator, index=denominator.index, dtype=float) / denominator

    if denominator == 0 or pd.isna(denominator):
        return float("nan")
    return float(numerator) / denominator


def sharpe_ratio(
    returns: PandasLike,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float | pd.Series:
    """Compute annualized Sharpe ratio from simple returns."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    ann_return = annualized_return(excess_returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(excess_returns, periods_per_year=periods_per_year)
    return _safe_ratio(ann_return, ann_vol)


def sortino_ratio(
    returns: PandasLike,
    periods_per_year: int = 252,
    minimum_acceptable_return: float = 0.0,
) -> float | pd.Series:
    """Compute annualized Sortino ratio from simple returns."""
    ann_return = annualized_return(returns, periods_per_year=periods_per_year)
    downside_vol = downside_volatility(
        returns,
        periods_per_year=periods_per_year,
        minimum_acceptable_return=minimum_acceptable_return,
    )
    return _safe_ratio(ann_return, downside_vol)


def calmar_ratio(
    returns: PandasLike,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Compute Calmar ratio from simple returns."""
    ann_return = annualized_return(returns, periods_per_year=periods_per_year)
    worst_drawdown = max_drawdown_from_returns(returns)
    if isinstance(worst_drawdown, pd.Series):
        worst_drawdown = worst_drawdown.abs()
    else:
        worst_drawdown = abs(worst_drawdown)
    return _safe_ratio(ann_return, worst_drawdown)


def risk_summary(
    returns: PandasLike,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    minimum_acceptable_return: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """Return a compact table of annualized and drawdown-aware metrics."""
    metrics = {
        "annualized_return": annualized_return(returns, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year=periods_per_year),
        "downside_volatility": downside_volatility(
            returns,
            periods_per_year=periods_per_year,
            minimum_acceptable_return=minimum_acceptable_return,
        ),
        "sharpe_ratio": sharpe_ratio(
            returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        "sortino_ratio": sortino_ratio(
            returns,
            periods_per_year=periods_per_year,
            minimum_acceptable_return=minimum_acceptable_return,
        ),
        "max_drawdown": max_drawdown_from_returns(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year=periods_per_year),
    }

    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(metrics)

    return pd.Series(metrics)


def rolling_volatility(
    returns: PandasLike,
    window: int = 63,
    periods_per_year: int = 252,
    min_periods: int | None = None,
) -> PandasLike:
    """Compute annualized rolling volatility from simple returns."""
    min_periods = window if min_periods is None else min_periods
    return returns.rolling(window=window, min_periods=min_periods).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_sharpe_ratio(
    returns: PandasLike,
    window: int = 63,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int | None = None,
) -> PandasLike:
    """Compute annualized rolling Sharpe ratio from simple returns."""
    min_periods = window if min_periods is None else min_periods
    periodic_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns - periodic_risk_free_rate
    rolling_mean = excess_returns.rolling(window=window, min_periods=min_periods).mean() * periods_per_year
    rolling_vol = rolling_volatility(
        excess_returns,
        window=window,
        periods_per_year=periods_per_year,
        min_periods=min_periods,
    )
    return rolling_mean / rolling_vol.replace(0.0, np.nan)


def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Compute portfolio variance from asset weights and a covariance matrix."""
    aligned_covariance = covariance.reindex(index=weights.index, columns=weights.index)
    if aligned_covariance.isna().any().any():
        missing_assets = [
            asset for asset in weights.index
            if asset not in covariance.index or asset not in covariance.columns
        ]
        if missing_assets:
            raise ValueError(f"Covariance matrix is missing weighted assets: {missing_assets}")
        raise ValueError("Covariance matrix contains NaN values for weighted assets.")
    return float(weights.T @ aligned_covariance @ weights)


def marginal_contribution_to_risk(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Compute each asset's marginal contribution to portfolio volatility."""
    aligned_covariance = covariance.reindex(index=weights.index, columns=weights.index)
    variance = portfolio_variance(weights, aligned_covariance)
    if variance <= 0.0 or pd.isna(variance):
        return pd.Series(float("nan"), index=weights.index, name="marginal_contribution_to_risk")

    portfolio_volatility = float(np.sqrt(variance))
    marginal = aligned_covariance @ weights / portfolio_volatility
    marginal.name = "marginal_contribution_to_risk"
    return marginal


def risk_contribution_table(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Build an auditable per-asset risk contribution table."""
    weights = weights.astype(float).copy()
    weights.name = "weight"
    marginal = marginal_contribution_to_risk(weights, covariance)
    absolute = (weights * marginal).rename("absolute_risk_contribution")
    total_absolute = float(absolute.sum())
    if total_absolute == 0.0 or pd.isna(total_absolute):
        percent = pd.Series(float("nan"), index=weights.index, name="percent_risk_contribution")
    else:
        percent = (absolute / total_absolute).rename("percent_risk_contribution")

    table = pd.concat([weights, marginal, absolute, percent], axis=1)
    table["portfolio_volatility"] = float(np.sqrt(portfolio_variance(weights, covariance)))
    table.index.name = "asset"
    return table
