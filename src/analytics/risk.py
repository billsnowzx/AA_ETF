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
