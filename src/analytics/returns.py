"""Return calculation utilities."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

PandasLike = Union[pd.Series, pd.DataFrame]


def simple_returns(prices: PandasLike) -> PandasLike:
    """Compute period-over-period simple returns from price levels."""
    return prices.pct_change()


def log_returns(prices: PandasLike) -> PandasLike:
    """Compute period-over-period log returns from price levels."""
    return np.log(prices / prices.shift(1))


def cumulative_return_series(
    returns: PandasLike,
    starting_value: float = 1.0,
) -> PandasLike:
    """Compound a return series into a cumulative value series."""
    return starting_value * (1.0 + returns.fillna(0.0)).cumprod()


def annualized_return(
    returns: PandasLike,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Compute annualized return from a simple return series."""
    cleaned = returns.dropna()
    periods = len(cleaned)
    if periods == 0:
        if isinstance(returns, pd.DataFrame):
            return pd.Series(np.nan, index=returns.columns, dtype=float)
        return float("nan")

    compounded = (1.0 + cleaned).prod()
    return compounded ** (periods_per_year / periods) - 1.0


def annualized_volatility(
    returns: PandasLike,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """Compute annualized volatility from a simple return series."""
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def annualized_statistics(
    returns: PandasLike,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """Return annualized return and volatility in a compact tabular shape."""
    ann_return = annualized_return(returns, periods_per_year=periods_per_year)
    ann_volatility = annualized_volatility(returns, periods_per_year=periods_per_year)

    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                "annualized_return": ann_return,
                "annualized_volatility": ann_volatility,
            }
        )

    return pd.Series(
        {
            "annualized_return": ann_return,
            "annualized_volatility": ann_volatility,
        }
    )
