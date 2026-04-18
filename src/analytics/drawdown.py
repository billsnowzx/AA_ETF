"""Drawdown calculation utilities."""

from __future__ import annotations

from typing import Union

import pandas as pd

from src.analytics.returns import cumulative_return_series

PandasLike = Union[pd.Series, pd.DataFrame]


def drawdown_series(cumulative_values: PandasLike) -> PandasLike:
    """Compute drawdown relative to the running peak."""
    running_peak = cumulative_values.cummax()
    return (cumulative_values / running_peak) - 1.0


def drawdown_from_returns(
    returns: PandasLike,
    starting_value: float = 1.0,
) -> PandasLike:
    """Compute drawdown directly from a simple return series."""
    cumulative_values = cumulative_return_series(returns, starting_value=starting_value)
    return drawdown_series(cumulative_values)


def max_drawdown(cumulative_values: PandasLike) -> float | pd.Series:
    """Return the worst drawdown observed in the cumulative series."""
    drawdowns = drawdown_series(cumulative_values)
    return drawdowns.min()


def max_drawdown_from_returns(
    returns: PandasLike,
    starting_value: float = 1.0,
) -> float | pd.Series:
    """Return the worst drawdown observed in a simple return series."""
    drawdowns = drawdown_from_returns(returns, starting_value=starting_value)
    return drawdowns.min()
