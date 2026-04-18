import math

import pandas as pd

from src.analytics.drawdown import (
    drawdown_from_returns,
    drawdown_series,
    max_drawdown,
    max_drawdown_from_returns,
)


def test_drawdown_series_matches_running_peak_logic() -> None:
    cumulative = pd.Series([1.0, 1.10, 1.00, 1.20], index=pd.date_range("2024-01-01", periods=4))

    result = drawdown_series(cumulative)

    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.0
    assert math.isclose(result.iloc[2], (1.0 / 1.10) - 1.0, rel_tol=1e-9)
    assert result.iloc[3] == 0.0


def test_max_drawdown_from_cumulative_series() -> None:
    cumulative = pd.Series([1.0, 1.10, 0.99, 1.05], index=pd.date_range("2024-01-01", periods=4))

    result = max_drawdown(cumulative)

    assert math.isclose(result, -0.10, rel_tol=1e-9)


def test_drawdown_from_returns_and_max_drawdown_from_returns() -> None:
    returns = pd.Series([0.10, -0.10, 0.05], index=pd.date_range("2024-01-01", periods=3))

    drawdowns = drawdown_from_returns(returns)

    assert drawdowns.iloc[0] == 0.0
    assert math.isclose(drawdowns.iloc[1], -0.10, rel_tol=1e-9)
    assert math.isclose(max_drawdown_from_returns(returns), -0.10, rel_tol=1e-9)
