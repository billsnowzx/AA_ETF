import math

import pandas as pd

from src.analytics.risk import (
    calmar_ratio,
    downside_volatility,
    risk_summary,
    rolling_sharpe_ratio,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
)


def test_downside_volatility_is_zero_when_no_negative_returns() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])

    result = downside_volatility(returns, periods_per_year=3)

    assert result == 0.0


def test_sharpe_ratio_returns_nan_for_zero_volatility() -> None:
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])

    result = sharpe_ratio(returns, periods_per_year=4)

    assert math.isnan(result)


def test_sortino_and_calmar_are_positive_for_positive_return_stream() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, 0.01], index=pd.date_range("2024-01-01", periods=4))

    sortino = sortino_ratio(returns, periods_per_year=4)
    calmar = calmar_ratio(returns, periods_per_year=4)

    assert sortino > 0
    assert calmar > 0


def test_risk_summary_for_dataframe_contains_expected_metrics() -> None:
    returns = pd.DataFrame(
        {
            "VTI": [0.01, -0.02, 0.03, 0.01],
            "AGG": [0.002, 0.001, -0.001, 0.0],
        }
    )

    summary = risk_summary(returns, periods_per_year=4)

    assert list(summary.columns) == [
        "annualized_return",
        "annualized_volatility",
        "downside_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
    ]
    assert list(summary.index) == ["VTI", "AGG"]


def test_rolling_volatility_and_sharpe_return_aligned_series() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.00], index=pd.date_range("2024-01-01", periods=4))

    vol = rolling_volatility(returns, window=2, periods_per_year=2)
    sharpe = rolling_sharpe_ratio(returns, window=2, periods_per_year=2)

    assert vol.index.equals(returns.index)
    assert sharpe.index.equals(returns.index)
    assert pd.isna(vol.iloc[0])
    assert vol.iloc[-1] > 0
