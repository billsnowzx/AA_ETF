import math

import pandas as pd

from src.analytics.returns import (
    annualized_return,
    annualized_statistics,
    annualized_volatility,
    cumulative_return_series,
    log_returns,
    simple_returns,
)


def test_simple_returns_for_price_series() -> None:
    prices = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024-01-01", periods=3))

    result = simple_returns(prices)

    assert pd.isna(result.iloc[0])
    assert math.isclose(result.iloc[1], 0.10, rel_tol=1e-9)
    assert math.isclose(result.iloc[2], 0.10, rel_tol=1e-9)


def test_log_returns_matches_known_values() -> None:
    prices = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-01", periods=2))

    result = log_returns(prices)

    assert math.isclose(result.iloc[1], math.log(1.10), rel_tol=1e-9)


def test_cumulative_return_series_with_zero_and_positive_returns() -> None:
    returns = pd.Series([0.0, 0.10, -0.05], index=pd.date_range("2024-01-01", periods=3))

    result = cumulative_return_series(returns)

    assert result.tolist() == [1.0, 1.1, 1.045]


def test_annualized_return_for_constant_daily_returns() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    result = annualized_return(returns, periods_per_year=4)

    assert math.isclose(result, (1.01**4) - 1.0, rel_tol=1e-9)


def test_annualized_volatility_is_zero_for_flat_returns() -> None:
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])

    result = annualized_volatility(returns, periods_per_year=252)

    assert result == 0.0


def test_annualized_statistics_for_dataframe() -> None:
    returns = pd.DataFrame(
        {
            "VTI": [0.01, 0.00, -0.01],
            "AGG": [0.002, 0.001, 0.0],
        }
    )

    result = annualized_statistics(returns, periods_per_year=3)

    assert list(result.columns) == ["annualized_return", "annualized_volatility"]
    assert list(result.index) == ["VTI", "AGG"]
