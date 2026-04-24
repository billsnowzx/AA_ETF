import math

import pandas as pd

from src.analytics.risk import (
    calmar_ratio,
    downside_volatility,
    marginal_contribution_to_risk,
    portfolio_variance,
    risk_contribution_table,
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


def test_risk_contribution_table_sums_to_portfolio_volatility() -> None:
    weights = pd.Series({"VTI": 0.6, "AGG": 0.4})
    covariance = pd.DataFrame(
        [[0.04, 0.00], [0.00, 0.01]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )

    table = risk_contribution_table(weights, covariance)

    assert math.isclose(portfolio_variance(weights, covariance), 0.016, rel_tol=1e-9)
    assert math.isclose(float(table["absolute_risk_contribution"].sum()), math.sqrt(0.016), rel_tol=1e-9)
    assert math.isclose(float(table["percent_risk_contribution"].sum()), 1.0, rel_tol=1e-9)
    assert table.loc["VTI", "percent_risk_contribution"] > table.loc["AGG", "percent_risk_contribution"]


def test_marginal_contribution_to_risk_returns_nan_for_zero_variance() -> None:
    weights = pd.Series({"VTI": 1.0})
    covariance = pd.DataFrame([[0.0]], index=["VTI"], columns=["VTI"])

    marginal = marginal_contribution_to_risk(weights, covariance)

    assert math.isnan(float(marginal.loc["VTI"]))
