import pandas as pd

from src.analytics.evaluation import (
    annual_win_rate,
    build_portfolio_evaluation_summary,
    max_drawdown_recovery_days,
    monthly_win_rate,
    rolling_sharpe_stability,
)


def test_monthly_and_annual_win_rate_from_known_returns() -> None:
    index = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
            "2024-03-29",
            "2025-01-31",
            "2025-02-28",
            "2025-03-31",
        ]
    )
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, -0.01, 0.00], index=index)

    assert monthly_win_rate(returns) == (2.0 / 6.0)
    assert annual_win_rate(returns) == 0.5


def test_max_drawdown_recovery_days_uses_longest_underwater_stretch() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    nav = pd.Series([1.0, 1.1, 1.05, 1.0, 1.03, 1.08, 1.12, 1.11], index=index)

    assert max_drawdown_recovery_days(nav) == 4


def test_rolling_sharpe_stability_handles_empty_and_high_dispersion() -> None:
    assert rolling_sharpe_stability(pd.Series(dtype=float)) == 0.0

    stable = pd.Series([0.50, 0.55, 0.60, 0.65])
    unstable = pd.Series([-2.0, 0.0, 2.0, 4.0])

    assert rolling_sharpe_stability(stable) > 0.9
    assert rolling_sharpe_stability(unstable) == 0.0


def test_build_portfolio_evaluation_summary_generates_expected_columns() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    return_table = pd.DataFrame(
        {
            "balanced": [0.01, -0.01, 0.02, 0.0, 0.01],
            "benchmark_a": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    nav_table = pd.DataFrame(
        {
            "balanced": [1.0, 1.01, 1.00, 1.02, 1.03],
            "benchmark_a": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    rolling_sharpe = pd.DataFrame(
        {
            "balanced": [0.4, 0.5, 0.6, 0.7, 0.8],
            "benchmark_a": [0.1, 0.1, 0.1, 0.1, 0.1],
        },
        index=index,
    )

    summary = build_portfolio_evaluation_summary(return_table, nav_table, rolling_sharpe)

    assert summary.index.tolist() == ["balanced", "benchmark_a"]
    assert summary.columns.tolist() == [
        "monthly_win_rate",
        "annual_win_rate",
        "max_drawdown_recovery_days",
        "rolling_sharpe_stability",
    ]
    assert float(summary.loc["benchmark_a", "monthly_win_rate"]) == 0.0
    assert int(summary.loc["balanced", "max_drawdown_recovery_days"]) >= 0
