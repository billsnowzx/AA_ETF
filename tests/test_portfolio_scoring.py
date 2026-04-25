import math

import pandas as pd

from src.portfolio.portfolio_scoring import (
    annual_win_rate,
    build_portfolio_score_summary,
    monthly_win_rate,
    rolling_sharpe_stability_score,
)


def test_monthly_and_annual_win_rate_from_deterministic_returns() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29", "2025-01-02"])
    returns = pd.Series([0.01, -0.005, 0.02, 0.01, -0.01], index=index)

    monthly = monthly_win_rate(returns)
    yearly = annual_win_rate(returns)

    assert math.isclose(monthly, 2.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(yearly, 0.5, rel_tol=1e-9)


def test_rolling_sharpe_stability_score_penalizes_dispersion() -> None:
    stable = pd.Series([0.5, 0.5, 0.5, 0.5])
    volatile = pd.Series([0.5, -0.5, 1.0, -1.0])

    stable_score = rolling_sharpe_stability_score(stable)
    volatile_score = rolling_sharpe_stability_score(volatile)

    assert stable_score >= volatile_score
    assert 0.0 <= stable_score <= 1.0
    assert 0.0 <= volatile_score <= 1.0


def test_build_portfolio_score_summary_orders_higher_quality_portfolio_first() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    performance_summary = pd.DataFrame(
        {
            "annualized_return": [0.10, 0.02],
            "annualized_volatility": [0.12, 0.22],
            "max_drawdown": [-0.15, -0.35],
            "sharpe_ratio": [1.0, 0.2],
            "sortino_ratio": [1.4, 0.3],
            "calmar_ratio": [0.8, 0.1],
        },
        index=pd.Index(["balanced", "benchmark_a"], name="portfolio"),
    )
    turnover_summary = pd.DataFrame(
        {
            "average_turnover": [0.03, 0.10],
            "total_transaction_cost_drag": [0.003, 0.012],
        },
        index=pd.Index(["balanced", "benchmark_a"], name="portfolio"),
    )
    return_table = pd.DataFrame(
        {
            "balanced": [0.01, 0.005, 0.0, -0.002],
            "benchmark_a": [0.002, -0.004, 0.001, -0.003],
        },
        index=index,
    )
    rolling_sharpe = pd.DataFrame(
        {
            "balanced": [0.8, 0.82, 0.79, 0.81],
            "benchmark_a": [0.2, 0.0, 0.4, -0.1],
        },
        index=index,
    )

    summary = build_portfolio_score_summary(
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        return_table=return_table,
        rolling_sharpe_table=rolling_sharpe,
    )

    assert summary.index.tolist()[0] == "balanced"
    assert int(summary.loc["balanced", "rank"]) == 1
    assert summary.loc["balanced", "total_score"] > summary.loc["benchmark_a", "total_score"]
    assert 0.0 <= float(summary.loc["balanced", "score_pct"]) <= 1.0

