import math

import pandas as pd

from src.backtest.engine import (
    calendar_rebalance_dates,
    compile_annual_return_table,
    compile_benchmark_annual_excess_returns,
    compile_benchmark_comparisons,
    compile_benchmark_drawdown_comparisons,
    run_fixed_weight_backtest,
)


def test_calendar_rebalance_dates_monthly_uses_first_trading_day_of_new_month() -> None:
    index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"])

    result = calendar_rebalance_dates(index, frequency="monthly")

    assert result.tolist() == [
        pd.Timestamp("2024-01-30"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]


def test_run_fixed_weight_backtest_single_asset_without_costs() -> None:
    asset_returns = pd.DataFrame(
        {"VTI": [0.01, 0.02]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    result = run_fixed_weight_backtest(
        asset_returns,
        {"VTI": 1.0},
        rebalance_frequency="monthly",
        one_way_bps=0.0,
        periods_per_year=2,
    )

    portfolio_nav = result["portfolio_nav"]
    turnover = result["turnover"]

    assert portfolio_nav.tolist() == [1.01, 1.0302]
    assert turnover.tolist() == [1.0, 0.0]


def test_run_fixed_weight_backtest_applies_rebalance_costs_and_turnover() -> None:
    asset_returns = pd.DataFrame(
        {
            "VTI": [0.10, 0.00],
            "AGG": [0.00, 0.00],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-01"]),
    )

    result = run_fixed_weight_backtest(
        asset_returns,
        {"VTI": 0.5, "AGG": 0.5},
        rebalance_frequency="monthly",
        one_way_bps=0.0,
        periods_per_year=2,
    )

    turnover = result["turnover"]
    rebalance_flags = result["rebalance_flags"]

    assert rebalance_flags.tolist() == [True, True]
    assert math.isclose(turnover.iloc[0], 1.0, rel_tol=1e-9)
    assert math.isclose(turnover.iloc[1], 0.04761904761904767, rel_tol=1e-9)


def test_run_fixed_weight_backtest_initial_cost_drag_reduces_first_day_return() -> None:
    asset_returns = pd.DataFrame(
        {"VTI": [0.01]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    result = run_fixed_weight_backtest(
        asset_returns,
        {"VTI": 1.0},
        rebalance_frequency="monthly",
        one_way_bps=5.0,
        periods_per_year=1,
    )

    assert math.isclose(result["transaction_costs"].iloc[0], 0.0005, rel_tol=1e-9)
    assert math.isclose(result["portfolio_returns"].iloc[0], 0.0095, rel_tol=1e-9)


def test_compile_annual_return_table_and_benchmark_comparisons() -> None:
    index = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-12-31"])
    portfolio_returns = pd.Series([0.10, 0.05, -0.02], index=index)
    benchmark_returns = {"benchmark_a": pd.Series([0.08, 0.04, -0.01], index=index)}

    annual_table = compile_annual_return_table(portfolio_returns, benchmark_returns=benchmark_returns)
    comparisons = compile_benchmark_comparisons(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
        periods_per_year=3,
    )
    excess_table = compile_benchmark_annual_excess_returns(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )
    drawdown_comparisons = compile_benchmark_drawdown_comparisons(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    assert list(annual_table.columns) == ["portfolio", "benchmark_a"]
    assert "tracking_error" in comparisons.columns
    assert "benchmark_a" in excess_table.columns
    assert "max_drawdown_gap" in drawdown_comparisons.columns


def test_run_fixed_weight_backtest_includes_benchmark_outputs() -> None:
    asset_returns = pd.DataFrame(
        {"VTI": [0.01, 0.02, -0.01]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    benchmark_returns = {"benchmark_a": pd.Series([0.005, 0.01, -0.005], index=asset_returns.index)}

    result = run_fixed_weight_backtest(
        asset_returns,
        {"VTI": 1.0},
        benchmark_returns=benchmark_returns,
        periods_per_year=3,
    )

    assert "benchmark_a" in result["annual_return_table"].columns
    assert "benchmark_a" in result["benchmark_comparisons"].index
    assert "benchmark_a" in result["benchmark_annual_excess_returns"].columns
    assert "benchmark_a" in result["benchmark_drawdown_comparisons"].index


def test_run_fixed_weight_backtest_applies_trend_filter_on_rebalance_dates() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-03-01"])
    asset_returns = pd.DataFrame(
        {
            "VTI": [0.0, 0.0, 0.0],
            "AGG": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    adj_close = pd.DataFrame(
        {
            "VTI": [100.0, 90.0, 80.0],
            "AGG": [100.0, 100.0, 100.0],
        },
        index=index,
    )

    result = run_fixed_weight_backtest(
        asset_returns,
        {"VTI": 0.6, "AGG": 0.4},
        rebalance_frequency="monthly",
        one_way_bps=0.0,
        adj_close=adj_close,
        trend_filter={
            "enabled": True,
            "moving_average_days": 2,
            "reduction_fraction": 0.5,
            "assets": ["VTI"],
        },
    )

    weights_start = result["weights_start"]
    trend_active = result["trend_filter_active"]
    trend_scales = result["trend_filter_scales"]

    assert math.isclose(float(weights_start.loc[pd.Timestamp("2024-03-01"), "VTI"]), 0.4285714286, rel_tol=1e-9)
    assert bool(trend_active.loc[pd.Timestamp("2024-03-01")]) is True
    assert math.isclose(float(trend_scales.loc[pd.Timestamp("2024-03-01"), "VTI"]), 0.5, rel_tol=1e-9)


def test_run_fixed_weight_backtest_trend_filter_requires_adj_close_data() -> None:
    asset_returns = pd.DataFrame(
        {"VTI": [0.01, 0.02]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    try:
        run_fixed_weight_backtest(
            asset_returns,
            {"VTI": 1.0},
            trend_filter={
                "enabled": True,
                "moving_average_days": 2,
                "reduction_fraction": 0.5,
                "assets": ["VTI"],
            },
        )
    except ValueError as exc:
        assert "adjusted-close data" in str(exc)
    else:
        raise AssertionError("Expected ValueError when trend filter is enabled without adjusted-close data.")
