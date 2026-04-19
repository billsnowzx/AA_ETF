import math

import pandas as pd
import pytest

from src.analytics.attribution import (
    align_return_series,
    annual_return_table,
    benchmark_annual_excess_return_table,
    benchmark_comparison,
    benchmark_drawdown_comparison,
    excess_returns,
    information_ratio,
    tracking_error,
)


def test_align_return_series_inner_joins_dates() -> None:
    strategy = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    benchmark = pd.Series(
        [0.00, 0.01, 0.02],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    aligned = align_return_series(strategy, benchmark)

    assert aligned.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert list(aligned.columns) == ["strategy", "benchmark"]


def test_align_return_series_raises_when_no_overlap() -> None:
    strategy = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"]))
    benchmark = pd.Series([0.01], index=pd.to_datetime(["2024-01-02"]))

    with pytest.raises(ValueError, match="do not overlap"):
        align_return_series(strategy, benchmark)


def test_excess_returns_and_tracking_error() -> None:
    strategy = pd.Series([0.02, 0.01, 0.00, 0.03], index=pd.date_range("2024-01-01", periods=4))
    benchmark = pd.Series([0.01, 0.01, 0.00, 0.01], index=pd.date_range("2024-01-01", periods=4))

    excess = excess_returns(strategy, benchmark)
    te = tracking_error(strategy, benchmark, periods_per_year=4)

    assert excess.tolist() == [0.01, 0.0, 0.0, 0.019999999999999997]
    assert te > 0


def test_information_ratio_is_nan_when_tracking_error_is_zero() -> None:
    strategy = pd.Series([0.01, 0.01, 0.01, 0.01], index=pd.date_range("2024-01-01", periods=4))
    benchmark = pd.Series([0.01, 0.01, 0.01, 0.01], index=pd.date_range("2024-01-01", periods=4))

    result = information_ratio(strategy, benchmark, periods_per_year=4)

    assert math.isnan(result)


def test_annual_return_table_for_series_and_dataframe() -> None:
    index = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-12-31"])
    series = pd.Series([0.10, 0.05, -0.02], index=index)
    frame = pd.DataFrame({"VTI": series, "AGG": [0.02, 0.01, 0.00]}, index=index)

    series_table = annual_return_table(series)
    frame_table = annual_return_table(frame)

    assert list(series_table.index) == [2023, 2024]
    assert "annual_return" in series_table.columns
    assert list(frame_table.columns) == ["VTI", "AGG"]


def test_benchmark_comparison_contains_expected_fields() -> None:
    strategy = pd.Series([0.02, 0.01, -0.01, 0.03], index=pd.date_range("2024-01-01", periods=4))
    benchmark = pd.Series([0.01, 0.00, -0.01, 0.01], index=pd.date_range("2024-01-01", periods=4))

    comparison = benchmark_comparison(strategy, benchmark, periods_per_year=4)

    assert list(comparison.index) == [
        "strategy_annualized_return",
        "benchmark_annualized_return",
        "annualized_excess_return",
        "tracking_error",
        "information_ratio",
    ]
    assert comparison["tracking_error"] > 0


def test_benchmark_annual_excess_return_table_builds_calendar_gaps() -> None:
    index = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-12-31"])
    strategy = pd.Series([0.10, 0.05, -0.02], index=index)
    benchmark_returns = {
        "benchmark_a": pd.Series([0.08, 0.04, -0.01], index=index),
    }

    result = benchmark_annual_excess_return_table(strategy, benchmark_returns)

    assert list(result.columns) == ["benchmark_a"]
    assert list(result.index) == [2023, 2024]
    assert result.loc[2023, "benchmark_a"] > 0


def test_benchmark_drawdown_comparison_contains_gap_fields() -> None:
    strategy = pd.Series([0.05, -0.10, 0.02, -0.03], index=pd.date_range("2024-01-01", periods=4))
    benchmark_returns = {
        "benchmark_a": pd.Series([0.03, -0.05, 0.01, -0.01], index=strategy.index),
    }

    result = benchmark_drawdown_comparison(strategy, benchmark_returns)

    assert list(result.columns) == [
        "strategy_max_drawdown",
        "benchmark_max_drawdown",
        "max_drawdown_gap",
    ]
    assert "benchmark_a" in result.index
