import pandas as pd

from src.portfolio.policy import (
    build_backtest_universe_validation,
    summarize_backtest_universe_validation,
)


def test_build_backtest_universe_validation_marks_non_liquid_assets() -> None:
    strategy_weights = pd.Series({"VTI": 0.6, "IAGG": 0.4})
    benchmark_weights = {
        "benchmark_a": {"weights": {"VTI": 1.0}},
        "benchmark_b": {"weights": {"VTI": 0.5, "IAGG": 0.5}},
    }

    validation = build_backtest_universe_validation(
        strategy_name="balanced",
        strategy_weights=strategy_weights,
        benchmark_weights=benchmark_weights,
        liquid_tickers=["VTI"],
    )

    assert bool(validation.loc[("balanced", "IAGG"), "requires_policy_override"]) is True
    assert bool(validation.loc[("benchmark_a", "VTI"), "passes_liquidity_filter"]) is True


def test_summarize_backtest_universe_validation_counts_overrides() -> None:
    validation = pd.DataFrame(
        {
            "target_weight": [0.6, 0.4, 1.0],
            "passes_liquidity_filter": [True, False, True],
            "requires_policy_override": [False, True, False],
        },
        index=pd.MultiIndex.from_tuples(
            [("balanced", "VTI"), ("balanced", "IAGG"), ("benchmark_a", "VTI")],
            names=["portfolio", "ticker"],
        ),
    )

    summary = summarize_backtest_universe_validation(validation)

    assert int(summary.loc["balanced", "policy_override_assets"]) == 1
    assert bool(summary.loc["benchmark_a", "is_fully_liquid"]) is True
