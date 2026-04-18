"""Portfolio policy and backtest-universe validation helpers."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def build_backtest_universe_validation(
    strategy_name: str,
    strategy_weights: pd.Series,
    benchmark_weights: Mapping[str, dict],
    liquid_tickers: list[str],
) -> pd.DataFrame:
    """Build a validation table for configured strategy and benchmark assets."""
    liquid_set = set(liquid_tickers)
    rows: list[dict[str, object]] = []

    for ticker, weight in strategy_weights.items():
        rows.append(
            {
                "portfolio": strategy_name,
                "ticker": ticker,
                "target_weight": float(weight),
                "passes_liquidity_filter": ticker in liquid_set,
            }
        )

    for benchmark_name, benchmark_config in benchmark_weights.items():
        for ticker, weight in benchmark_config["weights"].items():
            rows.append(
                {
                    "portfolio": benchmark_name,
                    "ticker": ticker,
                    "target_weight": float(weight),
                    "passes_liquidity_filter": ticker in liquid_set,
                }
            )

    validation = pd.DataFrame(rows)
    validation["requires_policy_override"] = ~validation["passes_liquidity_filter"]
    return validation.set_index(["portfolio", "ticker"]).sort_index()


def summarize_backtest_universe_validation(validation_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize liquidity-policy issues by portfolio."""
    grouped = validation_table.groupby(level="portfolio").agg(
        configured_assets=("target_weight", "count"),
        liquidity_passing_assets=("passes_liquidity_filter", "sum"),
        policy_override_assets=("requires_policy_override", "sum"),
    )
    grouped["is_fully_liquid"] = grouped["policy_override_assets"] == 0
    return grouped
