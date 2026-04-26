"""Robustness scenario sweeps for fixed-weight backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.backtest.engine import run_fixed_weight_backtest


def run_robustness_scenarios(
    asset_returns: pd.DataFrame,
    target_weights: dict[str, float] | pd.Series,
    rebalance_frequencies: list[str],
    one_way_bps_values: list[float],
    benchmark_returns: Mapping[str, pd.Series] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Run a deterministic scenario grid and return an auditable summary table."""
    if asset_returns.empty:
        raise ValueError("Asset return matrix cannot be empty.")
    if not rebalance_frequencies:
        raise ValueError("rebalance_frequencies cannot be empty.")
    if not one_way_bps_values:
        raise ValueError("one_way_bps_values cannot be empty.")

    frequency_values = sorted({str(value).lower() for value in rebalance_frequencies})
    cost_values = sorted({float(value) for value in one_way_bps_values})
    rows: list[dict[str, float | str]] = []

    for frequency in frequency_values:
        for one_way_bps in cost_values:
            result = run_fixed_weight_backtest(
                asset_returns=asset_returns,
                target_weights=target_weights,
                rebalance_frequency=frequency,
                one_way_bps=one_way_bps,
                benchmark_returns=benchmark_returns,
                periods_per_year=periods_per_year,
            )

            summary = result["summary"]
            if not isinstance(summary, pd.Series):
                raise ValueError("Backtest summary must be a pandas Series for scenario aggregation.")
            turnover = result["turnover"]
            transaction_costs = result["transaction_costs"]
            nav = result["portfolio_nav"]
            if not isinstance(turnover, pd.Series) or not isinstance(transaction_costs, pd.Series) or not isinstance(nav, pd.Series):
                raise ValueError("Scenario backtest outputs must include Series turnover, transaction_costs, and portfolio_nav.")

            rows.append(
                {
                    "scenario_id": f"frequency={frequency}|cost_bps={one_way_bps:.2f}",
                    "rebalance_frequency": frequency,
                    "one_way_bps": one_way_bps,
                    "annualized_return": float(summary["annualized_return"]),
                    "annualized_volatility": float(summary["annualized_volatility"]),
                    "sharpe_ratio": float(summary["sharpe_ratio"]),
                    "max_drawdown": float(summary["max_drawdown"]),
                    "calmar_ratio": float(summary["calmar_ratio"]),
                    "ending_nav": float(nav.iloc[-1]) if len(nav) > 0 else float("nan"),
                    "total_turnover": float(turnover.sum()),
                    "total_transaction_cost_drag": float(transaction_costs.sum()),
                }
            )

    scenario_table = pd.DataFrame(rows)
    scenario_table = scenario_table.sort_values(["rebalance_frequency", "one_way_bps"], ascending=[True, True])
    scenario_table = scenario_table.set_index("scenario_id")
    return scenario_table


def write_robustness_scenarios(
    scenario_table: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "robustness_scenarios.csv",
) -> Path:
    """Persist scenario sweep results as an auditable CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    scenario_table.to_csv(path, index=True)
    return path
