"""Start-date robustness helpers for fixed-weight backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.backtest.engine import run_fixed_weight_backtest


def run_start_date_robustness(
    asset_returns: pd.DataFrame,
    target_weights: dict[str, float] | pd.Series,
    start_dates: list[str | pd.Timestamp],
    rebalance_frequency: str = "quarterly",
    one_way_bps: float = 5.0,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Run deterministic start-date sensitivity backtests and return summary metrics."""
    if asset_returns.empty:
        raise ValueError("Asset return matrix cannot be empty.")
    if not start_dates:
        raise ValueError("start_dates cannot be empty.")

    sorted_asset_returns = asset_returns.sort_index()
    unique_start_dates = sorted({pd.Timestamp(value) for value in start_dates})
    rows: list[dict[str, float | str | int]] = []

    for start_date in unique_start_dates:
        scenario_returns = sorted_asset_returns.loc[sorted_asset_returns.index >= start_date]
        if len(scenario_returns) < 2:
            raise ValueError(
                f"Insufficient history from start date {start_date.date()} for backtest. Need at least 2 rows."
            )

        scenario_benchmarks: dict[str, pd.Series] | None = None
        if benchmark_returns is not None:
            scenario_benchmarks = {
                name: series.loc[series.index >= start_date]
                for name, series in benchmark_returns.items()
            }

        result = run_fixed_weight_backtest(
            asset_returns=scenario_returns,
            target_weights=target_weights,
            rebalance_frequency=rebalance_frequency,
            one_way_bps=one_way_bps,
            benchmark_returns=scenario_benchmarks,
            periods_per_year=periods_per_year,
        )
        summary = result["summary"]
        turnover = result["turnover"]
        transaction_costs = result["transaction_costs"]
        nav = result["portfolio_nav"]

        if not isinstance(summary, pd.Series):
            raise ValueError("Backtest summary must be a pandas Series for start-date robustness aggregation.")
        if not isinstance(turnover, pd.Series) or not isinstance(transaction_costs, pd.Series) or not isinstance(nav, pd.Series):
            raise ValueError("Backtest outputs must include Series turnover, transaction_costs, and portfolio_nav.")

        rows.append(
            {
                "scenario_id": f"start={start_date.strftime('%Y-%m-%d')}",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": scenario_returns.index.max().strftime("%Y-%m-%d"),
                "observations": int(len(scenario_returns)),
                "rebalance_frequency": rebalance_frequency,
                "one_way_bps": float(one_way_bps),
                "annualized_return": float(summary["annualized_return"]),
                "annualized_volatility": float(summary["annualized_volatility"]),
                "sharpe_ratio": float(summary["sharpe_ratio"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "calmar_ratio": float(summary["calmar_ratio"]),
                "ending_nav": float(nav.iloc[-1]),
                "total_turnover": float(turnover.sum()),
                "total_transaction_cost_drag": float(transaction_costs.sum()),
            }
        )

    scenario_table = pd.DataFrame(rows).set_index("scenario_id")
    return scenario_table.sort_values("start_date")


def write_start_date_robustness(
    scenario_table: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "start_date_robustness.csv",
) -> Path:
    """Persist start-date robustness results as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    scenario_table.to_csv(path, index=True)
    return path
