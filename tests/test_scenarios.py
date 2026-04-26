import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.backtest.scenarios import run_robustness_scenarios, write_robustness_scenarios


def _toy_asset_returns() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    return pd.DataFrame(
        {
            "VTI": [0.01, -0.005, 0.004, 0.003, -0.002, 0.006, -0.001, 0.002],
            "AGG": [0.002, 0.001, -0.001, 0.001, 0.0, 0.001, 0.001, 0.0],
        },
        index=index,
    )


def test_run_robustness_scenarios_returns_expected_rows_and_columns() -> None:
    scenarios = run_robustness_scenarios(
        asset_returns=_toy_asset_returns(),
        target_weights={"VTI": 0.6, "AGG": 0.4},
        rebalance_frequencies=["quarterly", "monthly"],
        one_way_bps_values=[0.0, 5.0],
        periods_per_year=252,
    )

    assert len(scenarios) == 4
    assert scenarios.index.name == "scenario_id"
    assert scenarios.columns.tolist() == [
        "rebalance_frequency",
        "one_way_bps",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "ending_nav",
        "total_turnover",
        "total_transaction_cost_drag",
    ]
    assert scenarios["rebalance_frequency"].tolist() == ["monthly", "monthly", "quarterly", "quarterly"]
    assert sorted(scenarios["one_way_bps"].unique().tolist()) == [0.0, 5.0]


def test_run_robustness_scenarios_higher_costs_reduce_ending_nav() -> None:
    scenarios = run_robustness_scenarios(
        asset_returns=_toy_asset_returns(),
        target_weights={"VTI": 0.6, "AGG": 0.4},
        rebalance_frequencies=["monthly"],
        one_way_bps_values=[0.0, 20.0],
        periods_per_year=252,
    )
    no_cost_nav = float(scenarios.loc["frequency=monthly|cost_bps=0.00", "ending_nav"])
    high_cost_nav = float(scenarios.loc["frequency=monthly|cost_bps=20.00", "ending_nav"])

    assert high_cost_nav < no_cost_nav


def test_write_robustness_scenarios_persists_csv() -> None:
    scenarios = run_robustness_scenarios(
        asset_returns=_toy_asset_returns(),
        target_weights={"VTI": 0.6, "AGG": 0.4},
        rebalance_frequencies=["quarterly"],
        one_way_bps_values=[5.0],
        periods_per_year=252,
    )
    output_dir = Path("data/cache") / f"test_scenarios_{uuid.uuid4().hex}"
    try:
        path = write_robustness_scenarios(scenarios, output_dir=output_dir)
        loaded = pd.read_csv(path, index_col=0)

        assert path.exists()
        assert "ending_nav" in loaded.columns
        assert "frequency=quarterly|cost_bps=5.00" in loaded.index
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
