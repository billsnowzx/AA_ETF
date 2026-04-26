import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.backtest.stress_test import run_start_date_robustness, write_start_date_robustness


def _toy_asset_returns() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "VTI": [0.01, -0.004, 0.005, 0.003, -0.002, 0.006, -0.001, 0.002, 0.001, 0.004],
            "AGG": [0.002, 0.001, -0.001, 0.001, 0.0, 0.001, 0.001, 0.0, 0.001, 0.001],
        },
        index=index,
    )


def test_run_start_date_robustness_returns_expected_shape_and_order() -> None:
    scenarios = run_start_date_robustness(
        asset_returns=_toy_asset_returns(),
        target_weights={"VTI": 0.6, "AGG": 0.4},
        start_dates=["2024-01-01", "2024-01-05", "2024-01-01"],
        rebalance_frequency="quarterly",
        one_way_bps=5.0,
        periods_per_year=252,
    )

    assert scenarios.index.tolist() == ["start=2024-01-01", "start=2024-01-05"]
    assert scenarios.columns.tolist() == [
        "start_date",
        "end_date",
        "observations",
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
    assert int(scenarios.loc["start=2024-01-01", "observations"]) > int(
        scenarios.loc["start=2024-01-05", "observations"]
    )


def test_run_start_date_robustness_rejects_too_short_history() -> None:
    returns = _toy_asset_returns()
    late_start = (returns.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        run_start_date_robustness(
            asset_returns=returns,
            target_weights={"VTI": 0.6, "AGG": 0.4},
            start_dates=[late_start],
        )
    except ValueError as exc:
        assert "Insufficient history" in str(exc)
    else:
        raise AssertionError("Expected ValueError for insufficient start-date history.")


def test_write_start_date_robustness_persists_csv() -> None:
    scenarios = run_start_date_robustness(
        asset_returns=_toy_asset_returns(),
        target_weights={"VTI": 0.6, "AGG": 0.4},
        start_dates=["2024-01-01"],
        rebalance_frequency="monthly",
        one_way_bps=0.0,
        periods_per_year=252,
    )
    output_dir = Path("data/cache") / f"test_start_date_robustness_{uuid.uuid4().hex}"
    try:
        path = write_start_date_robustness(scenarios, output_dir=output_dir)
        loaded = pd.read_csv(path, index_col=0)

        assert path.exists()
        assert loaded.loc["start=2024-01-01", "rebalance_frequency"] == "monthly"
        assert float(loaded.loc["start=2024-01-01", "one_way_bps"]) == 0.0
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
