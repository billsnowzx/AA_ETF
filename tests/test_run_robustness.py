import shutil
import uuid
from pathlib import Path

import pandas as pd

from scripts.run_robustness import (
    build_default_stress_start_dates,
    parse_csv_floats,
    parse_csv_strings,
    run_robustness_workflow,
)


def _raw_price_frame(prices: list[float], ticker: str) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "adj_close": prices,
            "volume": [1_000_000] * len(prices),
            "ticker": [ticker] * len(prices),
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


def test_parse_csv_helpers_parse_and_trim_values() -> None:
    assert parse_csv_strings(" monthly,quarterly , yearly ") == ["monthly", "quarterly", "yearly"]
    assert parse_csv_floats("0, 5,10.5") == [0.0, 5.0, 10.5]


def test_build_default_stress_start_dates_uses_earliest_date_per_year() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-03-01", "2021-01-04", "2021-04-01", "2022-01-03"])
    returns = pd.DataFrame({"VTI": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=index)

    starts = build_default_stress_start_dates(returns, max_starts=2)

    assert starts == [pd.Timestamp("2020-01-02"), pd.Timestamp("2021-01-04")]


def test_run_robustness_workflow_writes_scenario_outputs(monkeypatch) -> None:
    output_root = Path("data/cache") / f"test_run_robustness_{uuid.uuid4().hex}"
    raw_dir = output_root / "raw"
    table_dir = output_root / "tables"
    raw_frames = {
        "VTI": _raw_price_frame([100, 101, 102, 103, 104, 105], "VTI"),
        "AGG": _raw_price_frame([100, 100.2, 100.4, 100.5, 100.7, 100.9], "AGG"),
    }

    monkeypatch.setattr(
        "scripts.run_robustness.fetch_prices",
        lambda **kwargs: raw_frames,
    )
    monkeypatch.setattr(
        "scripts.run_robustness.load_portfolio_template",
        lambda *_args, **_kwargs: pd.Series({"VTI": 0.6, "AGG": 0.4}),
    )

    try:
        paths = run_robustness_workflow(
            universe_config="config/etf_universe.yaml",
            portfolio_config="config/portfolio_templates.yaml",
            template_name="balanced",
            start="2020-01-01",
            end="2020-12-31",
            raw_dir=raw_dir,
            output_dir=table_dir,
            rebalance_frequencies=["monthly", "quarterly"],
            one_way_bps_values=[0.0, 5.0],
            stress_start_dates=["2020-01-01"],
            stress_rebalance_frequency="quarterly",
            stress_one_way_bps=5.0,
            download_retries=1,
            download_retry_delay=0.0,
        )

        assert paths["robustness_scenarios"].exists()
        assert paths["start_date_robustness"].exists()

        scenarios = pd.read_csv(paths["robustness_scenarios"], index_col=0)
        stress = pd.read_csv(paths["start_date_robustness"], index_col=0)
        assert "ending_nav" in scenarios.columns
        assert "start_date" in stress.columns
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
