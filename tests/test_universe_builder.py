import math

import pandas as pd

from src.universe.universe_builder import build_universe_summary


def test_build_universe_summary_without_liquidity_returns_metadata_only() -> None:
    summary = build_universe_summary("config/etf_universe.yaml")

    assert list(summary.columns) == ["asset_class", "description"]
    assert summary.index.tolist()[0] == "VTI"


def test_build_universe_summary_merges_liquidity_metrics() -> None:
    liquidity_summary = pd.DataFrame(
        {
            "average_dollar_volume": [60_000_000.0],
            "passes_liquidity_filter": [True],
        },
        index=pd.Index(["VTI"], name="ticker"),
    )

    summary = build_universe_summary("config/etf_universe.yaml", liquidity_summary=liquidity_summary)

    assert math.isclose(summary.loc["VTI", "average_dollar_volume"], 60_000_000.0, rel_tol=1e-9)
    assert bool(summary.loc["VTI", "passes_liquidity_filter"]) is True
    assert pd.isna(summary.loc["VEA", "average_dollar_volume"])
