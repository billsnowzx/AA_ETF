import math

import pandas as pd

from src.universe.liquidity_filter import filter_liquid_universe, summarize_liquidity


def _build_frame(dollar_volumes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(dollar_volumes), freq="B")
    adj_close = pd.Series(100.0, index=index)
    volume = pd.Series(dollar_volumes, index=index) / adj_close
    return pd.DataFrame(
        {
            "adj_close": adj_close,
            "volume": volume,
            "dollar_volume": dollar_volumes,
        },
        index=index,
    )


def test_filter_liquid_universe_passes_obvious_liquid_asset() -> None:
    frames = {"VTI": _build_frame([60_000_000.0] * 120)}

    filtered, summary = filter_liquid_universe(frames)

    assert filtered == ["VTI"]
    assert bool(summary.loc["VTI", "passes_liquidity_filter"]) is True
    assert math.isclose(summary.loc["VTI", "recent_pass_ratio"], 1.0, rel_tol=1e-9)


def test_filter_liquid_universe_fails_on_recent_liquidity_instability() -> None:
    dollar_volumes = ([60_000_000.0] * 59) + ([10_000_000.0] * 61)
    frames = {"IEMG": _build_frame(dollar_volumes)}

    filtered, summary = filter_liquid_universe(frames)

    assert filtered == []
    assert bool(summary.loc["IEMG", "passes_average_volume_threshold"]) is False
    assert bool(summary.loc["IEMG", "passes_liquidity_filter"]) is False


def test_summarize_liquidity_marks_insufficient_history() -> None:
    frames = {"GLD": _build_frame([60_000_000.0] * 40)}

    summary = summarize_liquidity(frames)

    assert bool(summary.loc["GLD", "has_sufficient_history"]) is False
    assert math.isnan(summary.loc["GLD", "recent_pass_ratio"])
