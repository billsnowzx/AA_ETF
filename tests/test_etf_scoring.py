import math

import pandas as pd

from src.universe.etf_scoring import (
    PHASE1_MAX_SCORE,
    score_data_quality_component,
    score_etf_universe,
    score_liquidity_component,
    score_strategy_fit_component,
)


def test_score_liquidity_component_caps_at_maximum() -> None:
    score = score_liquidity_component(
        average_dollar_volume=100_000_000.0,
        recent_pass_ratio=1.0,
    )

    assert math.isclose(score, 25.0, rel_tol=1e-9)


def test_score_data_quality_component_rewards_history_and_observations() -> None:
    score = score_data_quality_component(observations=252, has_sufficient_history=True)

    assert math.isclose(score, 10.0, rel_tol=1e-9)


def test_score_strategy_fit_component_requires_metadata() -> None:
    score = score_strategy_fit_component(asset_class="us_equity", description="US total market equity")

    assert math.isclose(score, 5.0, rel_tol=1e-9)


def test_score_etf_universe_builds_ranked_summary() -> None:
    liquidity_summary = pd.DataFrame(
        {
            "observations": [300, 100],
            "average_dollar_volume": [80_000_000.0, 20_000_000.0],
            "recent_pass_ratio": [1.0, 0.2],
            "has_sufficient_history": [True, False],
            "passes_average_volume_threshold": [True, False],
            "passes_recent_liquidity_threshold": [True, False],
            "passes_liquidity_filter": [True, False],
        },
        index=pd.Index(["VTI", "VNQ"], name="ticker"),
    )

    scored = score_etf_universe("config/etf_universe.yaml", liquidity_summary)

    assert scored.index.tolist()[0] == "VTI"
    assert math.isclose(scored.loc["VTI", "phase1_score_pct"], 1.0, rel_tol=1e-9)
    assert scored.loc["VNQ", "phase1_total_score"] < PHASE1_MAX_SCORE
    assert scored.loc["VTI", "phase1_rank"] == 1
