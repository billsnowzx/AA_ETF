import math

import pandas as pd

from src.universe.etf_scoring import (
    PHASE1_MAX_SCORE,
    load_etf_scoring_rules,
    score_data_quality_component,
    score_etf_universe,
    score_liquidity_component,
    score_strategy_fit_with_metadata,
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


def test_score_strategy_fit_with_metadata_penalizes_missing_metadata() -> None:
    score = score_strategy_fit_with_metadata(
        asset_class="us_equity",
        description="US total market equity",
        metadata_available=False,
        expense_ratio=None,
        total_assets=None,
    )

    assert score < 5.0
    assert math.isclose(score, 2.5, rel_tol=1e-9)


def test_score_strategy_fit_with_metadata_rewards_low_expense_and_large_assets() -> None:
    score = score_strategy_fit_with_metadata(
        asset_class="us_equity",
        description="US total market equity",
        metadata_available=True,
        expense_ratio=0.0003,
        total_assets=5_000_000_000,
    )

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


def test_score_etf_universe_uses_metadata_summary_when_provided() -> None:
    liquidity_summary = pd.DataFrame(
        {
            "observations": [300, 300],
            "average_dollar_volume": [80_000_000.0, 80_000_000.0],
            "recent_pass_ratio": [1.0, 1.0],
            "has_sufficient_history": [True, True],
            "passes_average_volume_threshold": [True, True],
            "passes_recent_liquidity_threshold": [True, True],
            "passes_liquidity_filter": [True, True],
        },
        index=pd.Index(["VTI", "VNQ"], name="ticker"),
    )
    metadata_summary = pd.DataFrame(
        {
            "metadata_available": [True, False],
            "expense_ratio": [0.0003, None],
            "total_assets": [1_000_000_000, None],
        },
        index=pd.Index(["VTI", "VNQ"], name="ticker"),
    )

    scored = score_etf_universe("config/etf_universe.yaml", liquidity_summary, metadata_summary=metadata_summary)

    assert scored.loc["VTI", "strategy_fit_score"] > scored.loc["VNQ", "strategy_fit_score"]


def test_etf_scoring_rules_can_be_loaded_from_config() -> None:
    rules = load_etf_scoring_rules("config/scoring_rules.yaml")

    assert rules["liquidity"]["adv_threshold"] == 50_000_000.0
    assert rules["strategy_fit"]["full_credit_total_assets"] == 1_000_000_000.0
