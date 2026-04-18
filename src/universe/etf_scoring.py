"""Phase 1 ETF summary and scoring helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.universe.universe_builder import build_universe_summary

LIQUIDITY_MAX_SCORE = 25.0
DATA_QUALITY_MAX_SCORE = 10.0
STRATEGY_FIT_MAX_SCORE = 5.0
PHASE1_MAX_SCORE = LIQUIDITY_MAX_SCORE + DATA_QUALITY_MAX_SCORE + STRATEGY_FIT_MAX_SCORE


def _clip_ratio(value: float) -> float:
    """Clip a numeric ratio into the unit interval."""
    if pd.isna(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def score_liquidity_component(
    average_dollar_volume: float,
    recent_pass_ratio: float,
    adv_threshold: float = 50_000_000.0,
) -> float:
    """Score ETF liquidity from average ADV and recent stability."""
    adv_ratio = _clip_ratio(average_dollar_volume / adv_threshold) if adv_threshold > 0 else 0.0
    recent_ratio = _clip_ratio(recent_pass_ratio)
    return LIQUIDITY_MAX_SCORE * (0.5 * adv_ratio + 0.5 * recent_ratio)


def score_data_quality_component(
    observations: float,
    has_sufficient_history: bool,
    target_observations: int = 252,
) -> float:
    """Score ETF data quality from history depth and sufficiency flags."""
    observation_ratio = _clip_ratio(observations / target_observations) if target_observations > 0 else 0.0
    history_flag = 1.0 if bool(has_sufficient_history) else 0.0
    return DATA_QUALITY_MAX_SCORE * (0.5 * observation_ratio + 0.5 * history_flag)


def score_strategy_fit_component(asset_class: object, description: object) -> float:
    """Score whether the ETF has enough metadata to fit the configured universe."""
    has_asset_class = float(pd.notna(asset_class) and str(asset_class).strip() != "")
    has_description = float(pd.notna(description) and str(description).strip() != "")
    return STRATEGY_FIT_MAX_SCORE * (0.5 * has_asset_class + 0.5 * has_description)


def score_etf_universe(
    universe_config_path: str | Path,
    liquidity_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a scored ETF summary for the Phase 1 universe."""
    summary = build_universe_summary(universe_config_path, liquidity_summary=liquidity_summary).copy()

    summary["liquidity_score"] = summary.apply(
        lambda row: score_liquidity_component(
            average_dollar_volume=float(row.get("average_dollar_volume", np.nan)),
            recent_pass_ratio=float(row.get("recent_pass_ratio", np.nan)),
        ),
        axis=1,
    )
    summary["data_quality_score"] = summary.apply(
        lambda row: score_data_quality_component(
            observations=float(row.get("observations", np.nan)),
            has_sufficient_history=bool(row.get("has_sufficient_history", False)),
        ),
        axis=1,
    )
    summary["strategy_fit_score"] = summary.apply(
        lambda row: score_strategy_fit_component(
            asset_class=row.get("asset_class"),
            description=row.get("description"),
        ),
        axis=1,
    )
    summary["phase1_total_score"] = (
        summary["liquidity_score"] + summary["data_quality_score"] + summary["strategy_fit_score"]
    )
    summary["phase1_score_pct"] = summary["phase1_total_score"] / PHASE1_MAX_SCORE
    summary["phase1_rank"] = summary["phase1_total_score"].rank(method="dense", ascending=False).astype(int)
    return summary.sort_values(by=["phase1_total_score", "average_dollar_volume"], ascending=[False, False])
