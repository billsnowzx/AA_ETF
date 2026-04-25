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
    return score_strategy_fit_with_metadata(
        asset_class=asset_class,
        description=description,
        metadata_available=None,
        expense_ratio=None,
        total_assets=None,
    )


def _expense_ratio_quality(expense_ratio: float) -> float:
    """Score expense ratio where lower is better; 25 bps or below receives full credit."""
    if expense_ratio <= 0.0025:
        return 1.0
    if expense_ratio >= 0.01:
        return 0.0
    return _clip_ratio((0.01 - expense_ratio) / (0.01 - 0.0025))


def score_strategy_fit_with_metadata(
    *,
    asset_class: object,
    description: object,
    metadata_available: object,
    expense_ratio: object,
    total_assets: object,
) -> float:
    """Score ETF strategy fit using config metadata plus fetched ETF metadata."""
    has_asset_class = float(pd.notna(asset_class) and str(asset_class).strip() != "")
    has_description = float(pd.notna(description) and str(description).strip() != "")

    # Backward compatibility: when metadata is not provided, keep neutral/full metadata scores.
    if metadata_available is None:
        metadata_available_score = 1.0
        expense_ratio_score = 1.0
        total_assets_score = 1.0
    elif bool(metadata_available):
        metadata_available_score = 1.0
        if pd.isna(expense_ratio):
            expense_ratio_score = 0.0
        else:
            expense_ratio_score = _expense_ratio_quality(float(expense_ratio))
        if pd.isna(total_assets):
            total_assets_score = 0.0
        else:
            total_assets_score = _clip_ratio(float(total_assets) / 1_000_000_000.0)
    else:
        metadata_available_score = 0.0
        expense_ratio_score = 0.0
        total_assets_score = 0.0

    weighted_score = (
        0.30 * has_asset_class
        + 0.20 * has_description
        + 0.20 * metadata_available_score
        + 0.20 * expense_ratio_score
        + 0.10 * total_assets_score
    )
    return STRATEGY_FIT_MAX_SCORE * weighted_score


def score_etf_universe(
    universe_config_path: str | Path,
    liquidity_summary: pd.DataFrame,
    metadata_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a scored ETF summary for the Phase 1 universe."""
    summary = build_universe_summary(universe_config_path, liquidity_summary=liquidity_summary).copy()
    if metadata_summary is not None and not metadata_summary.empty:
        metadata_columns = [
            column for column in ["metadata_available", "expense_ratio", "total_assets", "long_name", "category"]
            if column in metadata_summary.columns
        ]
        metadata_view = metadata_summary[metadata_columns].copy()
        summary = summary.join(metadata_view, how="left", rsuffix="_metadata")
        if "long_name" in summary.columns:
            summary["long_name"] = summary["long_name"].fillna("")
        if "category" in summary.columns:
            summary["category"] = summary["category"].fillna("")

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
        lambda row: score_strategy_fit_with_metadata(
            asset_class=row.get("asset_class"),
            description=row.get("description"),
            metadata_available=row.get("metadata_available"),
            expense_ratio=row.get("expense_ratio"),
            total_assets=row.get("total_assets"),
        ),
        axis=1,
    )
    summary["phase1_total_score"] = (
        summary["liquidity_score"] + summary["data_quality_score"] + summary["strategy_fit_score"]
    )
    summary["phase1_score_pct"] = summary["phase1_total_score"] / PHASE1_MAX_SCORE
    summary["phase1_rank"] = summary["phase1_total_score"].rank(method="dense", ascending=False).astype(int)
    return summary.sort_values(by=["phase1_total_score", "average_dollar_volume"], ascending=[False, False])
