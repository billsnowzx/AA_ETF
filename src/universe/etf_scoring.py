"""Phase 1 ETF summary and scoring helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.universe.universe_builder import build_universe_summary
from src.utils.config import load_yaml_file

LIQUIDITY_MAX_SCORE = 25.0
DATA_QUALITY_MAX_SCORE = 10.0
STRATEGY_FIT_MAX_SCORE = 5.0
PHASE1_MAX_SCORE = LIQUIDITY_MAX_SCORE + DATA_QUALITY_MAX_SCORE + STRATEGY_FIT_MAX_SCORE

DEFAULT_ETF_SCORING_RULES: dict[str, dict[str, float]] = {
    "liquidity": {
        "max_score": LIQUIDITY_MAX_SCORE,
        "adv_threshold": 50_000_000.0,
        "average_adv_weight": 0.50,
        "recent_pass_ratio_weight": 0.50,
    },
    "data_quality": {
        "max_score": DATA_QUALITY_MAX_SCORE,
        "target_observations": 252.0,
        "observation_weight": 0.50,
        "sufficient_history_weight": 0.50,
    },
    "strategy_fit": {
        "max_score": STRATEGY_FIT_MAX_SCORE,
        "asset_class_weight": 0.30,
        "description_weight": 0.20,
        "metadata_available_weight": 0.20,
        "expense_ratio_weight": 0.20,
        "total_assets_weight": 0.10,
        "full_credit_expense_ratio": 0.0025,
        "zero_credit_expense_ratio": 0.0100,
        "full_credit_total_assets": 1_000_000_000.0,
    },
}


def load_etf_scoring_rules(config_path: str | Path = "config/scoring_rules.yaml") -> dict[str, dict[str, float]]:
    """Load ETF scoring thresholds from YAML with defaults filled in."""
    config = load_yaml_file(config_path)
    rules = deepcopy(DEFAULT_ETF_SCORING_RULES)
    configured = config.get("etf_scoring", {})
    if not isinstance(configured, dict):
        raise ValueError(f"{config_path}: etf_scoring must be a mapping.")
    for section, values in configured.items():
        if section not in rules:
            raise ValueError(f"{config_path}: unsupported etf_scoring section '{section}'.")
        if not isinstance(values, dict):
            raise ValueError(f"{config_path}: etf_scoring.{section} must be a mapping.")
        for key, value in values.items():
            if key not in rules[section]:
                raise ValueError(f"{config_path}: unsupported etf_scoring.{section}.{key}.")
            rules[section][key] = float(value)
    return rules


def _etf_scoring_rules(scoring_rules: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    rules = deepcopy(DEFAULT_ETF_SCORING_RULES)
    if scoring_rules is None:
        return rules
    for section, values in scoring_rules.items():
        if section in rules and isinstance(values, dict):
            for key, value in values.items():
                if key in rules[section]:
                    rules[section][key] = float(value)
    return rules


def _clip_ratio(value: float) -> float:
    """Clip a numeric ratio into the unit interval."""
    if pd.isna(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def score_liquidity_component(
    average_dollar_volume: float,
    recent_pass_ratio: float,
    adv_threshold: float = 50_000_000.0,
    scoring_rules: dict[str, Any] | None = None,
) -> float:
    """Score ETF liquidity from average ADV and recent stability."""
    rules = _etf_scoring_rules(scoring_rules)["liquidity"]
    adv_threshold = float(rules.get("adv_threshold", adv_threshold))
    adv_ratio = _clip_ratio(average_dollar_volume / adv_threshold) if adv_threshold > 0 else 0.0
    recent_ratio = _clip_ratio(recent_pass_ratio)
    return rules["max_score"] * (
        rules["average_adv_weight"] * adv_ratio
        + rules["recent_pass_ratio_weight"] * recent_ratio
    )


def score_data_quality_component(
    observations: float,
    has_sufficient_history: bool,
    target_observations: int = 252,
    scoring_rules: dict[str, Any] | None = None,
) -> float:
    """Score ETF data quality from history depth and sufficiency flags."""
    rules = _etf_scoring_rules(scoring_rules)["data_quality"]
    target_observations = int(rules.get("target_observations", target_observations))
    observation_ratio = _clip_ratio(observations / target_observations) if target_observations > 0 else 0.0
    history_flag = 1.0 if bool(has_sufficient_history) else 0.0
    return rules["max_score"] * (
        rules["observation_weight"] * observation_ratio
        + rules["sufficient_history_weight"] * history_flag
    )


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


def _expense_ratio_quality_with_rules(expense_ratio: float, rules: dict[str, float]) -> float:
    full_credit = rules["full_credit_expense_ratio"]
    zero_credit = rules["zero_credit_expense_ratio"]
    if expense_ratio <= full_credit:
        return 1.0
    if expense_ratio >= zero_credit:
        return 0.0
    return _clip_ratio((zero_credit - expense_ratio) / (zero_credit - full_credit))


def score_strategy_fit_with_metadata(
    *,
    asset_class: object,
    description: object,
    metadata_available: object,
    expense_ratio: object,
    total_assets: object,
    scoring_rules: dict[str, Any] | None = None,
) -> float:
    """Score ETF strategy fit using config metadata plus fetched ETF metadata."""
    rules = _etf_scoring_rules(scoring_rules)["strategy_fit"]
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
            expense_ratio_score = _expense_ratio_quality_with_rules(float(expense_ratio), rules)
        if pd.isna(total_assets):
            total_assets_score = 0.0
        else:
            total_assets_score = _clip_ratio(float(total_assets) / rules["full_credit_total_assets"])
    else:
        metadata_available_score = 0.0
        expense_ratio_score = 0.0
        total_assets_score = 0.0

    weighted_score = (
        rules["asset_class_weight"] * has_asset_class
        + rules["description_weight"] * has_description
        + rules["metadata_available_weight"] * metadata_available_score
        + rules["expense_ratio_weight"] * expense_ratio_score
        + rules["total_assets_weight"] * total_assets_score
    )
    return rules["max_score"] * weighted_score


def score_etf_universe(
    universe_config_path: str | Path,
    liquidity_summary: pd.DataFrame,
    metadata_summary: pd.DataFrame | None = None,
    scoring_rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a scored ETF summary for the Phase 1 universe."""
    rules = _etf_scoring_rules(scoring_rules)
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
            scoring_rules=rules,
        ),
        axis=1,
    )
    summary["data_quality_score"] = summary.apply(
        lambda row: score_data_quality_component(
            observations=float(row.get("observations", np.nan)),
            has_sufficient_history=bool(row.get("has_sufficient_history", False)),
            scoring_rules=rules,
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
            scoring_rules=rules,
        ),
        axis=1,
    )
    summary["phase1_total_score"] = (
        summary["liquidity_score"] + summary["data_quality_score"] + summary["strategy_fit_score"]
    )
    max_total_score = sum(float(section["max_score"]) for section in rules.values())
    summary["phase1_score_pct"] = summary["phase1_total_score"] / max_total_score if max_total_score > 0 else 0.0
    summary["phase1_rank"] = summary["phase1_total_score"].rank(method="dense", ascending=False).astype(int)
    return summary.sort_values(by=["phase1_total_score", "average_dollar_volume"], ascending=[False, False])
