"""Deterministic rebalance rule helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.validators import validate_long_only_weights
from src.utils.config import load_yaml_file


def load_rebalance_rules(config_path: str | Path) -> dict:
    """Load rebalance-rule configuration from YAML."""
    config = load_yaml_file(config_path)
    if "weight_drift_rule" not in config or "standard_rebalance" not in config:
        raise ValueError(
            f"Rebalance config '{config_path}' must define 'standard_rebalance' and 'weight_drift_rule'."
        )
    return config


def load_standard_rebalance_frequency(config_path: str | Path) -> str:
    """Load the configured standard rebalance frequency."""
    config = load_rebalance_rules(config_path)
    frequency = config["standard_rebalance"].get("frequency")
    if not frequency:
        raise ValueError(f"Rebalance config '{config_path}' must define standard_rebalance.frequency.")
    return str(frequency)


def load_rebalance_trigger_mode(config_path: str | Path) -> str:
    """Load rebalance trigger mode from config."""
    config = load_rebalance_rules(config_path)
    mode = str(config["standard_rebalance"].get("trigger_mode", "calendar")).lower()
    supported_modes = {"calendar", "drift_only", "calendar_or_drift"}
    if mode not in supported_modes:
        raise ValueError(
            f"Unsupported standard_rebalance.trigger_mode '{mode}'. Supported values: {sorted(supported_modes)}."
        )
    return mode


def load_drift_rule_enabled(config_path: str | Path) -> bool:
    """Load whether drift-based rebalance checks are enabled."""
    config = load_rebalance_rules(config_path)
    return bool(config["weight_drift_rule"].get("enabled", False))


def load_relative_drift_threshold(config_path: str | Path) -> float:
    """Load the configured relative drift threshold."""
    config = load_rebalance_rules(config_path)
    drift_rule = config["weight_drift_rule"]
    if "relative_deviation_threshold" not in drift_rule:
        raise ValueError(
            f"Rebalance config '{config_path}' must define weight_drift_rule.relative_deviation_threshold."
        )
    return float(drift_rule["relative_deviation_threshold"])


def load_trend_filter_settings(config_path: str | Path) -> dict[str, bool | int | float]:
    """Load and validate trend-filter settings from rebalance config."""
    config = load_rebalance_rules(config_path)
    trend_filter = config.get("trend_filter", {})

    enabled = bool(trend_filter.get("enabled", False))
    moving_average_months = int(trend_filter.get("moving_average_months", 10))
    reduction_fraction = float(trend_filter.get("reduction_fraction", 0.50))

    if moving_average_months < 1:
        raise ValueError("trend_filter.moving_average_months must be >= 1.")
    if reduction_fraction < 0.0 or reduction_fraction > 1.0:
        raise ValueError("trend_filter.reduction_fraction must be between 0 and 1.")

    return {
        "enabled": enabled,
        "moving_average_months": moving_average_months,
        "reduction_fraction": reduction_fraction,
    }


def weight_drift_table(
    target_weights: dict[str, float] | pd.Series,
    current_weights: dict[str, float] | pd.Series,
    relative_deviation_threshold: float = 0.20,
) -> pd.DataFrame:
    """Build an auditable table of target weights, current weights, and drift status."""
    target = pd.Series(target_weights, dtype=float)
    current = pd.Series(current_weights, dtype=float)
    validate_long_only_weights(target.to_dict())
    validate_long_only_weights(current.to_dict())

    aligned_index = target.index.union(current.index)
    target = target.reindex(aligned_index, fill_value=0.0)
    current = current.reindex(aligned_index, fill_value=0.0)

    absolute_deviation = (current - target).abs()
    allowed_deviation = target.abs() * relative_deviation_threshold
    relative_deviation = pd.Series(float("nan"), index=target.index, dtype=float)

    nonzero_target = target != 0
    relative_deviation.loc[nonzero_target] = (
        absolute_deviation.loc[nonzero_target] / target.loc[nonzero_target].abs()
    )
    relative_deviation.loc[~nonzero_target] = float("inf")

    table = pd.DataFrame(
        {
            "target_weight": target,
            "current_weight": current,
            "absolute_deviation": absolute_deviation,
            "allowed_deviation": allowed_deviation,
            "relative_deviation": relative_deviation,
            "breach": absolute_deviation > allowed_deviation,
        }
    )
    return table


def breached_rebalance_assets(
    target_weights: dict[str, float] | pd.Series,
    current_weights: dict[str, float] | pd.Series,
    relative_deviation_threshold: float = 0.20,
) -> list[str]:
    """Return the assets whose weight drift breaches the configured threshold."""
    drift = weight_drift_table(
        target_weights=target_weights,
        current_weights=current_weights,
        relative_deviation_threshold=relative_deviation_threshold,
    )
    return drift.index[drift["breach"]].tolist()


def should_rebalance_by_drift(
    target_weights: dict[str, float] | pd.Series,
    current_weights: dict[str, float] | pd.Series,
    relative_deviation_threshold: float = 0.20,
) -> bool:
    """Return whether any asset breaches the configured drift threshold."""
    breached_assets = breached_rebalance_assets(
        target_weights=target_weights,
        current_weights=current_weights,
        relative_deviation_threshold=relative_deviation_threshold,
    )
    return bool(breached_assets)


def should_rebalance_by_config(
    target_weights: dict[str, float] | pd.Series,
    current_weights: dict[str, float] | pd.Series,
    config_path: str | Path,
) -> bool:
    """Return whether drift-based rebalance logic is triggered using YAML config."""
    config = load_rebalance_rules(config_path)
    drift_rule = config["weight_drift_rule"]
    if not drift_rule.get("enabled", False):
        return False

    return should_rebalance_by_drift(
        target_weights=target_weights,
        current_weights=current_weights,
        relative_deviation_threshold=float(drift_rule["relative_deviation_threshold"]),
    )
