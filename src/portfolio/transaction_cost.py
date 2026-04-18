"""Transaction cost and turnover helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.portfolio.weights import normalize_weights
from src.utils.config import load_yaml_file


def align_weight_vectors(
    target_weights: dict[str, float] | pd.Series,
    current_weights: dict[str, float] | pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Align target and current weights across the union of assets."""
    target = normalize_weights(target_weights)

    if current_weights is None:
        current = pd.Series(0.0, index=target.index, dtype=float)
    else:
        current = pd.Series(current_weights, dtype=float)
        aligned_index = target.index.union(current.index)
        target = target.reindex(aligned_index, fill_value=0.0)
        current = current.reindex(aligned_index, fill_value=0.0)

        current_total = float(current.sum())
        if current_total < 0:
            raise ValueError("Current weights must not sum to a negative value.")
        if current_total > 0:
            current = normalize_weights(current)

    return target, current


def turnover_traded_weight(
    target_weights: dict[str, float] | pd.Series,
    current_weights: dict[str, float] | pd.Series | None = None,
) -> float:
    """Compute gross traded portfolio weight required to move to target weights."""
    target, current = align_weight_vectors(target_weights, current_weights=current_weights)
    return float((target - current).abs().sum())


def transaction_cost_drag(
    target_weights: dict[str, float] | pd.Series,
    current_weights: dict[str, float] | pd.Series | None = None,
    one_way_bps: float = 5.0,
) -> float:
    """Compute transaction cost drag from gross traded weight and one-way bps."""
    turnover = turnover_traded_weight(target_weights, current_weights=current_weights)
    return turnover * (one_way_bps / 10_000.0)


def load_one_way_transaction_cost_bps(config_path: str | Path) -> float:
    """Load the default one-way transaction cost assumption from YAML."""
    config = load_yaml_file(config_path)
    transaction_costs = config.get("transaction_costs")
    if not transaction_costs or "one_way_bps" not in transaction_costs:
        raise ValueError(f"Rebalance rules config '{config_path}' must define transaction_costs.one_way_bps.")

    return float(transaction_costs["one_way_bps"])
