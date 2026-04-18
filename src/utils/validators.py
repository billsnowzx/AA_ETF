"""Validation helpers for config-driven portfolio inputs."""

from __future__ import annotations

from collections.abc import Mapping


def validate_long_only_weights(weights: Mapping[str, float]) -> None:
    """Validate that a portfolio weight mapping is non-empty and long-only."""
    if not weights:
        raise ValueError("Weight mapping cannot be empty.")

    negative_weights = {ticker: weight for ticker, weight in weights.items() if weight < 0}
    if negative_weights:
        raise ValueError(f"Negative weights are not supported in Phase 1: {negative_weights}")


def validate_weight_sum(
    weights: Mapping[str, float],
    target_sum: float = 1.0,
    tolerance: float = 1e-8,
) -> None:
    """Validate that portfolio weights sum to the expected total."""
    total_weight = float(sum(weights.values()))
    if abs(total_weight - target_sum) > tolerance:
        raise ValueError(
            f"Portfolio weights must sum to {target_sum}, received {total_weight:.10f}."
        )
