"""Portfolio template loading and weight normalization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.validators import validate_long_only_weights


def normalize_weights(
    weights: dict[str, float] | pd.Series,
    target_sum: float = 1.0,
) -> pd.Series:
    """Normalize a long-only weight vector to the requested total."""
    series = pd.Series(weights, dtype=float)
    if series.empty:
        raise ValueError("Weights cannot be empty.")

    validate_long_only_weights(series.to_dict())

    total_weight = float(series.sum())
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive value.")

    normalized = series / total_weight
    return normalized * target_sum


def load_portfolio_templates(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load portfolio template configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    templates = config.get("templates")
    if not templates:
        raise ValueError(f"No portfolio templates found in '{config_path}'.")

    return templates


def load_portfolio_template(
    config_path: str | Path,
    template_name: str | None = None,
) -> pd.Series:
    """Load one portfolio template and return normalized weights."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    templates = config.get("templates")
    if not templates:
        raise ValueError(f"No portfolio templates found in '{config_path}'.")

    selected_name = template_name or config.get("default_template")
    if selected_name not in templates:
        raise ValueError(f"Unknown portfolio template '{selected_name}'.")

    weights = templates[selected_name].get("weights", {})
    return normalize_weights(weights)
