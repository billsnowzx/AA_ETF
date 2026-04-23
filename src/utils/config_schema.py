"""Schema validation helpers for Phase 1 YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_file


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _require_non_negative_number(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    number = float(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return number


def validate_etf_universe_schema(config: dict[str, Any], config_path: str | Path) -> None:
    """Validate ETF universe YAML shape and required fields."""
    tickers = _require_mapping(config.get("tickers"), f"{config_path}: tickers")
    if not tickers:
        raise ValueError(f"{config_path}: tickers must not be empty.")

    for ticker, metadata in tickers.items():
        _require_mapping(metadata, f"{config_path}: tickers.{ticker}")
        _require_bool(metadata.get("enabled"), f"{config_path}: tickers.{ticker}.enabled")
        _require_string(metadata.get("asset_class"), f"{config_path}: tickers.{ticker}.asset_class")


def validate_portfolio_templates_schema(config: dict[str, Any], config_path: str | Path) -> None:
    """Validate portfolio template YAML shape and required fields."""
    templates = _require_mapping(config.get("templates"), f"{config_path}: templates")
    if not templates:
        raise ValueError(f"{config_path}: templates must not be empty.")

    default_template = _require_string(config.get("default_template"), f"{config_path}: default_template")
    if default_template not in templates:
        raise ValueError(
            f"{config_path}: default_template '{default_template}' must exist in templates."
        )

    for template_name, template_data in templates.items():
        _require_mapping(template_data, f"{config_path}: templates.{template_name}")
        weights = _require_mapping(
            template_data.get("weights"),
            f"{config_path}: templates.{template_name}.weights",
        )
        if not weights:
            raise ValueError(f"{config_path}: templates.{template_name}.weights must not be empty.")
        for ticker, weight in weights.items():
            _require_non_negative_number(
                weight,
                f"{config_path}: templates.{template_name}.weights.{ticker}",
            )


def validate_benchmark_schema(config: dict[str, Any], config_path: str | Path) -> None:
    """Validate benchmark YAML shape and required fields."""
    benchmarks = _require_mapping(config.get("benchmarks"), f"{config_path}: benchmarks")
    if not benchmarks:
        raise ValueError(f"{config_path}: benchmarks must not be empty.")

    for benchmark_name, benchmark_data in benchmarks.items():
        _require_mapping(benchmark_data, f"{config_path}: benchmarks.{benchmark_name}")
        _require_string(
            benchmark_data.get("name"),
            f"{config_path}: benchmarks.{benchmark_name}.name",
        )
        weights = _require_mapping(
            benchmark_data.get("weights"),
            f"{config_path}: benchmarks.{benchmark_name}.weights",
        )
        if not weights:
            raise ValueError(f"{config_path}: benchmarks.{benchmark_name}.weights must not be empty.")
        for ticker, weight in weights.items():
            _require_non_negative_number(
                weight,
                f"{config_path}: benchmarks.{benchmark_name}.weights.{ticker}",
            )


def validate_rebalance_schema(config: dict[str, Any], config_path: str | Path) -> None:
    """Validate rebalance-rule YAML shape and required fields."""
    standard_rebalance = _require_mapping(
        config.get("standard_rebalance"),
        f"{config_path}: standard_rebalance",
    )
    _require_string(
        standard_rebalance.get("frequency"),
        f"{config_path}: standard_rebalance.frequency",
    )
    trigger_mode = _require_string(
        standard_rebalance.get("trigger_mode", "calendar"),
        f"{config_path}: standard_rebalance.trigger_mode",
    ).lower()
    if trigger_mode not in {"calendar", "drift_only", "calendar_or_drift"}:
        raise ValueError(
            f"{config_path}: standard_rebalance.trigger_mode '{trigger_mode}' is unsupported."
        )

    weight_drift_rule = _require_mapping(config.get("weight_drift_rule"), f"{config_path}: weight_drift_rule")
    _require_bool(weight_drift_rule.get("enabled"), f"{config_path}: weight_drift_rule.enabled")
    _require_non_negative_number(
        weight_drift_rule.get("relative_deviation_threshold"),
        f"{config_path}: weight_drift_rule.relative_deviation_threshold",
    )

    transaction_costs = _require_mapping(config.get("transaction_costs"), f"{config_path}: transaction_costs")
    _require_non_negative_number(
        transaction_costs.get("one_way_bps"),
        f"{config_path}: transaction_costs.one_way_bps",
    )

    trend_filter = config.get("trend_filter")
    if trend_filter is not None:
        trend_filter_mapping = _require_mapping(trend_filter, f"{config_path}: trend_filter")
        _require_bool(trend_filter_mapping.get("enabled"), f"{config_path}: trend_filter.enabled")
        _require_non_negative_number(
            trend_filter_mapping.get("moving_average_months"),
            f"{config_path}: trend_filter.moving_average_months",
        )
        reduction_fraction = _require_non_negative_number(
            trend_filter_mapping.get("reduction_fraction"),
            f"{config_path}: trend_filter.reduction_fraction",
        )
        if reduction_fraction > 1.0:
            raise ValueError(f"{config_path}: trend_filter.reduction_fraction must be <= 1.0.")


def validate_phase1_config_files(
    *,
    universe_config_path: str | Path,
    portfolio_config_path: str | Path,
    benchmark_config_path: str | Path,
    rebalance_config_path: str | Path,
) -> None:
    """Validate all Phase 1 config files and raise one aggregated error if invalid."""
    config_paths = {
        "universe": Path(universe_config_path),
        "portfolio_templates": Path(portfolio_config_path),
        "benchmarks": Path(benchmark_config_path),
        "rebalance_rules": Path(rebalance_config_path),
    }
    errors: list[str] = []

    for name, path in config_paths.items():
        if not path.exists():
            errors.append(f"{name}: config file not found at '{path}'.")

    if errors:
        raise ValueError("Config schema validation failed:\n- " + "\n- ".join(errors))

    validators = [
        ("universe", validate_etf_universe_schema),
        ("portfolio_templates", validate_portfolio_templates_schema),
        ("benchmarks", validate_benchmark_schema),
        ("rebalance_rules", validate_rebalance_schema),
    ]

    for name, validator in validators:
        path = config_paths[name]
        try:
            config = load_yaml_file(path)
            validator(config, path)
        except Exception as exc:  # pragma: no cover - exercised via tests
            errors.append(str(exc))

    if errors:
        raise ValueError("Config schema validation failed:\n- " + "\n- ".join(errors))
