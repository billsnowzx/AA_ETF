import math

from src.portfolio.rebalancer import (
    breached_rebalance_assets,
    load_drift_rule_enabled,
    load_rebalance_trigger_mode,
    load_relative_drift_threshold,
    load_standard_rebalance_frequency,
    load_trend_filter_settings,
    should_rebalance_by_config,
    should_rebalance_by_drift,
    weight_drift_table,
)


def test_load_rebalance_rule_values_from_config() -> None:
    frequency = load_standard_rebalance_frequency("config/rebalance_rules.yaml")
    trigger_mode = load_rebalance_trigger_mode("config/rebalance_rules.yaml")
    drift_enabled = load_drift_rule_enabled("config/rebalance_rules.yaml")
    threshold = load_relative_drift_threshold("config/rebalance_rules.yaml")
    trend_filter = load_trend_filter_settings("config/rebalance_rules.yaml")

    assert frequency == "quarterly"
    assert trigger_mode == "calendar"
    assert drift_enabled is True
    assert math.isclose(threshold, 0.20, rel_tol=1e-9)
    assert trend_filter["enabled"] is False
    assert trend_filter["moving_average_months"] == 10
    assert math.isclose(float(trend_filter["reduction_fraction"]), 0.50, rel_tol=1e-9)


def test_weight_drift_table_matches_spec_example() -> None:
    drift = weight_drift_table(
        target_weights={"VTI": 0.10},
        current_weights={"VTI": 0.12},
        relative_deviation_threshold=0.20,
    )

    assert math.isclose(drift.loc["VTI", "allowed_deviation"], 0.02, rel_tol=1e-9)
    assert math.isclose(drift.loc["VTI", "absolute_deviation"], 0.02, rel_tol=1e-9)
    assert bool(drift.loc["VTI", "breach"]) is False


def test_should_rebalance_by_drift_when_asset_breaches_threshold() -> None:
    triggered = should_rebalance_by_drift(
        target_weights={"VTI": 0.10, "AGG": 0.90},
        current_weights={"VTI": 0.13, "AGG": 0.87},
        relative_deviation_threshold=0.20,
    )
    breached = breached_rebalance_assets(
        target_weights={"VTI": 0.10, "AGG": 0.90},
        current_weights={"VTI": 0.13, "AGG": 0.87},
        relative_deviation_threshold=0.20,
    )

    assert triggered is True
    assert breached == ["VTI"]


def test_should_rebalance_by_config_uses_yaml_threshold() -> None:
    triggered = should_rebalance_by_config(
        target_weights={"VTI": 0.10, "AGG": 0.90},
        current_weights={"VTI": 0.121, "AGG": 0.879},
        config_path="config/rebalance_rules.yaml",
    )

    assert triggered is True


def test_new_asset_with_zero_target_weight_is_breach() -> None:
    breached = breached_rebalance_assets(
        target_weights={"VTI": 1.0},
        current_weights={"VTI": 0.95, "GLD": 0.05},
        relative_deviation_threshold=0.20,
    )

    assert "GLD" in breached
