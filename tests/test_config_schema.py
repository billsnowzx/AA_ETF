import shutil
import uuid
from pathlib import Path

import pytest
import yaml

from src.utils.config_schema import validate_phase1_config_files


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _valid_universe() -> dict:
    return {
        "tickers": {
            "VTI": {"enabled": True, "asset_class": "us_equity"},
            "AGG": {"enabled": True, "asset_class": "us_bond"},
        }
    }


def _valid_templates() -> dict:
    return {
        "default_template": "balanced",
        "templates": {
            "balanced": {"weights": {"VTI": 0.6, "AGG": 0.4}},
        },
    }


def _valid_benchmarks() -> dict:
    return {
        "benchmarks": {
            "benchmark_a": {"name": "60/40", "weights": {"VTI": 0.6, "AGG": 0.4}},
        }
    }


def _valid_rebalance() -> dict:
    return {
        "standard_rebalance": {"frequency": "quarterly", "trigger_mode": "calendar"},
        "weight_drift_rule": {"enabled": True, "relative_deviation_threshold": 0.2},
        "transaction_costs": {"one_way_bps": 5},
        "trend_filter": {"enabled": False, "moving_average_months": 10, "reduction_fraction": 0.5},
    }


def _valid_risk_limits() -> dict:
    return {
        "risk_limits": {
            "portfolio": {
                "annualized_volatility_warning": 0.25,
                "max_drawdown_warning": 0.30,
            },
            "liquidity": {
                "minimum_average_daily_dollar_volume": 50_000_000,
                "recent_liquidity_pass_ratio": 0.80,
            },
        }
    }


def test_validate_phase1_config_files_accepts_valid_files() -> None:
    root = Path("data/cache") / f"config_schema_valid_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    universe = root / "etf_universe.yaml"
    templates = root / "portfolio_templates.yaml"
    benchmarks = root / "benchmark_config.yaml"
    rebalance = root / "rebalance_rules.yaml"
    risk_limits = root / "risk_limits.yaml"

    try:
        _write_yaml(universe, _valid_universe())
        _write_yaml(templates, _valid_templates())
        _write_yaml(benchmarks, _valid_benchmarks())
        _write_yaml(rebalance, _valid_rebalance())
        _write_yaml(risk_limits, _valid_risk_limits())

        validate_phase1_config_files(
            universe_config_path=universe,
            portfolio_config_path=templates,
            benchmark_config_path=benchmarks,
            rebalance_config_path=rebalance,
            risk_limits_config_path=risk_limits,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validate_phase1_config_files_raises_clear_error_for_invalid_schema() -> None:
    root = Path("data/cache") / f"config_schema_invalid_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    universe = root / "etf_universe.yaml"
    templates = root / "portfolio_templates.yaml"
    benchmarks = root / "benchmark_config.yaml"
    rebalance = root / "rebalance_rules.yaml"
    risk_limits = root / "risk_limits.yaml"

    invalid_universe = _valid_universe()
    invalid_universe["tickers"]["VTI"]["enabled"] = "yes"
    invalid_rebalance = _valid_rebalance()
    invalid_rebalance["transaction_costs"]["one_way_bps"] = -1

    try:
        _write_yaml(universe, invalid_universe)
        _write_yaml(templates, _valid_templates())
        _write_yaml(benchmarks, _valid_benchmarks())
        _write_yaml(rebalance, invalid_rebalance)
        _write_yaml(risk_limits, _valid_risk_limits())

        with pytest.raises(ValueError, match="Config schema validation failed"):
            validate_phase1_config_files(
                universe_config_path=universe,
                portfolio_config_path=templates,
                benchmark_config_path=benchmarks,
                rebalance_config_path=rebalance,
                risk_limits_config_path=risk_limits,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validate_phase1_config_files_raises_for_invalid_risk_limits_schema() -> None:
    root = Path("data/cache") / f"config_schema_invalid_risk_limits_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    universe = root / "etf_universe.yaml"
    templates = root / "portfolio_templates.yaml"
    benchmarks = root / "benchmark_config.yaml"
    rebalance = root / "rebalance_rules.yaml"
    risk_limits = root / "risk_limits.yaml"

    invalid_risk_limits = _valid_risk_limits()
    invalid_risk_limits["risk_limits"]["liquidity"]["recent_liquidity_pass_ratio"] = 1.2

    try:
        _write_yaml(universe, _valid_universe())
        _write_yaml(templates, _valid_templates())
        _write_yaml(benchmarks, _valid_benchmarks())
        _write_yaml(rebalance, _valid_rebalance())
        _write_yaml(risk_limits, invalid_risk_limits)

        with pytest.raises(ValueError, match="recent_liquidity_pass_ratio"):
            validate_phase1_config_files(
                universe_config_path=universe,
                portfolio_config_path=templates,
                benchmark_config_path=benchmarks,
                rebalance_config_path=rebalance,
                risk_limits_config_path=risk_limits,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)
