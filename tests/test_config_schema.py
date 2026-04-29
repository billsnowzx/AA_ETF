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


def _valid_scoring_rules() -> dict:
    return {
        "portfolio_scoring": {
            "return_score": {
                "max_score": 25.0,
                "annualized_return_lower": -0.05,
                "annualized_return_upper": 0.12,
            }
        },
        "etf_scoring": {
            "liquidity": {
                "max_score": 25.0,
                "adv_threshold": 50_000_000,
            }
        },
    }


def test_validate_phase1_config_files_accepts_valid_files() -> None:
    root = Path("data/cache") / f"config_schema_valid_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    universe = root / "etf_universe.yaml"
    templates = root / "portfolio_templates.yaml"
    benchmarks = root / "benchmark_config.yaml"
    rebalance = root / "rebalance_rules.yaml"
    risk_limits = root / "risk_limits.yaml"
    scoring = root / "scoring_rules.yaml"

    try:
        _write_yaml(universe, _valid_universe())
        _write_yaml(templates, _valid_templates())
        _write_yaml(benchmarks, _valid_benchmarks())
        _write_yaml(rebalance, _valid_rebalance())
        _write_yaml(risk_limits, _valid_risk_limits())
        _write_yaml(scoring, _valid_scoring_rules())

        validate_phase1_config_files(
            universe_config_path=universe,
            portfolio_config_path=templates,
            benchmark_config_path=benchmarks,
            rebalance_config_path=rebalance,
            risk_limits_config_path=risk_limits,
            scoring_config_path=scoring,
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


def test_validate_phase1_config_files_raises_for_unknown_portfolio_ticker() -> None:
    root = Path("data/cache") / f"config_schema_unknown_ticker_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    universe = root / "etf_universe.yaml"
    templates = root / "portfolio_templates.yaml"
    benchmarks = root / "benchmark_config.yaml"
    rebalance = root / "rebalance_rules.yaml"
    risk_limits = root / "risk_limits.yaml"

    invalid_templates = _valid_templates()
    invalid_templates["templates"]["balanced"]["weights"]["MISSING"] = 0.1

    try:
        _write_yaml(universe, _valid_universe())
        _write_yaml(templates, invalid_templates)
        _write_yaml(benchmarks, _valid_benchmarks())
        _write_yaml(rebalance, _valid_rebalance())
        _write_yaml(risk_limits, _valid_risk_limits())

        with pytest.raises(ValueError, match="not enabled in universe"):
            validate_phase1_config_files(
                universe_config_path=universe,
                portfolio_config_path=templates,
                benchmark_config_path=benchmarks,
                rebalance_config_path=rebalance,
                risk_limits_config_path=risk_limits,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validate_phase1_config_files_raises_for_unknown_risk_switch_destination() -> None:
    root = Path("data/cache") / f"config_schema_unknown_destination_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    universe = root / "etf_universe.yaml"
    templates = root / "portfolio_templates.yaml"
    benchmarks = root / "benchmark_config.yaml"
    rebalance = root / "rebalance_rules.yaml"
    risk_limits = root / "risk_limits.yaml"

    invalid_rebalance = _valid_rebalance()
    invalid_rebalance["risk_switch"] = {
        "enabled": False,
        "lookback_days": 20,
        "annualized_volatility_threshold": None,
        "reduction_fraction": 0.5,
        "destination_assets": ["MISSING"],
    }

    try:
        _write_yaml(universe, _valid_universe())
        _write_yaml(templates, _valid_templates())
        _write_yaml(benchmarks, _valid_benchmarks())
        _write_yaml(rebalance, invalid_rebalance)
        _write_yaml(risk_limits, _valid_risk_limits())

        with pytest.raises(ValueError, match="risk_switch.destination_assets"):
            validate_phase1_config_files(
                universe_config_path=universe,
                portfolio_config_path=templates,
                benchmark_config_path=benchmarks,
                rebalance_config_path=rebalance,
                risk_limits_config_path=risk_limits,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)
