import math
import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.portfolio.risk_limits import (
    build_risk_limit_breach_summary,
    build_portfolio_risk_limit_checks,
    find_risk_limit_breaches,
    load_risk_limits,
)


def test_load_risk_limits_reads_nullable_thresholds() -> None:
    limits = load_risk_limits("config/risk_limits.yaml")
    assert "annualized_volatility_warning" in limits
    assert "max_drawdown_warning" in limits
    assert limits["annualized_volatility_warning"] is None
    assert limits["max_drawdown_warning"] is None


def test_build_portfolio_risk_limit_checks_flags_breaches() -> None:
    performance_summary = pd.DataFrame(
        {
            "annualized_volatility": [0.15, 0.30],
            "max_drawdown": [-0.10, -0.35],
        },
        index=pd.Index(["balanced", "benchmark_a"], name="portfolio"),
    )
    risk_limits = {
        "annualized_volatility_warning": 0.20,
        "max_drawdown_warning": 0.25,
    }

    checks = build_portfolio_risk_limit_checks(performance_summary, risk_limits)
    breaches = find_risk_limit_breaches(checks)
    summary = build_risk_limit_breach_summary(checks)

    assert bool(checks.loc["balanced:annualized_volatility", "breach"]) is False
    assert bool(checks.loc["benchmark_a:annualized_volatility", "breach"]) is True
    assert bool(checks.loc["benchmark_a:max_drawdown", "breach"]) is True
    assert checks.loc["benchmark_a:max_drawdown", "comparison_value"] == 0.35
    assert math.isclose(float(checks.loc["balanced:max_drawdown", "comparison_value"]), 0.10, rel_tol=1e-9)
    assert sorted(breaches.index.tolist()) == [
        "benchmark_a:annualized_volatility",
        "benchmark_a:max_drawdown",
    ]
    assert int(summary.loc["balanced", "total_enabled_checks"]) == 2
    assert int(summary.loc["balanced", "breached_checks"]) == 0
    assert int(summary.loc["benchmark_a", "breached_checks"]) == 2
    assert math.isclose(float(summary.loc["overall", "breach_ratio"]), 0.5, rel_tol=1e-9)


def test_load_risk_limits_raises_for_missing_root_mapping() -> None:
    root = Path("data/cache") / f"risk_limits_invalid_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "risk_limits.yaml"
    config_path.write_text("not_risk_limits: {}\n", encoding="utf-8")

    try:
        try:
            load_risk_limits(config_path)
        except ValueError as exc:
            assert "risk_limits" in str(exc)
        else:
            raise AssertionError("Expected ValueError when risk_limits mapping is missing.")
    finally:
        shutil.rmtree(root, ignore_errors=True)
