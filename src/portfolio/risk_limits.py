"""Risk-limit configuration and audit helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_yaml_file


def load_risk_limits(config_path: str | Path) -> dict[str, Any]:
    """Load risk-limit configuration from YAML."""
    config = load_yaml_file(config_path)
    risk_limits = config.get("risk_limits")
    if not isinstance(risk_limits, dict):
        raise ValueError(f"Risk limits config '{config_path}' must define a 'risk_limits' mapping.")

    portfolio_limits = risk_limits.get("portfolio", {})
    if not isinstance(portfolio_limits, dict):
        raise ValueError(f"Risk limits config '{config_path}' must define risk_limits.portfolio as a mapping.")

    annualized_volatility_warning = portfolio_limits.get("annualized_volatility_warning")
    max_drawdown_warning = portfolio_limits.get("max_drawdown_warning")

    return {
        "annualized_volatility_warning": (
            None if annualized_volatility_warning is None else float(annualized_volatility_warning)
        ),
        "max_drawdown_warning": (
            None if max_drawdown_warning is None else float(max_drawdown_warning)
        ),
    }


def build_portfolio_risk_limit_checks(
    performance_summary: pd.DataFrame,
    risk_limits: dict[str, Any],
) -> pd.DataFrame:
    """Build an auditable risk-limit check table for each portfolio row."""
    if performance_summary.empty:
        return pd.DataFrame(
            columns=[
                "portfolio",
                "metric",
                "threshold",
                "observed",
                "comparison_value",
                "limit_enabled",
                "breach",
            ]
        )

    metric_to_threshold_key = {
        "annualized_volatility": "annualized_volatility_warning",
        "max_drawdown": "max_drawdown_warning",
    }

    rows: list[dict[str, object]] = []
    for portfolio in performance_summary.index:
        for metric, threshold_key in metric_to_threshold_key.items():
            if metric not in performance_summary.columns:
                continue
            threshold = risk_limits.get(threshold_key)
            observed = float(performance_summary.loc[portfolio, metric])
            comparison_value = abs(observed) if metric == "max_drawdown" else observed
            limit_enabled = threshold is not None
            breach = bool(limit_enabled and comparison_value > float(threshold))
            rows.append(
                {
                    "rule_id": f"{portfolio}:{metric}",
                    "portfolio": portfolio,
                    "metric": metric,
                    "threshold": threshold,
                    "observed": observed,
                    "comparison_value": comparison_value,
                    "limit_enabled": limit_enabled,
                    "breach": breach,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "portfolio",
                "metric",
                "threshold",
                "observed",
                "comparison_value",
                "limit_enabled",
                "breach",
            ]
        )

    return pd.DataFrame(rows).set_index("rule_id")


def find_risk_limit_breaches(risk_limit_checks: pd.DataFrame) -> pd.DataFrame:
    """Return risk-limit rows that are enabled and breached."""
    if risk_limit_checks.empty:
        return risk_limit_checks.copy()
    if "breach" not in risk_limit_checks.columns:
        return pd.DataFrame(columns=risk_limit_checks.columns)
    return risk_limit_checks.loc[risk_limit_checks["breach"].astype(bool)].copy()


def build_risk_limit_breach_summary(risk_limit_checks: pd.DataFrame) -> pd.DataFrame:
    """Build a compact breach summary table by portfolio plus an overall row."""
    if risk_limit_checks.empty or "portfolio" not in risk_limit_checks.columns:
        return pd.DataFrame(
            columns=[
                "portfolio",
                "total_enabled_checks",
                "breached_checks",
                "breach_ratio",
            ]
        )

    checks = risk_limit_checks.copy()
    enabled_mask = checks.get("limit_enabled", False).astype(bool)
    checks = checks.loc[enabled_mask]

    rows: list[dict[str, object]] = []
    grouped = checks.groupby("portfolio") if not checks.empty else []
    for portfolio, frame in grouped:
        total_enabled = int(len(frame))
        breached = int(frame["breach"].astype(bool).sum()) if "breach" in frame.columns else 0
        rows.append(
            {
                "portfolio": portfolio,
                "total_enabled_checks": total_enabled,
                "breached_checks": breached,
                "breach_ratio": float(breached / total_enabled) if total_enabled > 0 else 0.0,
            }
        )

    total_enabled_overall = int(len(checks))
    breached_overall = int(checks["breach"].astype(bool).sum()) if "breach" in checks.columns else 0
    rows.append(
        {
            "portfolio": "overall",
            "total_enabled_checks": total_enabled_overall,
            "breached_checks": breached_overall,
            "breach_ratio": (
                float(breached_overall / total_enabled_overall) if total_enabled_overall > 0 else 0.0
            ),
        }
    )

    return pd.DataFrame(rows).set_index("portfolio")
