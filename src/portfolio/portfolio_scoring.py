"""Portfolio scoring helpers for Phase 1 strategy and benchmark evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_yaml_file

DEFAULT_PORTFOLIO_SCORING_RULES: dict[str, dict[str, float]] = {
    "return_score": {
        "max_score": 25.0,
        "annualized_return_lower": -0.05,
        "annualized_return_upper": 0.12,
    },
    "risk_control_score": {
        "max_score": 25.0,
        "annualized_volatility_lower": 0.08,
        "annualized_volatility_upper": 0.25,
        "max_drawdown_lower": 0.10,
        "max_drawdown_upper": 0.40,
        "volatility_weight": 0.50,
        "drawdown_weight": 0.50,
    },
    "risk_adjusted_score": {
        "max_score": 20.0,
        "sharpe_lower": 0.0,
        "sharpe_upper": 1.5,
        "sortino_lower": 0.0,
        "sortino_upper": 2.0,
        "calmar_lower": 0.0,
        "calmar_upper": 1.0,
        "sharpe_weight": 0.50,
        "sortino_weight": 0.30,
        "calmar_weight": 0.20,
    },
    "stability_score": {
        "max_score": 15.0,
        "monthly_win_weight": 0.40,
        "annual_win_weight": 0.30,
        "sharpe_stability_weight": 0.30,
    },
    "executability_score": {
        "max_score": 15.0,
        "average_turnover_lower": 0.02,
        "average_turnover_upper": 0.20,
        "transaction_cost_drag_lower": 0.002,
        "transaction_cost_drag_upper": 0.020,
        "turnover_weight": 0.60,
        "transaction_cost_weight": 0.40,
    },
}


def load_portfolio_scoring_rules(config_path: str | Path = "config/scoring_rules.yaml") -> dict[str, dict[str, float]]:
    """Load portfolio scoring thresholds from YAML with defaults filled in."""
    config = load_yaml_file(config_path)
    rules = deepcopy(DEFAULT_PORTFOLIO_SCORING_RULES)
    configured = config.get("portfolio_scoring", {})
    if not isinstance(configured, dict):
        raise ValueError(f"{config_path}: portfolio_scoring must be a mapping.")
    for section, values in configured.items():
        if section not in rules:
            raise ValueError(f"{config_path}: unsupported portfolio_scoring section '{section}'.")
        if not isinstance(values, dict):
            raise ValueError(f"{config_path}: portfolio_scoring.{section} must be a mapping.")
        for key, value in values.items():
            if key not in rules[section]:
                raise ValueError(f"{config_path}: unsupported portfolio_scoring.{section}.{key}.")
            rules[section][key] = float(value)
    return rules


def _portfolio_scoring_rules(scoring_rules: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    rules = deepcopy(DEFAULT_PORTFOLIO_SCORING_RULES)
    if scoring_rules is None:
        return rules
    for section, values in scoring_rules.items():
        if section in rules and isinstance(values, dict):
            for key, value in values.items():
                if key in rules[section]:
                    rules[section][key] = float(value)
    return rules


def _clip_unit(value: float) -> float:
    """Clip a numeric value into the unit interval."""
    return float(min(max(value, 0.0), 1.0))


def _linear_score(value: float, lower: float, upper: float) -> float:
    """Map a value linearly from [lower, upper] into [0, 1]."""
    if upper <= lower:
        raise ValueError("upper must be greater than lower for linear scoring.")
    return _clip_unit((value - lower) / (upper - lower))


def _inverse_linear_score(value: float, lower: float, upper: float) -> float:
    """Map a value linearly from [lower, upper] into [1, 0]."""
    return 1.0 - _linear_score(value, lower=lower, upper=upper)


def monthly_win_rate(returns: pd.Series) -> float:
    """Compute monthly win rate from daily/periodic returns."""
    if returns.empty:
        return 0.0
    monthly = returns.groupby(returns.index.to_period("M")).apply(lambda x: (1.0 + x).prod() - 1.0)
    if monthly.empty:
        return 0.0
    return float((monthly > 0).mean())


def annual_win_rate(returns: pd.Series) -> float:
    """Compute annual win rate from daily/periodic returns."""
    if returns.empty:
        return 0.0
    annual = returns.groupby(returns.index.to_period("Y")).apply(lambda x: (1.0 + x).prod() - 1.0)
    if annual.empty:
        return 0.0
    return float((annual > 0).mean())


def rolling_sharpe_stability_score(rolling_sharpe: pd.Series) -> float:
    """Map rolling Sharpe variability into a stability quality score."""
    clean = pd.to_numeric(rolling_sharpe, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    dispersion = float(clean.std(ddof=1))
    return _inverse_linear_score(dispersion, lower=0.0, upper=1.0)


def build_portfolio_score_summary(
    performance_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    return_table: pd.DataFrame,
    rolling_sharpe_table: pd.DataFrame | None = None,
    scoring_rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a component-based portfolio score summary (0-100)."""
    rules = _portfolio_scoring_rules(scoring_rules)
    def _get_float(frame: pd.DataFrame, row: str, column: str, default: float = 0.0) -> float:
        if column not in frame.columns:
            return float(default)
        value = frame.loc[row, column]
        if pd.isna(value):
            return float(default)
        return float(value)

    rows: list[dict[str, float | str]] = []
    portfolios = [name for name in performance_summary.index if name in turnover_summary.index and name in return_table.columns]
    for portfolio in portfolios:
        ann_return = _get_float(performance_summary, portfolio, "annualized_return", default=0.0)
        ann_vol = _get_float(performance_summary, portfolio, "annualized_volatility", default=0.0)
        max_drawdown = abs(_get_float(performance_summary, portfolio, "max_drawdown", default=0.0))
        sharpe = _get_float(performance_summary, portfolio, "sharpe_ratio", default=0.0)
        sortino = _get_float(performance_summary, portfolio, "sortino_ratio", default=0.0)
        calmar = _get_float(performance_summary, portfolio, "calmar_ratio", default=0.0)
        avg_turnover = _get_float(turnover_summary, portfolio, "average_turnover", default=0.0)
        total_cost_drag = _get_float(turnover_summary, portfolio, "total_transaction_cost_drag", default=0.0)
        returns = pd.to_numeric(return_table[portfolio], errors="coerce").dropna()

        return_rules = rules["return_score"]
        return_score = return_rules["max_score"] * _linear_score(
            ann_return,
            lower=return_rules["annualized_return_lower"],
            upper=return_rules["annualized_return_upper"],
        )

        risk_rules = rules["risk_control_score"]
        vol_quality = _inverse_linear_score(
            ann_vol,
            lower=risk_rules["annualized_volatility_lower"],
            upper=risk_rules["annualized_volatility_upper"],
        )
        drawdown_quality = _inverse_linear_score(
            max_drawdown,
            lower=risk_rules["max_drawdown_lower"],
            upper=risk_rules["max_drawdown_upper"],
        )
        risk_control_score = risk_rules["max_score"] * (
            risk_rules["volatility_weight"] * vol_quality
            + risk_rules["drawdown_weight"] * drawdown_quality
        )

        risk_adjusted_rules = rules["risk_adjusted_score"]
        sharpe_quality = _linear_score(
            sharpe,
            lower=risk_adjusted_rules["sharpe_lower"],
            upper=risk_adjusted_rules["sharpe_upper"],
        )
        sortino_quality = _linear_score(
            sortino,
            lower=risk_adjusted_rules["sortino_lower"],
            upper=risk_adjusted_rules["sortino_upper"],
        )
        calmar_quality = _linear_score(
            calmar,
            lower=risk_adjusted_rules["calmar_lower"],
            upper=risk_adjusted_rules["calmar_upper"],
        )
        risk_adjusted_score = risk_adjusted_rules["max_score"] * (
            risk_adjusted_rules["sharpe_weight"] * sharpe_quality
            + risk_adjusted_rules["sortino_weight"] * sortino_quality
            + risk_adjusted_rules["calmar_weight"] * calmar_quality
        )

        month_win = monthly_win_rate(returns)
        year_win = annual_win_rate(returns)
        rolling_series = (
            rolling_sharpe_table[portfolio]
            if rolling_sharpe_table is not None and portfolio in rolling_sharpe_table.columns
            else pd.Series(dtype=float)
        )
        sharpe_stability = rolling_sharpe_stability_score(rolling_series)
        stability_rules = rules["stability_score"]
        stability_score = stability_rules["max_score"] * (
            stability_rules["monthly_win_weight"] * month_win
            + stability_rules["annual_win_weight"] * year_win
            + stability_rules["sharpe_stability_weight"] * sharpe_stability
        )

        executability_rules = rules["executability_score"]
        turnover_quality = _inverse_linear_score(
            avg_turnover,
            lower=executability_rules["average_turnover_lower"],
            upper=executability_rules["average_turnover_upper"],
        )
        cost_quality = _inverse_linear_score(
            total_cost_drag,
            lower=executability_rules["transaction_cost_drag_lower"],
            upper=executability_rules["transaction_cost_drag_upper"],
        )
        executability_score = executability_rules["max_score"] * (
            executability_rules["turnover_weight"] * turnover_quality
            + executability_rules["transaction_cost_weight"] * cost_quality
        )

        total_score = (
            return_score
            + risk_control_score
            + risk_adjusted_score
            + stability_score
            + executability_score
        )
        max_total_score = sum(float(section["max_score"]) for section in rules.values())
        rows.append(
            {
                "portfolio": portfolio,
                "return_score": return_score,
                "risk_control_score": risk_control_score,
                "risk_adjusted_score": risk_adjusted_score,
                "stability_score": stability_score,
                "executability_score": executability_score,
                "total_score": total_score,
                "score_pct": total_score / max_total_score if max_total_score > 0 else 0.0,
                "monthly_win_rate": month_win,
                "annual_win_rate": year_win,
                "avg_turnover": avg_turnover,
                "total_transaction_cost_drag": total_cost_drag,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "return_score",
                "risk_control_score",
                "risk_adjusted_score",
                "stability_score",
                "executability_score",
                "total_score",
                "score_pct",
                "monthly_win_rate",
                "annual_win_rate",
                "avg_turnover",
                "total_transaction_cost_drag",
                "rank",
            ]
        )
    summary = pd.DataFrame(rows).set_index("portfolio")
    summary["rank"] = summary["total_score"].rank(method="dense", ascending=False).astype(int)
    return summary.sort_values(["total_score", "monthly_win_rate"], ascending=[False, False])
