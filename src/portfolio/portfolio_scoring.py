"""Portfolio scoring helpers for Phase 1 strategy and benchmark evaluation."""

from __future__ import annotations

import pandas as pd


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
) -> pd.DataFrame:
    """Build a component-based portfolio score summary (0-100)."""
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

        return_score = 25.0 * _linear_score(ann_return, lower=-0.05, upper=0.12)
        vol_quality = _inverse_linear_score(ann_vol, lower=0.08, upper=0.25)
        drawdown_quality = _inverse_linear_score(max_drawdown, lower=0.10, upper=0.40)
        risk_control_score = 25.0 * (0.5 * vol_quality + 0.5 * drawdown_quality)

        sharpe_quality = _linear_score(sharpe, lower=0.0, upper=1.5)
        sortino_quality = _linear_score(sortino, lower=0.0, upper=2.0)
        calmar_quality = _linear_score(calmar, lower=0.0, upper=1.0)
        risk_adjusted_score = 20.0 * (0.5 * sharpe_quality + 0.3 * sortino_quality + 0.2 * calmar_quality)

        month_win = monthly_win_rate(returns)
        year_win = annual_win_rate(returns)
        rolling_series = (
            rolling_sharpe_table[portfolio]
            if rolling_sharpe_table is not None and portfolio in rolling_sharpe_table.columns
            else pd.Series(dtype=float)
        )
        sharpe_stability = rolling_sharpe_stability_score(rolling_series)
        stability_score = 15.0 * (0.4 * month_win + 0.3 * year_win + 0.3 * sharpe_stability)

        turnover_quality = _inverse_linear_score(avg_turnover, lower=0.02, upper=0.20)
        cost_quality = _inverse_linear_score(total_cost_drag, lower=0.002, upper=0.02)
        executability_score = 15.0 * (0.6 * turnover_quality + 0.4 * cost_quality)

        total_score = (
            return_score
            + risk_control_score
            + risk_adjusted_score
            + stability_score
            + executability_score
        )
        rows.append(
            {
                "portfolio": portfolio,
                "return_score": return_score,
                "risk_control_score": risk_control_score,
                "risk_adjusted_score": risk_adjusted_score,
                "stability_score": stability_score,
                "executability_score": executability_score,
                "total_score": total_score,
                "score_pct": total_score / 100.0,
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
