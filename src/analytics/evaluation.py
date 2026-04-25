"""Portfolio evaluation metrics for stability and consistency diagnostics."""

from __future__ import annotations

import pandas as pd


def monthly_win_rate(returns: pd.Series) -> float:
    """Compute monthly win rate from periodic return series."""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    monthly = clean.groupby(clean.index.to_period("M")).apply(lambda x: (1.0 + x).prod() - 1.0)
    if monthly.empty:
        return 0.0
    return float((monthly > 0).mean())


def annual_win_rate(returns: pd.Series) -> float:
    """Compute annual win rate from periodic return series."""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    annual = clean.groupby(clean.index.to_period("Y")).apply(lambda x: (1.0 + x).prod() - 1.0)
    if annual.empty:
        return 0.0
    return float((annual > 0).mean())


def max_drawdown_recovery_days(nav: pd.Series) -> int:
    """Compute the longest recovery stretch (in observations) from drawdown back to prior peak."""
    clean = pd.to_numeric(nav, errors="coerce").dropna()
    if clean.empty:
        return 0

    running_max = clean.cummax()
    in_drawdown = clean < running_max

    max_stretch = 0
    current_stretch = 0
    for flag in in_drawdown:
        if bool(flag):
            current_stretch += 1
            if current_stretch > max_stretch:
                max_stretch = current_stretch
        else:
            current_stretch = 0
    return int(max_stretch)


def rolling_sharpe_stability(rolling_sharpe: pd.Series) -> float:
    """Compute a simple stability score from rolling Sharpe variability."""
    clean = pd.to_numeric(rolling_sharpe, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    dispersion = float(clean.std(ddof=1))
    if dispersion >= 1.0:
        return 0.0
    return float(max(0.0, 1.0 - dispersion))


def build_portfolio_evaluation_summary(
    return_table: pd.DataFrame,
    nav_table: pd.DataFrame,
    rolling_sharpe_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-portfolio evaluation summary from return/NAV diagnostics."""
    portfolios = [name for name in return_table.columns if name in nav_table.columns]
    rows: list[dict[str, object]] = []
    for portfolio in portfolios:
        rolling_sharpe_series = (
            rolling_sharpe_table[portfolio]
            if rolling_sharpe_table is not None and portfolio in rolling_sharpe_table.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "portfolio": portfolio,
                "monthly_win_rate": monthly_win_rate(return_table[portfolio]),
                "annual_win_rate": annual_win_rate(return_table[portfolio]),
                "max_drawdown_recovery_days": max_drawdown_recovery_days(nav_table[portfolio]),
                "rolling_sharpe_stability": rolling_sharpe_stability(rolling_sharpe_series),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "monthly_win_rate",
                "annual_win_rate",
                "max_drawdown_recovery_days",
                "rolling_sharpe_stability",
            ]
        )
    return pd.DataFrame(rows).set_index("portfolio")

