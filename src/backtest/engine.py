"""Auditable pandas backtest engine for fixed-weight portfolios."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.analytics.attribution import (
    annual_return_table,
    benchmark_annual_excess_return_table,
    benchmark_comparison,
    benchmark_drawdown_comparison,
)
from src.analytics.risk import risk_summary
from src.portfolio.rebalancer import should_rebalance_by_drift
from src.portfolio.saa import align_weights_to_returns
from src.portfolio.transaction_cost import transaction_cost_drag, turnover_traded_weight

FREQUENCY_TO_PERIOD = {
    "monthly": "M",
    "quarterly": "Q",
}


def apply_trend_scale_to_target_weights(
    target_weights: pd.Series,
    trend_scales: pd.Series,
) -> pd.Series:
    """Apply per-asset trend scales to target weights and renormalize."""
    scaled = target_weights.copy()
    if not trend_scales.empty:
        aligned_scales = trend_scales.reindex(scaled.index, fill_value=1.0)
        scaled = scaled * aligned_scales
    total_weight = float(scaled.sum())
    if total_weight <= 0:
        return target_weights.copy()
    return scaled / total_weight


def build_trend_scale_table(
    adj_close: pd.DataFrame,
    asset_index: pd.Index,
    trend_assets: list[str],
    moving_average_days: int,
    reduction_fraction: float,
) -> pd.DataFrame:
    """Build execution-date trend scales using prior-day close vs moving average."""
    trend_scale_table = pd.DataFrame(1.0, index=asset_index, columns=adj_close.columns, dtype=float)
    if not trend_assets:
        return trend_scale_table

    missing_assets = [asset for asset in trend_assets if asset not in adj_close.columns]
    if missing_assets:
        raise ValueError(f"Trend filter assets are missing from adjusted-close data: {missing_assets}")

    trend_prices = adj_close[trend_assets].sort_index()
    moving_average = trend_prices.rolling(window=moving_average_days, min_periods=moving_average_days).mean()
    scale_for_signal_dates = pd.DataFrame(
        1.0,
        index=trend_prices.index,
        columns=trend_assets,
        dtype=float,
    )
    scale_for_signal_dates = scale_for_signal_dates.where(
        trend_prices >= moving_average,
        other=1.0 - reduction_fraction,
    )
    execution_scales = scale_for_signal_dates.shift(1).reindex(asset_index).fillna(1.0)
    trend_scale_table.loc[:, trend_assets] = execution_scales
    return trend_scale_table


def apply_risk_switch_to_target_weights(
    target_weights: pd.Series,
    risk_assets: list[str],
    destination_assets: list[str],
    reduction_fraction: float,
) -> pd.Series:
    """Reduce risky assets and transfer freed weight to destination assets."""
    adjusted = target_weights.copy()
    if not risk_assets or not destination_assets or reduction_fraction <= 0.0:
        return adjusted

    active_risk_assets = [asset for asset in risk_assets if asset in adjusted.index]
    active_destination_assets = [asset for asset in destination_assets if asset in adjusted.index]
    if not active_risk_assets or not active_destination_assets:
        return adjusted

    original_risk_weights = adjusted.loc[active_risk_assets].copy()
    reduced_risk_weights = original_risk_weights * (1.0 - reduction_fraction)
    freed_weight = float((original_risk_weights - reduced_risk_weights).sum())
    adjusted.loc[active_risk_assets] = reduced_risk_weights

    destination_weights = adjusted.loc[active_destination_assets].copy()
    destination_total = float(destination_weights.sum())
    if destination_total > 0:
        adjusted.loc[active_destination_assets] = destination_weights + (destination_weights / destination_total) * freed_weight
    else:
        adjusted.loc[active_destination_assets] = destination_weights + freed_weight / float(len(active_destination_assets))

    total_weight = float(adjusted.sum())
    if total_weight > 0:
        adjusted = adjusted / total_weight
    return adjusted


def trailing_annualized_volatility(
    returns: pd.Series,
    lookback_days: int,
    periods_per_year: int = 252,
) -> float | None:
    """Compute trailing annualized volatility using only historical returns."""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < lookback_days:
        return None
    window = clean.tail(lookback_days)
    volatility = float(window.std(ddof=1)) * (float(periods_per_year) ** 0.5)
    return volatility


def resolve_rebalance_period_alias(frequency: str) -> str:
    """Resolve a user-facing rebalance frequency to a pandas period alias."""
    normalized = frequency.lower()
    if normalized not in FREQUENCY_TO_PERIOD:
        raise ValueError(
            f"Unsupported rebalance frequency '{frequency}'. Supported values: {sorted(FREQUENCY_TO_PERIOD)}."
        )
    return FREQUENCY_TO_PERIOD[normalized]


def calendar_rebalance_dates(
    index: pd.Index,
    frequency: str = "quarterly",
) -> pd.DatetimeIndex:
    """Return execution dates for calendar rebalances using next-period first trading day."""
    if len(index) == 0:
        return pd.DatetimeIndex([])

    datetime_index = pd.DatetimeIndex(index).sort_values()
    period_alias = resolve_rebalance_period_alias(frequency)
    periods = datetime_index.to_period(period_alias)
    execution_mask = pd.Series(periods, index=datetime_index).ne(
        pd.Series(periods, index=datetime_index).shift(1)
    )
    execution_mask.iloc[0] = True
    return datetime_index[execution_mask.to_numpy()]


def compile_annual_return_table(
    portfolio_returns: pd.Series,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build an annual return table for the portfolio and any benchmarks."""
    annual_tables = [annual_return_table(portfolio_returns)["annual_return"].rename("portfolio")]

    if benchmark_returns:
        for name, returns in benchmark_returns.items():
            annual_tables.append(annual_return_table(returns)["annual_return"].rename(name))

    return pd.concat(annual_tables, axis=1)


def compile_benchmark_comparisons(
    portfolio_returns: pd.Series,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build a benchmark comparison table keyed by benchmark name."""
    if not benchmark_returns:
        return pd.DataFrame()

    comparisons = {
        name: benchmark_comparison(
            portfolio_returns,
            returns,
            periods_per_year=periods_per_year,
        )
        for name, returns in benchmark_returns.items()
    }
    return pd.DataFrame(comparisons).T


def compile_benchmark_annual_excess_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build a calendar-year excess-return table versus each benchmark."""
    return benchmark_annual_excess_return_table(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )


def compile_benchmark_drawdown_comparisons(
    portfolio_returns: pd.Series,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build a max-drawdown comparison table versus each benchmark."""
    return benchmark_drawdown_comparison(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )


def run_fixed_weight_backtest(
    asset_returns: pd.DataFrame,
    target_weights: dict[str, float] | pd.Series,
    rebalance_frequency: str = "quarterly",
    one_way_bps: float = 5.0,
    initial_nav: float = 1.0,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
    periods_per_year: int = 252,
    adj_close: pd.DataFrame | None = None,
    trend_filter: Mapping[str, object] | None = None,
    rebalance_trigger_mode: str = "calendar",
    drift_threshold: float = 0.20,
    drift_rule_enabled: bool = True,
    risk_switch: Mapping[str, object] | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Run a simple fixed-weight portfolio backtest with explicit turnover and costs."""
    if asset_returns.empty:
        raise ValueError("Asset return matrix cannot be empty.")

    asset_returns = asset_returns.sort_index()
    aligned_target = align_weights_to_returns(asset_returns, target_weights)
    rebalance_dates = set(calendar_rebalance_dates(asset_returns.index, frequency=rebalance_frequency))
    trigger_mode = rebalance_trigger_mode.lower()
    supported_trigger_modes = {"calendar", "drift_only", "calendar_or_drift"}
    if trigger_mode not in supported_trigger_modes:
        raise ValueError(
            f"Unsupported rebalance trigger mode '{rebalance_trigger_mode}'. "
            f"Supported values: {sorted(supported_trigger_modes)}."
        )
    trend_scale_table = pd.DataFrame(1.0, index=asset_returns.index, columns=asset_returns.columns, dtype=float)
    trend_active_flags = pd.Series(False, index=asset_returns.index, name="trend_filter_active")
    if trend_filter and bool(trend_filter.get("enabled", False)):
        if adj_close is None or adj_close.empty:
            raise ValueError("Trend filter requires non-empty adjusted-close data.")
        moving_average_days = int(trend_filter.get("moving_average_days", 210))
        reduction_fraction = float(trend_filter.get("reduction_fraction", 0.50))
        trend_assets = [asset for asset in trend_filter.get("assets", []) if asset in asset_returns.columns]
        if moving_average_days < 1:
            raise ValueError("Trend filter moving_average_days must be >= 1.")
        if reduction_fraction < 0.0 or reduction_fraction > 1.0:
            raise ValueError("Trend filter reduction_fraction must be between 0 and 1.")
        trend_scale_table = build_trend_scale_table(
            adj_close=adj_close.reindex(asset_returns.index),
            asset_index=asset_returns.index,
            trend_assets=trend_assets,
            moving_average_days=moving_average_days,
            reduction_fraction=reduction_fraction,
        )
        trend_active_flags = trend_scale_table.lt(1.0).any(axis=1).rename("trend_filter_active")

    risk_switch_scales = pd.DataFrame(1.0, index=asset_returns.index, columns=asset_returns.columns, dtype=float)
    risk_switch_active_flags = pd.Series(False, index=asset_returns.index, name="risk_switch_active")
    risk_switch_enabled = False
    risk_switch_lookback_days = 20
    risk_switch_vol_threshold = None
    risk_switch_reduction_fraction = 0.50
    risk_switch_risk_assets: list[str] = []
    risk_switch_destination_assets: list[str] = []
    previous_risk_switch_active = False

    if risk_switch and bool(risk_switch.get("enabled", False)):
        risk_switch_enabled = True
        risk_switch_lookback_days = int(risk_switch.get("lookback_days", 20))
        risk_switch_vol_threshold = float(risk_switch.get("annualized_volatility_threshold", 0.0))
        risk_switch_reduction_fraction = float(risk_switch.get("reduction_fraction", 0.50))
        risk_switch_risk_assets = [str(asset) for asset in risk_switch.get("risk_assets", []) if str(asset) in asset_returns.columns]
        risk_switch_destination_assets = [
            str(asset) for asset in risk_switch.get("destination_assets", []) if str(asset) in asset_returns.columns
        ]
        if risk_switch_lookback_days < 2:
            raise ValueError("Risk switch lookback_days must be >= 2.")
        if risk_switch_vol_threshold <= 0.0:
            raise ValueError("Risk switch annualized_volatility_threshold must be > 0.")
        if risk_switch_reduction_fraction < 0.0 or risk_switch_reduction_fraction > 1.0:
            raise ValueError("Risk switch reduction_fraction must be between 0 and 1.")

    current_weights = pd.Series(0.0, index=asset_returns.columns, dtype=float)
    nav = float(initial_nav)

    gross_returns: list[float] = []
    net_returns: list[float] = []
    transaction_costs: list[float] = []
    turnover_history: list[float] = []
    nav_history: list[float] = []
    rebalance_flags: list[bool] = []
    rebalance_reasons: list[str] = []
    start_weights_history: list[pd.Series] = []
    end_weights_history: list[pd.Series] = []

    first_date = asset_returns.index.min()
    for date, row in asset_returns.iterrows():
        calendar_flag = date in rebalance_dates
        desired_target = apply_trend_scale_to_target_weights(
            aligned_target,
            trend_scale_table.loc[date],
        )
        risk_switch_active = False
        if risk_switch_enabled:
            historical_returns = pd.Series(net_returns, index=asset_returns.index[: len(net_returns)])
            trailing_volatility = trailing_annualized_volatility(
                historical_returns,
                lookback_days=risk_switch_lookback_days,
                periods_per_year=periods_per_year,
            )
            risk_switch_active = (
                trailing_volatility is not None
                and trailing_volatility > float(risk_switch_vol_threshold)
            )
            risk_switch_active_flags.loc[date] = bool(risk_switch_active)
            if risk_switch_active:
                desired_target = apply_risk_switch_to_target_weights(
                    target_weights=desired_target,
                    risk_assets=risk_switch_risk_assets,
                    destination_assets=risk_switch_destination_assets,
                    reduction_fraction=risk_switch_reduction_fraction,
                )
                for asset in risk_switch_risk_assets:
                    if asset in risk_switch_scales.columns:
                        risk_switch_scales.loc[date, asset] = 1.0 - risk_switch_reduction_fraction

        drift_flag = False
        if drift_rule_enabled:
            drift_flag = should_rebalance_by_drift(
                target_weights=desired_target,
                current_weights=current_weights,
                relative_deviation_threshold=drift_threshold,
            )

        if date == first_date:
            rebalance_flag = True
            rebalance_reason = "initial"
        else:
            risk_switch_flag = bool(risk_switch_enabled and (risk_switch_active != previous_risk_switch_active))
            if trigger_mode == "calendar":
                base_rebalance_flag = calendar_flag
            elif trigger_mode == "drift_only":
                base_rebalance_flag = drift_flag
            else:
                base_rebalance_flag = calendar_flag or drift_flag

            rebalance_flag = base_rebalance_flag or risk_switch_flag
            reason_parts: list[str] = []
            if calendar_flag and trigger_mode in {"calendar", "calendar_or_drift"}:
                reason_parts.append("calendar")
            if drift_flag and trigger_mode in {"drift_only", "calendar_or_drift"}:
                reason_parts.append("drift")
            if risk_switch_flag:
                reason_parts.append("risk_switch")
            rebalance_reason = "_and_".join(reason_parts) if reason_parts else "none"
        weights_before_return = current_weights.copy()

        if rebalance_flag:
            turnover = turnover_traded_weight(desired_target, weights_before_return)
            cost_drag = transaction_cost_drag(
                desired_target,
                current_weights=weights_before_return,
                one_way_bps=one_way_bps,
            )
            weights_before_return = desired_target.copy()
        else:
            turnover = 0.0
            cost_drag = 0.0

        gross_return = float((row * weights_before_return).sum())
        net_return = gross_return - cost_drag
        nav *= 1.0 + net_return

        gross_returns.append(gross_return)
        net_returns.append(net_return)
        transaction_costs.append(cost_drag)
        turnover_history.append(turnover)
        nav_history.append(nav)
        rebalance_flags.append(rebalance_flag)
        rebalance_reasons.append(rebalance_reason)
        start_weights_history.append(weights_before_return.copy())

        end_values = weights_before_return * (1.0 + row)
        total_end_value = float(end_values.sum())
        if total_end_value > 0:
            current_weights = end_values / total_end_value
        else:
            current_weights = pd.Series(0.0, index=asset_returns.columns, dtype=float)

        end_weights_history.append(current_weights.copy())
        previous_risk_switch_active = bool(risk_switch_active)

    portfolio_returns = pd.Series(net_returns, index=asset_returns.index, name="portfolio_return")
    gross_return_series = pd.Series(gross_returns, index=asset_returns.index, name="portfolio_return_gross")
    transaction_cost_series = pd.Series(
        transaction_costs,
        index=asset_returns.index,
        name="transaction_cost_drag",
    )
    turnover_series = pd.Series(turnover_history, index=asset_returns.index, name="turnover_traded_weight")
    nav_series = pd.Series(nav_history, index=asset_returns.index, name="portfolio_nav")
    rebalance_flag_series = pd.Series(rebalance_flags, index=asset_returns.index, name="rebalance_flag")
    rebalance_reason_series = pd.Series(rebalance_reasons, index=asset_returns.index, name="rebalance_reason")
    weights_start = pd.DataFrame(start_weights_history, index=asset_returns.index)
    weights_end = pd.DataFrame(end_weights_history, index=asset_returns.index)

    results: dict[str, pd.Series | pd.DataFrame] = {
        "portfolio_returns": portfolio_returns,
        "portfolio_returns_gross": gross_return_series,
        "transaction_costs": transaction_cost_series,
        "turnover": turnover_series,
        "portfolio_nav": nav_series,
        "rebalance_flags": rebalance_flag_series,
        "weights_start": weights_start,
        "weights_end": weights_end,
        "rebalance_reasons": rebalance_reason_series,
        "trend_filter_scales": trend_scale_table,
        "trend_filter_active": trend_active_flags,
        "risk_switch_scales": risk_switch_scales,
        "risk_switch_active": risk_switch_active_flags,
        "summary": risk_summary(portfolio_returns, periods_per_year=periods_per_year),
        "annual_return_table": compile_annual_return_table(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
        ),
        "benchmark_comparisons": compile_benchmark_comparisons(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
            periods_per_year=periods_per_year,
        ),
        "benchmark_annual_excess_returns": compile_benchmark_annual_excess_returns(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
        ),
        "benchmark_drawdown_comparisons": compile_benchmark_drawdown_comparisons(
            portfolio_returns,
            benchmark_returns=benchmark_returns,
        ),
    }
    return results
