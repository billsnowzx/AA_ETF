"""Phase 1 pipeline runner for downloading, cleaning, and screening ETF data."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.correlation import (
    build_adjusted_close_matrix,
    correlation_matrix,
    covariance_matrix,
    matrix_to_long_table,
    return_matrix_from_prices,
    rolling_correlation,
)
from src.analytics.drawdown import drawdown_from_returns
from src.analytics.evaluation import build_portfolio_evaluation_summary
from src.analytics.risk import risk_contribution_table, rolling_sharpe_ratio, rolling_volatility
from src.backtest.engine import run_fixed_weight_backtest
from src.data.clean_data import batch_clean_price_frames, build_data_quality_summary
from src.data.fetch_etf_metadata import fetch_etf_metadata, save_etf_metadata_snapshots, save_etf_metadata_summary
from src.data.fetch_macro_data import fetch_macro_series, save_macro_series_per_symbol
from src.data.fetch_prices import fetch_prices
from src.dashboard.plots import write_phase1_chart_outputs
from src.dashboard.reporting import (
    build_rebalance_reason_summary,
    build_latest_rolling_metric_snapshot,
    build_phase1_risk_summary_tables,
    build_run_configuration_summary,
    write_phase1_html_report,
    write_phase1_report,
)
from src.portfolio.benchmarks import load_benchmarks
from src.portfolio.policy import (
    build_backtest_universe_validation,
    summarize_backtest_universe_validation,
)
from src.portfolio.rebalancer import load_standard_rebalance_frequency
from src.portfolio.rebalancer import load_trend_filter_settings
from src.portfolio.rebalancer import load_risk_switch_settings
from src.portfolio.portfolio_scoring import build_portfolio_score_summary, load_portfolio_scoring_rules
from src.portfolio.rebalancer import (
    load_rebalance_trigger_mode,
    load_drift_rule_enabled,
    load_relative_drift_threshold,
)
from src.portfolio.risk_limits import (
    build_risk_limit_breach_summary,
    build_portfolio_risk_limit_checks,
    find_risk_limit_breaches,
    load_risk_limits,
)
from src.portfolio.transaction_cost import load_one_way_transaction_cost_bps
from src.portfolio.weights import load_portfolio_template
from src.pipeline.data_sources import hash_files, hash_ticker_csvs, load_price_frames_from_csv
from src.universe.etf_scoring import load_etf_scoring_rules, score_etf_universe
from src.universe.liquidity_filter import filter_liquid_universe
from src.universe.universe_builder import load_enabled_universe
from src.utils.dates import validate_date_range
from src.utils.dates import parse_date
from src.utils.config_schema import validate_phase1_config_files
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def load_enabled_tickers(config_path: str | Path) -> list[str]:
    """Load enabled tickers from the ETF universe config."""
    return load_enabled_universe(config_path).index.tolist()


def save_processed_frames(frames: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Persist cleaned per-ticker frames to the processed data directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ticker, frame in frames.items():
        frame.to_csv(output_path / f"{ticker}.csv", index=True)


def build_asset_return_matrix(
    clean_frames: dict[str, pd.DataFrame],
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Build an adjusted-close return matrix from cleaned per-ticker frames."""
    selected_frames = clean_frames
    if tickers is not None:
        selected_frames = {ticker: clean_frames[ticker] for ticker in tickers if ticker in clean_frames}
        missing_tickers = [ticker for ticker in tickers if ticker not in selected_frames]
        if missing_tickers:
            raise ValueError(f"Missing cleaned frames for tickers: {missing_tickers}")

    price_matrix = build_adjusted_close_matrix(selected_frames)
    return return_matrix_from_prices(price_matrix)


def collect_required_backtest_tickers(
    portfolio_template_config: str | Path,
    benchmark_config: str | Path,
    template_name: str | None = None,
) -> list[str]:
    """Collect the union of tickers required by the strategy template and benchmarks."""
    strategy_weights = load_portfolio_template(
        portfolio_template_config,
        template_name if template_name else None,
    )
    benchmarks = load_benchmarks(benchmark_config)

    ordered_tickers: list[str] = list(strategy_weights.index)
    for benchmark in benchmarks.values():
        for ticker in benchmark["weights"]:
            if ticker not in ordered_tickers:
                ordered_tickers.append(ticker)
    return ordered_tickers


def warn_on_non_liquid_required_assets(
    required_tickers: list[str],
    liquid_tickers: list[str],
) -> list[str]:
    """Log and return any configured strategy assets that fail the liquidity screen."""
    liquid_set = set(liquid_tickers)
    non_liquid_required = [ticker for ticker in required_tickers if ticker not in liquid_set]
    if non_liquid_required:
        LOGGER.warning(
            "Configured strategy or benchmark assets failed the liquidity filter but will remain in the backtest universe: %s",
            non_liquid_required,
        )
    return non_liquid_required


def resolve_backtest_tickers(
    required_tickers: list[str],
    liquid_tickers: list[str],
    mode: str = "configured",
) -> list[str]:
    """Resolve which tickers should be used in the backtest universe."""
    normalized_mode = mode.lower()
    if normalized_mode not in {"configured", "liquidity_filtered"}:
        raise ValueError(
            f"Unsupported backtest universe mode '{mode}'. Supported values: ['configured', 'liquidity_filtered']."
        )

    non_liquid_required = [ticker for ticker in required_tickers if ticker not in set(liquid_tickers)]
    if normalized_mode == "configured":
        return required_tickers

    if non_liquid_required:
        raise ValueError(
            "Liquidity-filtered backtest mode cannot run because configured strategy or benchmark assets "
            f"failed the liquidity screen: {non_liquid_required}"
        )

    return [ticker for ticker in required_tickers if ticker in set(liquid_tickers)]


def write_liquidity_outputs(
    liquid_tickers: list[str],
    liquidity_table: pd.DataFrame,
    etf_summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Persist liquidity summary outputs for audit and downstream use."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    liquidity_summary_path = output_path / "liquidity_summary.csv"
    investable_universe_path = output_path / "investable_universe.csv"
    etf_summary_path = output_path / "etf_summary.csv"

    liquidity_table.to_csv(liquidity_summary_path, index=True)
    etf_summary.to_csv(etf_summary_path, index=True)
    investable_universe_path.write_text(
        "ticker\n" + "\n".join(liquid_tickers) + ("\n" if liquid_tickers else ""),
        encoding="utf-8",
    )

    LOGGER.info("Saved liquidity summary to %s", liquidity_summary_path)
    LOGGER.info("Saved ETF summary to %s", etf_summary_path)
    LOGGER.info("Saved investable universe to %s", investable_universe_path)


def write_data_quality_outputs(
    clean_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Persist per-ticker data-quality diagnostics and return the summary table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_quality_summary = build_data_quality_summary(clean_frames)
    data_quality_path = output_path / "data_quality_summary.csv"
    data_quality_summary.to_csv(data_quality_path, index=True)
    LOGGER.info("Saved data quality summary to %s", data_quality_path)
    return data_quality_summary


def write_macro_outputs(
    macro_series: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist macro observation summary for pipeline reporting and dashboarding."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    macro_path = output_path / "macro_observation_summary.csv"
    macro_series.to_csv(macro_path, index=True)
    LOGGER.info("Saved macro observation summary to %s", macro_path)
    return macro_path


def build_macro_regime_summary(
    macro_series: pd.DataFrame,
    lookback_days: int = 63,
) -> pd.DataFrame:
    """Build a compact macro regime table from the latest observations."""
    columns = [
        "as_of_date",
        "metric",
        "latest_value",
        "reference_value",
        "signal",
        "rule",
    ]
    if macro_series.empty:
        return pd.DataFrame(columns=columns)

    series = macro_series.copy().sort_index().dropna(how="all")
    if series.empty:
        return pd.DataFrame(columns=columns)

    as_of_date = pd.Timestamp(series.index.max()).strftime("%Y-%m-%d")
    lookback_window = max(int(lookback_days), 1)
    rows: list[dict[str, object]] = []
    signal_votes: list[str] = []

    def _latest_non_null(column: str) -> float | None:
        if column not in series.columns:
            return None
        values = pd.to_numeric(series[column], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.iloc[-1])

    def _rolling_median(column: str) -> float | None:
        if column not in series.columns:
            return None
        values = pd.to_numeric(series[column], errors="coerce").dropna().tail(lookback_window)
        if values.empty:
            return None
        return float(values.median())

    vix_latest = _latest_non_null("vix")
    if vix_latest is not None:
        vix_signal = "risk_off" if vix_latest >= 25.0 else "risk_on"
        signal_votes.append(vix_signal)
        rows.append(
            {
                "as_of_date": as_of_date,
                "metric": "vix",
                "latest_value": vix_latest,
                "reference_value": 25.0,
                "signal": vix_signal,
                "rule": "risk_off when latest VIX >= 25",
            }
        )

    slope_latest = _latest_non_null("us_2s10s_slope")
    if slope_latest is not None:
        slope_signal = "risk_off" if slope_latest < 0.0 else "risk_on"
        signal_votes.append(slope_signal)
        rows.append(
            {
                "as_of_date": as_of_date,
                "metric": "us_2s10s_slope",
                "latest_value": slope_latest,
                "reference_value": 0.0,
                "signal": slope_signal,
                "rule": "risk_off when latest 2s10s slope < 0",
            }
        )

    credit_latest = _latest_non_null("hyg_lqd_ratio")
    credit_reference = _rolling_median("hyg_lqd_ratio")
    if credit_latest is not None and credit_reference is not None:
        credit_signal = "risk_off" if credit_latest < credit_reference else "risk_on"
        signal_votes.append(credit_signal)
        rows.append(
            {
                "as_of_date": as_of_date,
                "metric": "hyg_lqd_ratio",
                "latest_value": credit_latest,
                "reference_value": credit_reference,
                "signal": credit_signal,
                "rule": f"risk_off when latest HYG/LQD < {lookback_window}-day median",
            }
        )

    usd_latest = _latest_non_null("usd_index")
    usd_reference = _rolling_median("usd_index")
    if usd_latest is not None and usd_reference is not None:
        usd_signal = "risk_off" if usd_latest > usd_reference else "risk_on"
        signal_votes.append(usd_signal)
        rows.append(
            {
                "as_of_date": as_of_date,
                "metric": "usd_index",
                "latest_value": usd_latest,
                "reference_value": usd_reference,
                "signal": usd_signal,
                "rule": f"risk_off when latest USD index > {lookback_window}-day median",
            }
        )

    risk_off_votes = int(sum(vote == "risk_off" for vote in signal_votes))
    risk_on_votes = int(sum(vote == "risk_on" for vote in signal_votes))
    if risk_off_votes >= 2:
        composite_signal = "risk_off"
    elif risk_on_votes >= 2:
        composite_signal = "risk_on"
    else:
        composite_signal = "mixed"
    rows.append(
        {
            "as_of_date": as_of_date,
            "metric": "composite_regime",
            "latest_value": float(risk_off_votes),
            "reference_value": float(len(signal_votes)),
            "signal": composite_signal,
            "rule": "risk_off if >=2 component risk_off signals; risk_on if >=2 risk_on; else mixed",
        }
    )
    return pd.DataFrame(rows, columns=columns)


def write_macro_regime_summary_output(
    macro_regime_summary: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist macro regime summary for pipeline reporting and dashboarding."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    regime_path = output_path / "macro_regime_summary.csv"
    macro_regime_summary.to_csv(regime_path, index=False)
    LOGGER.info("Saved macro regime summary to %s", regime_path)
    return regime_path


def write_etf_metadata_outputs(
    metadata_summary: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist ETF metadata summary for pipeline reporting and dashboarding."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "etf_metadata_summary.csv"
    save_etf_metadata_summary(metadata_summary, metadata_path)
    LOGGER.info("Saved ETF metadata summary to %s", metadata_path)
    return metadata_path


def run_strategy_backtests(
    asset_returns: pd.DataFrame,
    portfolio_template_config: str | Path,
    benchmark_config: str | Path,
    rebalance_config: str | Path,
    template_name: str | None = None,
    adj_close: pd.DataFrame | None = None,
    trend_filter_settings: dict[str, object] | None = None,
    risk_switch_settings: dict[str, object] | None = None,
) -> tuple[str, dict[str, pd.Series | pd.DataFrame], dict[str, dict[str, pd.Series | pd.DataFrame]]]:
    """Run the default strategy portfolio and configured benchmarks."""
    strategy_name = template_name or "balanced"
    strategy_weights = load_portfolio_template(portfolio_template_config, strategy_name if template_name else None)
    benchmark_weights = load_benchmarks(benchmark_config)
    rebalance_frequency = load_standard_rebalance_frequency(rebalance_config)
    one_way_bps = load_one_way_transaction_cost_bps(rebalance_config)
    rebalance_trigger_mode = load_rebalance_trigger_mode(rebalance_config)
    drift_rule_enabled = load_drift_rule_enabled(rebalance_config)
    drift_threshold = 0.20
    if drift_rule_enabled:
        drift_threshold = load_relative_drift_threshold(rebalance_config)

    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]] = {}
    benchmark_return_series: dict[str, pd.Series] = {}

    for benchmark_name, config in benchmark_weights.items():
        result = run_fixed_weight_backtest(
            asset_returns,
            config["weights"],
            rebalance_frequency=rebalance_frequency,
            one_way_bps=one_way_bps,
            rebalance_trigger_mode=rebalance_trigger_mode,
            drift_threshold=drift_threshold,
            drift_rule_enabled=drift_rule_enabled,
        )
        benchmark_results[benchmark_name] = result
        benchmark_return_series[benchmark_name] = result["portfolio_returns"]

    strategy_result = run_fixed_weight_backtest(
        asset_returns,
        strategy_weights,
        rebalance_frequency=rebalance_frequency,
        one_way_bps=one_way_bps,
        benchmark_returns=benchmark_return_series,
        adj_close=adj_close,
        trend_filter=trend_filter_settings,
        risk_switch=risk_switch_settings,
        rebalance_trigger_mode=rebalance_trigger_mode,
        drift_threshold=drift_threshold,
        drift_rule_enabled=drift_rule_enabled,
    )
    return strategy_name, strategy_result, benchmark_results


def build_trend_filter_overlay_settings(
    rebalance_config: str | Path,
    universe_config: str | Path,
    backtest_tickers: list[str],
) -> dict[str, object]:
    """Build trend-filter settings with equity-like assets resolved from universe metadata."""
    settings = load_trend_filter_settings(rebalance_config)
    if not bool(settings["enabled"]):
        return {
            "enabled": False,
            "moving_average_days": int(settings["moving_average_months"]) * 21,
            "reduction_fraction": float(settings["reduction_fraction"]),
            "assets": [],
        }

    universe_metadata = load_enabled_universe(universe_config)
    selected_assets: list[str] = []
    for ticker in backtest_tickers:
        if ticker not in universe_metadata.index:
            continue
        asset_class = str(universe_metadata.loc[ticker, "asset_class"] or "").lower()
        if "equity" in asset_class or "reit" in asset_class:
            selected_assets.append(ticker)

    return {
        "enabled": True,
        "moving_average_days": int(settings["moving_average_months"]) * 21,
        "reduction_fraction": float(settings["reduction_fraction"]),
        "assets": selected_assets,
    }


def build_backtest_policy_tables(
    strategy_name: str,
    portfolio_template_config: str | Path,
    benchmark_config: str | Path,
    liquid_tickers: list[str],
    template_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build detailed and summary tables for backtest-universe policy validation."""
    strategy_weights = load_portfolio_template(
        portfolio_template_config,
        template_name if template_name else None,
    )
    benchmark_weights = load_benchmarks(benchmark_config)
    validation = build_backtest_universe_validation(
        strategy_name=strategy_name,
        strategy_weights=strategy_weights,
        benchmark_weights=benchmark_weights,
        liquid_tickers=liquid_tickers,
    )
    summary = summarize_backtest_universe_validation(validation)
    return validation, summary


def build_risk_switch_overlay_settings(
    rebalance_config: str | Path,
    universe_config: str | Path,
    backtest_tickers: list[str],
) -> dict[str, object]:
    """Build risk-switch settings with risk assets derived from universe metadata."""
    settings = load_risk_switch_settings(rebalance_config)
    if not bool(settings["enabled"]):
        return {
            "enabled": False,
            "lookback_days": int(settings["lookback_days"]),
            "annualized_volatility_threshold": settings["annualized_volatility_threshold"],
            "reduction_fraction": float(settings["reduction_fraction"]),
            "risk_assets": [],
            "destination_assets": [
                asset for asset in settings["destination_assets"] if asset in backtest_tickers
            ],
        }

    universe_metadata = load_enabled_universe(universe_config)
    destination_assets = [asset for asset in settings["destination_assets"] if asset in backtest_tickers]
    risk_assets: list[str] = []
    destination_set = set(destination_assets)
    for ticker in backtest_tickers:
        if ticker not in universe_metadata.index or ticker in destination_set:
            continue
        asset_class = str(universe_metadata.loc[ticker, "asset_class"] or "").lower()
        if "equity" in asset_class or "reit" in asset_class:
            risk_assets.append(ticker)

    return {
        "enabled": True,
        "lookback_days": int(settings["lookback_days"]),
        "annualized_volatility_threshold": float(settings["annualized_volatility_threshold"]),
        "reduction_fraction": float(settings["reduction_fraction"]),
        "risk_assets": risk_assets,
        "destination_assets": destination_assets,
    }


def build_performance_summary(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a compact performance summary across the strategy and benchmarks."""
    rows: dict[str, pd.Series] = {}

    strategy_summary = strategy_result["summary"].copy()
    strategy_summary["ending_nav"] = float(strategy_result["portfolio_nav"].iloc[-1])
    strategy_summary["total_turnover"] = float(strategy_result["turnover"].sum())
    strategy_summary["total_transaction_cost_drag"] = float(strategy_result["transaction_costs"].sum())
    rows[strategy_name] = strategy_summary

    for benchmark_name, result in benchmark_results.items():
        benchmark_summary = result["summary"].copy()
        benchmark_summary["ending_nav"] = float(result["portfolio_nav"].iloc[-1])
        benchmark_summary["total_turnover"] = float(result["turnover"].sum())
        benchmark_summary["total_transaction_cost_drag"] = float(result["transaction_costs"].sum())
        rows[benchmark_name] = benchmark_summary

    return pd.DataFrame(rows).T


def build_turnover_summary(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a turnover-focused summary across the strategy and benchmarks."""
    rows: list[dict[str, float | str]] = []

    all_results = {strategy_name: strategy_result, **benchmark_results}
    for name, result in all_results.items():
        rows.append(
            {
                "portfolio": name,
                "total_turnover": float(result["turnover"].sum()),
                "average_turnover": float(result["turnover"].mean()),
                "rebalance_count": int(result["rebalance_flags"].sum()),
                "total_transaction_cost_drag": float(result["transaction_costs"].sum()),
            }
        )

    return pd.DataFrame(rows).set_index("portfolio")


def build_rebalance_reason_table(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a per-date rebalance-reason audit table for strategy and benchmarks."""
    def extract_reasons(result: dict[str, pd.Series | pd.DataFrame]) -> pd.Series:
        series = result.get("rebalance_reasons")
        if isinstance(series, pd.Series):
            return series

        for fallback_key in ["portfolio_returns", "portfolio_nav", "turnover"]:
            fallback = result.get(fallback_key)
            if isinstance(fallback, pd.Series):
                return pd.Series("none", index=fallback.index, dtype=object)
        return pd.Series(dtype=object)

    table = pd.DataFrame(
        {
            strategy_name: extract_reasons(strategy_result),
            **{name: extract_reasons(result) for name, result in benchmark_results.items()},
        }
    )
    table.index.name = "date"
    return table


def build_trend_filter_summary(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
) -> pd.DataFrame:
    """Build a compact summary table for trend-filter activity."""
    trend_active = strategy_result.get("trend_filter_active")
    trend_scales = strategy_result.get("trend_filter_scales")

    if not isinstance(trend_active, pd.Series) or trend_active.empty:
        return pd.DataFrame(
            [
                {
                    "portfolio": strategy_name,
                    "observations": 0,
                    "trend_active_days": 0,
                    "trend_active_ratio": 0.0,
                    "avg_reduced_assets": 0.0,
                    "max_reduced_assets": 0,
                    "first_active_date": None,
                    "last_active_date": None,
                }
            ]
        ).set_index("portfolio")

    active_mask = trend_active.astype(bool)
    observations = int(len(active_mask))
    trend_active_days = int(active_mask.sum())
    trend_active_ratio = float(trend_active_days / observations) if observations > 0 else 0.0

    avg_reduced_assets = 0.0
    max_reduced_assets = 0
    if isinstance(trend_scales, pd.DataFrame) and not trend_scales.empty:
        reduced_counts = trend_scales.lt(1.0).sum(axis=1)
        avg_reduced_assets = float(reduced_counts.mean())
        max_reduced_assets = int(reduced_counts.max())

    active_dates = trend_active.index[active_mask]
    first_active_date = active_dates.min() if len(active_dates) > 0 else None
    last_active_date = active_dates.max() if len(active_dates) > 0 else None

    return pd.DataFrame(
        [
            {
                "portfolio": strategy_name,
                "observations": observations,
                "trend_active_days": trend_active_days,
                "trend_active_ratio": trend_active_ratio,
                "avg_reduced_assets": avg_reduced_assets,
                "max_reduced_assets": max_reduced_assets,
                "first_active_date": first_active_date,
                "last_active_date": last_active_date,
            }
        ]
    ).set_index("portfolio")


def build_risk_switch_summary(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
) -> pd.DataFrame:
    """Build a compact summary table for risk-switch activity."""
    risk_switch_active = strategy_result.get("risk_switch_active")
    risk_switch_scales = strategy_result.get("risk_switch_scales")

    if not isinstance(risk_switch_active, pd.Series) or risk_switch_active.empty:
        return pd.DataFrame(
            [
                {
                    "portfolio": strategy_name,
                    "observations": 0,
                    "risk_switch_active_days": 0,
                    "risk_switch_active_ratio": 0.0,
                    "avg_reduced_assets": 0.0,
                    "max_reduced_assets": 0,
                    "first_active_date": None,
                    "last_active_date": None,
                }
            ]
        ).set_index("portfolio")

    active_mask = risk_switch_active.astype(bool)
    observations = int(len(active_mask))
    active_days = int(active_mask.sum())
    active_ratio = float(active_days / observations) if observations > 0 else 0.0

    avg_reduced_assets = 0.0
    max_reduced_assets = 0
    if isinstance(risk_switch_scales, pd.DataFrame) and not risk_switch_scales.empty:
        reduced_counts = risk_switch_scales.lt(1.0).sum(axis=1)
        avg_reduced_assets = float(reduced_counts.mean())
        max_reduced_assets = int(reduced_counts.max())

    active_dates = risk_switch_active.index[active_mask]
    first_active_date = active_dates.min() if len(active_dates) > 0 else None
    last_active_date = active_dates.max() if len(active_dates) > 0 else None

    return pd.DataFrame(
        [
            {
                "portfolio": strategy_name,
                "observations": observations,
                "risk_switch_active_days": active_days,
                "risk_switch_active_ratio": active_ratio,
                "avg_reduced_assets": avg_reduced_assets,
                "max_reduced_assets": max_reduced_assets,
                "first_active_date": first_active_date,
                "last_active_date": last_active_date,
            }
        ]
    ).set_index("portfolio")


def write_portfolio_score_summary(
    performance_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    return_table: pd.DataFrame,
    rolling_sharpe_table: pd.DataFrame,
    output_dir: str | Path,
    scoring_rules: dict | None = None,
) -> Path:
    """Persist portfolio-level score summary as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    portfolio_scores = build_portfolio_score_summary(
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        return_table=return_table,
        rolling_sharpe_table=rolling_sharpe_table,
        scoring_rules=scoring_rules,
    )
    path = output_path / "portfolio_score_summary.csv"
    portfolio_scores.to_csv(path, index=True)
    LOGGER.info("Saved portfolio score summary to %s", path)
    return path


def write_portfolio_evaluation_summary(
    return_table: pd.DataFrame,
    nav_table: pd.DataFrame,
    rolling_sharpe_table: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist portfolio evaluation summary as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    portfolio_evaluation = build_portfolio_evaluation_summary(
        return_table=return_table,
        nav_table=nav_table,
        rolling_sharpe_table=rolling_sharpe_table,
    )
    path = output_path / "portfolio_evaluation_summary.csv"
    portfolio_evaluation.to_csv(path, index=True)
    LOGGER.info("Saved portfolio evaluation summary to %s", path)
    return path


def build_risk_matrix_outputs(asset_returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build covariance and correlation outputs in matrix and long-table forms."""
    covariance = covariance_matrix(asset_returns)
    correlation = correlation_matrix(asset_returns)
    summary_tables = build_phase1_risk_summary_tables(
        correlation_pairs=matrix_to_long_table(correlation, "correlation"),
        correlation_matrix=correlation,
        covariance_matrix=covariance,
    )
    return {
        "covariance_matrix": covariance,
        "correlation_matrix": correlation,
        "covariance_pairs": matrix_to_long_table(covariance, "covariance"),
        "correlation_pairs": matrix_to_long_table(correlation, "correlation"),
        "top_correlation_pairs": summary_tables["top_correlation_pairs"],
        "asset_risk_snapshot": summary_tables["asset_risk_snapshot"],
    }


def build_portfolio_risk_contribution_table(
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Build latest portfolio risk contribution metrics from backtest weights."""
    weights = strategy_result.get("weights_start")
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return pd.DataFrame(
            columns=[
                "weight",
                "marginal_contribution_to_risk",
                "absolute_risk_contribution",
                "percent_risk_contribution",
                "portfolio_volatility",
            ]
        )

    latest_weights = weights.dropna(how="all").tail(1)
    if latest_weights.empty:
        return pd.DataFrame()
    return risk_contribution_table(latest_weights.iloc[0], covariance)


def write_backtest_outputs(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
    policy_validation: pd.DataFrame,
    policy_summary: pd.DataFrame,
    asset_returns: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Persist backtest outputs for the strategy and benchmarks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    performance_summary = build_performance_summary(strategy_name, strategy_result, benchmark_results)
    turnover_summary = build_turnover_summary(strategy_name, strategy_result, benchmark_results)
    annual_return_table = strategy_result["annual_return_table"]
    benchmark_comparisons = strategy_result["benchmark_comparisons"]
    benchmark_annual_excess_returns = strategy_result["benchmark_annual_excess_returns"]
    benchmark_drawdown_comparisons = strategy_result["benchmark_drawdown_comparisons"]
    trend_filter_summary = build_trend_filter_summary(strategy_name, strategy_result)
    risk_switch_summary = build_risk_switch_summary(strategy_name, strategy_result)
    rebalance_reason_table = build_rebalance_reason_table(strategy_name, strategy_result, benchmark_results)
    rebalance_reason_summary = build_rebalance_reason_summary(rebalance_reason_table)
    nav_table = build_nav_table(strategy_name, strategy_result, benchmark_results)
    return_table = build_return_table(strategy_name, strategy_result, benchmark_results)
    turnover_history = build_turnover_history_table(strategy_name, strategy_result, benchmark_results)
    transaction_cost_history = build_transaction_cost_history_table(strategy_name, strategy_result, benchmark_results)
    rebalance_flags = build_rebalance_flag_table(strategy_name, strategy_result, benchmark_results)
    risk_outputs = build_risk_matrix_outputs(asset_returns)
    portfolio_risk_contribution = build_portfolio_risk_contribution_table(
        strategy_result,
        risk_outputs["covariance_matrix"],
    )

    performance_summary.to_csv(output_path / "performance_summary.csv", index=True)
    turnover_summary.to_csv(output_path / "turnover_summary.csv", index=True)
    annual_return_table.to_csv(output_path / "annual_return_table.csv", index=True)
    benchmark_comparisons.to_csv(output_path / "benchmark_comparisons.csv", index=True)
    benchmark_annual_excess_returns.to_csv(output_path / "benchmark_annual_excess_returns.csv", index=True)
    benchmark_drawdown_comparisons.to_csv(output_path / "benchmark_drawdown_comparisons.csv", index=True)
    trend_filter_summary.to_csv(output_path / "trend_filter_summary.csv", index=True)
    risk_switch_summary.to_csv(output_path / "risk_switch_summary.csv", index=True)
    nav_table.to_csv(output_path / "nav_series.csv", index=True)
    return_table.to_csv(output_path / "return_series.csv", index=True)
    turnover_history.to_csv(output_path / "turnover_history.csv", index=True)
    transaction_cost_history.to_csv(output_path / "transaction_cost_history.csv", index=True)
    rebalance_flags.to_csv(output_path / "rebalance_flags.csv", index=True)
    rebalance_reason_table.to_csv(output_path / "rebalance_reason.csv", index=True)
    rebalance_reason_summary.to_csv(output_path / "rebalance_reason_summary.csv", index=True)
    strategy_result["weights_start"].to_csv(output_path / "weights_start_history.csv", index=True)
    strategy_result["weights_end"].to_csv(output_path / "weights_end_history.csv", index=True)
    policy_validation.to_csv(output_path / "backtest_universe_validation.csv", index=True)
    policy_summary.to_csv(output_path / "backtest_universe_policy_summary.csv", index=True)
    risk_outputs["covariance_matrix"].to_csv(output_path / "covariance_matrix.csv", index=True)
    risk_outputs["correlation_matrix"].to_csv(output_path / "correlation_matrix.csv", index=True)
    risk_outputs["covariance_pairs"].to_csv(output_path / "covariance_pairs.csv", index=False)
    risk_outputs["correlation_pairs"].to_csv(output_path / "correlation_pairs.csv", index=False)
    risk_outputs["top_correlation_pairs"].to_csv(output_path / "top_correlation_pairs.csv", index=False)
    risk_outputs["asset_risk_snapshot"].to_csv(output_path / "asset_risk_snapshot.csv", index=True)
    portfolio_risk_contribution.to_csv(output_path / "portfolio_risk_contribution.csv", index=True)
    if "trend_filter_active" in strategy_result:
        strategy_result["trend_filter_active"].to_csv(output_path / "trend_filter_active.csv", index=True)
    if "trend_filter_scales" in strategy_result:
        strategy_result["trend_filter_scales"].to_csv(output_path / "trend_filter_scales.csv", index=True)
    if "risk_switch_active" in strategy_result:
        strategy_result["risk_switch_active"].to_csv(output_path / "risk_switch_active.csv", index=True)
    if "risk_switch_scales" in strategy_result:
        strategy_result["risk_switch_scales"].to_csv(output_path / "risk_switch_scales.csv", index=True)

    LOGGER.info("Saved performance summary to %s", output_path / "performance_summary.csv")
    LOGGER.info("Saved turnover summary to %s", output_path / "turnover_summary.csv")
    LOGGER.info("Saved annual return table to %s", output_path / "annual_return_table.csv")
    LOGGER.info("Saved benchmark comparisons to %s", output_path / "benchmark_comparisons.csv")
    LOGGER.info("Saved benchmark annual excess returns to %s", output_path / "benchmark_annual_excess_returns.csv")
    LOGGER.info("Saved benchmark drawdown comparisons to %s", output_path / "benchmark_drawdown_comparisons.csv")
    LOGGER.info("Saved trend filter summary to %s", output_path / "trend_filter_summary.csv")
    LOGGER.info("Saved risk switch summary to %s", output_path / "risk_switch_summary.csv")
    LOGGER.info("Saved turnover history to %s", output_path / "turnover_history.csv")
    LOGGER.info("Saved transaction cost history to %s", output_path / "transaction_cost_history.csv")
    LOGGER.info("Saved rebalance flags to %s", output_path / "rebalance_flags.csv")
    LOGGER.info("Saved rebalance reasons to %s", output_path / "rebalance_reason.csv")
    LOGGER.info("Saved rebalance reason summary to %s", output_path / "rebalance_reason_summary.csv")
    LOGGER.info("Saved start-of-day weights to %s", output_path / "weights_start_history.csv")
    LOGGER.info("Saved end-of-day weights to %s", output_path / "weights_end_history.csv")
    LOGGER.info("Saved covariance matrix to %s", output_path / "covariance_matrix.csv")
    LOGGER.info("Saved correlation matrix to %s", output_path / "correlation_matrix.csv")
    LOGGER.info("Saved top correlation pairs to %s", output_path / "top_correlation_pairs.csv")
    LOGGER.info("Saved asset risk snapshot to %s", output_path / "asset_risk_snapshot.csv")
    LOGGER.info("Saved portfolio risk contribution to %s", output_path / "portfolio_risk_contribution.csv")
    if "trend_filter_active" in strategy_result:
        LOGGER.info("Saved trend filter active flags to %s", output_path / "trend_filter_active.csv")
    if "trend_filter_scales" in strategy_result:
        LOGGER.info("Saved trend filter scales to %s", output_path / "trend_filter_scales.csv")
    if "risk_switch_active" in strategy_result:
        LOGGER.info("Saved risk switch active flags to %s", output_path / "risk_switch_active.csv")
    if "risk_switch_scales" in strategy_result:
        LOGGER.info("Saved risk switch scales to %s", output_path / "risk_switch_scales.csv")


def build_nav_table(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a NAV table for the strategy and benchmarks."""
    return pd.DataFrame(
        {
            strategy_name: strategy_result["portfolio_nav"],
            **{name: result["portfolio_nav"] for name, result in benchmark_results.items()},
        }
    )


def build_return_table(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a return table for the strategy and benchmarks."""
    return pd.DataFrame(
        {
            strategy_name: strategy_result["portfolio_returns"],
            **{name: result["portfolio_returns"] for name, result in benchmark_results.items()},
        }
    )


def build_turnover_history_table(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a per-date turnover history table for strategy and benchmarks."""
    return pd.DataFrame(
        {
            strategy_name: strategy_result["turnover"],
            **{name: result["turnover"] for name, result in benchmark_results.items()},
        }
    )


def build_transaction_cost_history_table(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a per-date transaction-cost history table for strategy and benchmarks."""
    return pd.DataFrame(
        {
            strategy_name: strategy_result["transaction_costs"],
            **{name: result["transaction_costs"] for name, result in benchmark_results.items()},
        }
    )


def build_rebalance_flag_table(
    strategy_name: str,
    strategy_result: dict[str, pd.Series | pd.DataFrame],
    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Build a per-date rebalance-flag table for strategy and benchmarks."""
    return pd.DataFrame(
        {
            strategy_name: strategy_result["rebalance_flags"],
            **{name: result["rebalance_flags"] for name, result in benchmark_results.items()},
        }
    )


def build_rolling_metric_outputs(
    return_table: pd.DataFrame,
    window: int = 63,
    periods_per_year: int = 252,
) -> dict[str, pd.DataFrame]:
    """Build rolling volatility, Sharpe, and drawdown output tables."""
    return {
        "rolling_volatility": rolling_volatility(
            return_table,
            window=window,
            periods_per_year=periods_per_year,
        ),
        "rolling_sharpe": rolling_sharpe_ratio(
            return_table,
            window=window,
            periods_per_year=periods_per_year,
        ),
        "drawdown_series": drawdown_from_returns(return_table),
    }


def build_rolling_correlation_output(
    asset_returns: pd.DataFrame,
    window: int = 63,
    left: str = "VTI",
    right: str = "AGG",
) -> pd.DataFrame:
    """Build a rolling correlation table for a selected asset pair."""
    column_name = f"{left}_{right}_rolling_correlation"
    if asset_returns.empty:
        return pd.DataFrame(columns=[column_name])
    if left not in asset_returns.columns or right not in asset_returns.columns:
        return pd.DataFrame(columns=[column_name], index=asset_returns.index)

    series = rolling_correlation(
        asset_returns,
        left=left,
        right=right,
        window=window,
        min_periods=window,
    )
    table = pd.DataFrame({column_name: series})
    table.index.name = "date"
    return table


def write_rolling_metric_outputs(
    rolling_outputs: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Persist rolling metric outputs for the strategy and benchmarks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rolling_outputs["rolling_volatility"].to_csv(output_path / "rolling_volatility.csv", index=True)
    rolling_outputs["rolling_sharpe"].to_csv(output_path / "rolling_sharpe.csv", index=True)
    rolling_outputs["drawdown_series"].to_csv(output_path / "drawdown_series.csv", index=True)
    rolling_correlation_table = rolling_outputs.get("rolling_correlation")
    if isinstance(rolling_correlation_table, pd.DataFrame):
        rolling_correlation_table.to_csv(output_path / "rolling_correlation.csv", index=True)
    LOGGER.info("Saved rolling volatility to %s", output_path / "rolling_volatility.csv")
    LOGGER.info("Saved rolling Sharpe to %s", output_path / "rolling_sharpe.csv")
    LOGGER.info("Saved drawdown series to %s", output_path / "drawdown_series.csv")
    if isinstance(rolling_correlation_table, pd.DataFrame):
        LOGGER.info("Saved rolling correlation to %s", output_path / "rolling_correlation.csv")


def write_run_configuration_output(
    run_configuration: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist run parameters and config inputs as an auditable CSV table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "run_configuration.csv"
    run_configuration.to_csv(config_path, index=True)
    LOGGER.info("Saved run configuration to %s", config_path)
    return config_path


def write_risk_limit_output(
    risk_limit_checks: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist risk-limit check table as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    risk_limit_path = output_path / "risk_limit_checks.csv"
    risk_limit_checks.to_csv(risk_limit_path, index=True)
    LOGGER.info("Saved risk limit checks to %s", risk_limit_path)
    return risk_limit_path


def write_risk_limit_breaches_output(
    risk_limit_breaches: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist risk-limit breaches as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    risk_limit_breaches_path = output_path / "risk_limit_breaches.csv"
    risk_limit_breaches.to_csv(risk_limit_breaches_path, index=True)
    LOGGER.info("Saved risk limit breaches to %s", risk_limit_breaches_path)
    return risk_limit_breaches_path


def write_risk_limit_breach_summary_output(
    risk_limit_breach_summary: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist risk-limit breach summary as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    risk_limit_breach_summary_path = output_path / "risk_limit_breach_summary.csv"
    risk_limit_breach_summary.to_csv(risk_limit_breach_summary_path, index=True)
    LOGGER.info("Saved risk limit breach summary to %s", risk_limit_breach_summary_path)
    return risk_limit_breach_summary_path


def validate_risk_limit_artifacts(
    risk_limit_breaches: pd.DataFrame,
    risk_limit_breach_summary: pd.DataFrame,
) -> None:
    """Validate consistency between risk-limit breach detail and summary artifacts."""
    if risk_limit_breach_summary.empty:
        if not risk_limit_breaches.empty:
            raise ValueError("Risk-limit artifact validation failed: breaches exist but summary is empty.")
        return

    if "overall" not in risk_limit_breach_summary.index:
        raise ValueError("Risk-limit artifact validation failed: summary must include 'overall' row.")

    required_columns = {"total_enabled_checks", "breached_checks", "breach_ratio"}
    missing_columns = required_columns - set(risk_limit_breach_summary.columns)
    if missing_columns:
        raise ValueError(
            "Risk-limit artifact validation failed: summary is missing required columns "
            f"{sorted(missing_columns)}."
        )

    for portfolio, summary_row in risk_limit_breach_summary.iterrows():
        total_enabled = int(float(summary_row["total_enabled_checks"]))
        breached = int(float(summary_row["breached_checks"]))
        breach_ratio = float(summary_row["breach_ratio"])
        if total_enabled < 0 or breached < 0:
            raise ValueError(
                "Risk-limit artifact validation failed: total/breached checks must be non-negative for "
                f"{portfolio}."
            )
        if breached > total_enabled:
            raise ValueError(
                "Risk-limit artifact validation failed: breached_checks cannot exceed total_enabled_checks for "
                f"{portfolio}."
            )
        if breach_ratio < 0.0 or breach_ratio > 1.0:
            raise ValueError(
                "Risk-limit artifact validation failed: breach_ratio must be between 0 and 1 for "
                f"{portfolio}."
            )
        expected_ratio = (float(breached / total_enabled) if total_enabled > 0 else 0.0)
        if abs(breach_ratio - expected_ratio) > 1e-12:
            raise ValueError(
                "Risk-limit artifact validation failed: breach_ratio mismatch for "
                f"{portfolio} ({breach_ratio} != {expected_ratio})."
            )

    overall_breached_checks = int(float(risk_limit_breach_summary.loc["overall", "breached_checks"]))
    detail_breached_checks = int(len(risk_limit_breaches))
    if overall_breached_checks != detail_breached_checks:
        raise ValueError(
            "Risk-limit artifact validation failed: overall breached_checks does not match breach detail count "
            f"({overall_breached_checks} != {detail_breached_checks})."
        )

    for portfolio, summary_row in risk_limit_breach_summary.drop(index="overall", errors="ignore").iterrows():
        expected = int(float(summary_row["breached_checks"]))
        actual = (
            int((risk_limit_breaches["portfolio"].astype(str) == str(portfolio)).sum())
            if not risk_limit_breaches.empty and "portfolio" in risk_limit_breaches.columns
            else 0
        )
        if expected != actual:
            raise ValueError(
                "Risk-limit artifact validation failed: per-portfolio breached_checks mismatch for "
                f"{portfolio} ({expected} != {actual})."
            )


def collect_table_output_paths(output_dir: str | Path) -> dict[str, Path]:
    """Collect generated table output files for the pipeline manifest."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(output_path.glob("*.csv"))
    }


def build_output_inventory(
    *,
    table_paths: dict[str, Path],
    report_paths: list[Path],
    chart_paths: dict[str, Path],
    manifest_path: Path,
) -> pd.DataFrame:
    """Build a file-existence and size audit table for generated outputs."""
    rows: list[dict[str, object]] = []

    def add_row(output_type: str, name: str, path: Path) -> None:
        exists = path.exists()
        rows.append(
            {
                "output_type": output_type,
                "name": name,
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
            }
        )

    for name, path in sorted(table_paths.items()):
        add_row("table", name, Path(path))
    for path in report_paths:
        add_row("report", Path(path).stem, Path(path))
    for name, path in sorted(chart_paths.items()):
        add_row("figure", name, Path(path))
    add_row("manifest", manifest_path.stem, manifest_path)

    return pd.DataFrame(rows, columns=["output_type", "name", "path", "exists", "size_bytes"])


def write_output_inventory(inventory: pd.DataFrame, output_dir: str | Path) -> Path:
    """Persist output inventory as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    inventory_path = output_path / "output_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    LOGGER.info("Saved output inventory to %s", inventory_path)
    return inventory_path


def build_pipeline_health_summary(
    *,
    missing_outputs: pd.DataFrame,
    empty_outputs: pd.DataFrame,
    risk_limit_breaches: pd.DataFrame,
    fail_on_missing_outputs: bool,
    fail_on_empty_outputs: bool,
    fail_on_risk_limit_breach: bool,
) -> pd.DataFrame:
    """Build a compact pipeline health table from quality-gate diagnostics."""
    missing_count = int(len(missing_outputs))
    empty_count = int(len(empty_outputs))
    risk_breach_count = int(len(risk_limit_breaches))
    would_fail_missing = bool(fail_on_missing_outputs and missing_count > 0)
    would_fail_empty = bool(fail_on_empty_outputs and empty_count > 0)
    would_fail_risk = bool(fail_on_risk_limit_breach and risk_breach_count > 0)
    run_passed_quality_gates = not (would_fail_missing or would_fail_empty or would_fail_risk)

    return pd.DataFrame(
        [
            {
                "missing_output_count": missing_count,
                "empty_output_count": empty_count,
                "risk_limit_breach_count": risk_breach_count,
                "fail_on_missing_outputs": bool(fail_on_missing_outputs),
                "fail_on_empty_outputs": bool(fail_on_empty_outputs),
                "fail_on_risk_limit_breach": bool(fail_on_risk_limit_breach),
                "would_fail_missing_outputs": would_fail_missing,
                "would_fail_empty_outputs": would_fail_empty,
                "would_fail_risk_limit_breach": would_fail_risk,
                "run_passed_quality_gates": run_passed_quality_gates,
            }
        ],
        index=pd.Index(["health"], name="scope"),
    )


def write_pipeline_health_summary(
    health_summary: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Persist pipeline health summary as an auditable CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    health_path = output_path / "pipeline_health_summary.csv"
    health_summary.to_csv(health_path, index=True)
    LOGGER.info("Saved pipeline health summary to %s", health_path)
    return health_path


def find_missing_output_inventory_entries(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return output inventory rows where expected artifacts are missing."""
    if inventory.empty or "exists" not in inventory.columns:
        return pd.DataFrame(columns=inventory.columns)
    exists_mask = inventory["exists"].astype(bool)
    return inventory.loc[~exists_mask].reset_index(drop=True)


def find_empty_output_inventory_entries(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return output inventory rows where artifacts exist but are empty (size <= 0)."""
    if inventory.empty or "exists" not in inventory.columns or "size_bytes" not in inventory.columns:
        return pd.DataFrame(columns=inventory.columns)
    exists_mask = inventory["exists"].astype(bool)
    size_mask = pd.to_numeric(inventory["size_bytes"], errors="coerce").fillna(0) <= 0
    return inventory.loc[exists_mask & size_mask].reset_index(drop=True)


def build_pipeline_manifest(
    *,
    start: str,
    end: str | None,
    enabled_tickers: list[str],
    liquid_tickers: list[str],
    backtest_tickers: list[str],
    strategy_name: str,
    template_name: str | None,
    backtest_universe_mode: str,
    rolling_window: int,
    performance_summary: pd.DataFrame,
    table_paths: dict[str, Path] | None,
    report_paths: list[Path],
    chart_paths: dict[str, Path],
    config_paths: dict[str, str | Path],
    output_dir: str | Path,
    raw_dir: str | Path,
    metadata_dir: str | Path,
    processed_dir: str | Path,
    macro_dir: str | Path,
    figure_dir: str | Path,
    report_dir: str | Path,
    run_completed_at: str | None = None,
    as_of_date: str | None = None,
    seed: int | None = None,
    risk_limit_breaches: pd.DataFrame | None = None,
    risk_limit_breach_summary: pd.DataFrame | None = None,
    data_source_mode: str = "download",
    input_hashes: dict[str, object] | None = None,
) -> dict:
    """Build a reproducibility manifest for a completed Phase 1 pipeline run."""
    completed_at = run_completed_at or pd.Timestamp.utcnow().isoformat()
    strategy_row = performance_summary.loc[strategy_name]
    breach_table = risk_limit_breaches if risk_limit_breaches is not None else pd.DataFrame()
    breach_summary = risk_limit_breach_summary if risk_limit_breach_summary is not None else pd.DataFrame()
    breach_count = int(len(breach_table))
    has_breach = breach_count > 0
    breached_portfolios: list[str] = []
    if not breach_table.empty and "portfolio" in breach_table.columns:
        breached_portfolios = sorted({str(value) for value in breach_table["portfolio"].dropna().tolist()})

    breach_ratio_by_portfolio: dict[str, float] = {}
    if not breach_summary.empty and "breach_ratio" in breach_summary.columns:
        for portfolio, ratio in breach_summary["breach_ratio"].items():
            breach_ratio_by_portfolio[str(portfolio)] = float(ratio)

    return {
        "run_completed_at": completed_at,
        "date_range": {
            "start": start,
            "end": end,
        },
        "parameters": {
            "backtest_universe_mode": backtest_universe_mode,
            "rolling_window": rolling_window,
            "template_name": template_name,
            "data_source_mode": data_source_mode,
            "as_of_date": as_of_date,
            "seed": seed,
        },
        "input_hashes": input_hashes or {},
        "config_files": {
            name: str(Path(path))
            for name, path in config_paths.items()
        },
        "universes": {
            "enabled_tickers": enabled_tickers,
            "liquid_tickers": liquid_tickers,
            "backtest_tickers": backtest_tickers,
        },
        "strategy": {
            "name": strategy_name,
            "ending_nav": float(strategy_row["ending_nav"]),
            "annualized_return": float(strategy_row["annualized_return"]),
            "annualized_volatility": float(strategy_row["annualized_volatility"]),
            "max_drawdown": float(strategy_row["max_drawdown"]),
        },
        "risk_limits": {
            "has_breach": has_breach,
            "breach_count": breach_count,
            "breached_portfolios": breached_portfolios,
            "breach_ratio_by_portfolio": breach_ratio_by_portfolio,
        },
        "directories": {
            "raw": str(Path(raw_dir)),
            "metadata": str(Path(metadata_dir)),
            "processed": str(Path(processed_dir)),
            "macro": str(Path(macro_dir)),
            "tables": str(Path(output_dir)),
            "figures": str(Path(figure_dir)),
            "reports": str(Path(report_dir)),
        },
        "outputs": {
            "tables": {name: str(path) for name, path in (table_paths or {}).items()},
            "reports": [str(path) for path in report_paths],
            "figures": {name: str(path) for name, path in chart_paths.items()},
        },
    }


def write_pipeline_manifest(manifest: dict, output_dir: str | Path) -> Path:
    """Persist the pipeline manifest as auditable JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Saved pipeline manifest to %s", manifest_path)
    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI for the Phase 1 pipeline runner."""
    parser = argparse.ArgumentParser(description="Run the Phase 1 ETF data pipeline.")
    parser.add_argument("--universe-config", default="config/etf_universe.yaml")
    parser.add_argument("--portfolio-config", default="config/portfolio_templates.yaml")
    parser.add_argument("--benchmark-config", default="config/benchmark_config.yaml")
    parser.add_argument("--rebalance-config", default="config/rebalance_rules.yaml")
    parser.add_argument("--risk-limits-config", default="config/risk_limits.yaml")
    parser.add_argument("--scoring-config", default="config/scoring_rules.yaml")
    parser.add_argument("--template-name", default=None)
    parser.add_argument("--start", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="Exclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--reuse-raw-data",
        action="store_true",
        help="Load per-ticker CSV files from --raw-dir instead of downloading prices.",
    )
    parser.add_argument("--metadata-dir", default="data/raw/metadata")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--macro-dir", default="data/macro")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for yfinance downloads.",
    )
    parser.add_argument(
        "--download-retry-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between yfinance retry attempts.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=63,
        help="Trading-day window used for rolling volatility and rolling Sharpe outputs.",
    )
    parser.add_argument(
        "--backtest-universe-mode",
        default="configured",
        choices=["configured", "liquidity_filtered"],
        help="Use the full configured portfolio universe or require all backtest assets to pass the liquidity screen.",
    )
    parser.add_argument(
        "--fail-on-missing-outputs",
        action="store_true",
        help="Raise an error when output inventory reports missing artifacts.",
    )
    parser.add_argument(
        "--fail-on-empty-outputs",
        action="store_true",
        help="Raise an error when output inventory reports empty artifacts (size <= 0).",
    )
    parser.add_argument(
        "--fail-on-risk-limit-breach",
        action="store_true",
        help="Raise an error when enabled risk-limit checks are breached.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Optional deterministic run date (YYYY-MM-DD) recorded in the manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic runs.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """Execute the Phase 1 data pipeline."""
    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    args.start, args.end = validate_date_range(args.start, args.end)
    if args.as_of_date is not None:
        as_of_ts = parse_date(args.as_of_date)
        args.as_of_date = as_of_ts.strftime("%Y-%m-%d")

    validate_phase1_config_files(
        universe_config_path=args.universe_config,
        portfolio_config_path=args.portfolio_config,
        benchmark_config_path=args.benchmark_config,
        rebalance_config_path=args.rebalance_config,
        risk_limits_config_path=args.risk_limits_config,
        scoring_config_path=args.scoring_config,
    )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    etf_scoring_rules = load_etf_scoring_rules(args.scoring_config)
    portfolio_scoring_rules = load_portfolio_scoring_rules(args.scoring_config)

    tickers = load_enabled_tickers(args.universe_config)
    LOGGER.info("Loaded %s enabled tickers from %s", len(tickers), args.universe_config)
    etf_metadata_summary = fetch_etf_metadata(tickers)
    save_etf_metadata_snapshots(etf_metadata_summary, output_dir=args.metadata_dir)
    etf_metadata_summary_path = write_etf_metadata_outputs(etf_metadata_summary, args.output_dir)

    if args.reuse_raw_data:
        LOGGER.info("Loading raw price CSVs from %s instead of downloading.", args.raw_dir)
        raw_frames = load_price_frames_from_csv(tickers=tickers, input_dir=args.raw_dir)
        price_input_hashes = hash_ticker_csvs(tickers=tickers, input_dir=args.raw_dir)
        data_source_mode = "reuse_raw_data"
    else:
        raw_frames = fetch_prices(
            tickers=tickers,
            start=args.start,
            end=args.end,
            output_dir=args.raw_dir,
            save_raw=True,
            max_retries=args.download_retries,
            retry_delay_seconds=args.download_retry_delay,
        )
        price_input_hashes = hash_ticker_csvs(tickers=tickers, input_dir=args.raw_dir)
        data_source_mode = "download"
    clean_frames = batch_clean_price_frames(raw_frames)
    save_processed_frames(clean_frames, args.processed_dir)
    data_quality_summary = write_data_quality_outputs(clean_frames, args.output_dir)
    macro_series = fetch_macro_series(
        start=args.start,
        end=args.end,
        max_retries=args.download_retries,
        retry_delay_seconds=args.download_retry_delay,
    )
    macro_regime_summary = build_macro_regime_summary(macro_series)
    save_macro_series_per_symbol(macro_series, output_dir=args.macro_dir)
    macro_summary_path = write_macro_outputs(macro_series, args.output_dir)
    macro_regime_summary_path = write_macro_regime_summary_output(macro_regime_summary, args.output_dir)

    liquid_tickers, liquidity_table = filter_liquid_universe(clean_frames)
    etf_summary = score_etf_universe(
        args.universe_config,
        liquidity_table,
        metadata_summary=etf_metadata_summary,
        scoring_rules=etf_scoring_rules,
    )
    write_liquidity_outputs(liquid_tickers, liquidity_table, etf_summary, args.output_dir)

    required_backtest_tickers = collect_required_backtest_tickers(
        args.portfolio_config,
        args.benchmark_config,
        template_name=args.template_name,
    )
    non_liquid_required_assets = warn_on_non_liquid_required_assets(required_backtest_tickers, liquid_tickers)
    backtest_tickers = resolve_backtest_tickers(
        required_backtest_tickers,
        liquid_tickers,
        mode=args.backtest_universe_mode,
    )

    asset_returns = build_asset_return_matrix(clean_frames, tickers=backtest_tickers)
    backtest_adj_close = build_adjusted_close_matrix({ticker: clean_frames[ticker] for ticker in backtest_tickers})
    trend_filter_settings = build_trend_filter_overlay_settings(
        rebalance_config=args.rebalance_config,
        universe_config=args.universe_config,
        backtest_tickers=backtest_tickers,
    )
    risk_switch_settings = build_risk_switch_overlay_settings(
        rebalance_config=args.rebalance_config,
        universe_config=args.universe_config,
        backtest_tickers=backtest_tickers,
    )

    strategy_name, strategy_result, benchmark_results = run_strategy_backtests(
        asset_returns,
        portfolio_template_config=args.portfolio_config,
        benchmark_config=args.benchmark_config,
        rebalance_config=args.rebalance_config,
        template_name=args.template_name,
        adj_close=backtest_adj_close,
        trend_filter_settings=trend_filter_settings,
        risk_switch_settings=risk_switch_settings,
    )
    policy_validation, policy_summary = build_backtest_policy_tables(
        strategy_name=strategy_name,
        portfolio_template_config=args.portfolio_config,
        benchmark_config=args.benchmark_config,
        liquid_tickers=liquid_tickers,
        template_name=args.template_name,
    )
    write_backtest_outputs(
        strategy_name,
        strategy_result,
        benchmark_results,
        policy_validation,
        policy_summary,
        asset_returns,
        args.output_dir,
    )
    return_table = build_return_table(strategy_name, strategy_result, benchmark_results)
    rolling_outputs = build_rolling_metric_outputs(return_table, window=args.rolling_window)
    rolling_outputs["rolling_correlation"] = build_rolling_correlation_output(
        asset_returns=asset_returns,
        window=args.rolling_window,
        left="VTI",
        right="AGG",
    )
    write_rolling_metric_outputs(rolling_outputs, args.output_dir)
    risk_outputs = build_risk_matrix_outputs(asset_returns)
    portfolio_risk_contribution = build_portfolio_risk_contribution_table(
        strategy_result,
        risk_outputs["covariance_matrix"],
    )
    nav_table = build_nav_table(strategy_name, strategy_result, benchmark_results)
    chart_paths = write_phase1_chart_outputs(
        strategy_name,
        nav_table,
        strategy_result["annual_return_table"],
        asset_returns,
        args.figure_dir,
        rolling_volatility_table=rolling_outputs["rolling_volatility"],
        rolling_sharpe_table=rolling_outputs["rolling_sharpe"],
        rolling_correlation_table=rolling_outputs["rolling_correlation"],
        risk_contribution_table=portfolio_risk_contribution,
    )
    LOGGER.info("Saved Phase 1 charts: %s", chart_paths)

    performance_summary = build_performance_summary(strategy_name, strategy_result, benchmark_results)
    turnover_summary = build_turnover_summary(strategy_name, strategy_result, benchmark_results)
    portfolio_score_summary_path = write_portfolio_score_summary(
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        return_table=return_table,
        rolling_sharpe_table=rolling_outputs["rolling_sharpe"],
        scoring_rules=portfolio_scoring_rules,
        output_dir=args.output_dir,
    )
    portfolio_score_summary = pd.read_csv(portfolio_score_summary_path, index_col=0)
    portfolio_evaluation_summary_path = write_portfolio_evaluation_summary(
        return_table=return_table,
        nav_table=nav_table,
        rolling_sharpe_table=rolling_outputs["rolling_sharpe"],
        output_dir=args.output_dir,
    )
    portfolio_evaluation_summary = pd.read_csv(portfolio_evaluation_summary_path, index_col=0)
    trend_filter_summary = build_trend_filter_summary(strategy_name, strategy_result)
    risk_switch_summary = build_risk_switch_summary(strategy_name, strategy_result)
    rebalance_reason_table = build_rebalance_reason_table(strategy_name, strategy_result, benchmark_results)
    risk_limits = load_risk_limits(args.risk_limits_config)
    risk_limit_checks = build_portfolio_risk_limit_checks(performance_summary, risk_limits)
    risk_limit_path = write_risk_limit_output(risk_limit_checks, args.output_dir)
    risk_limit_breaches = find_risk_limit_breaches(risk_limit_checks)
    risk_limit_breach_summary = build_risk_limit_breach_summary(risk_limit_checks)
    validate_risk_limit_artifacts(risk_limit_breaches, risk_limit_breach_summary)
    risk_limit_breaches_path = write_risk_limit_breaches_output(risk_limit_breaches, args.output_dir)
    risk_limit_breach_summary_path = write_risk_limit_breach_summary_output(
        risk_limit_breach_summary, args.output_dir
    )
    rolling_metric_snapshot = build_latest_rolling_metric_snapshot(
        rolling_outputs["rolling_volatility"],
        rolling_outputs["rolling_sharpe"],
    )
    report_notes = []
    if non_liquid_required_assets:
        report_notes.append(
            "Configured strategy or benchmark assets failed the liquidity screen but were retained in the backtest: "
            + ", ".join(non_liquid_required_assets)
        )
    report_notes.append(f"Backtest universe mode: {args.backtest_universe_mode}")
    if bool(trend_filter_settings["enabled"]):
        report_notes.append(
            "Trend filter enabled for assets: "
            + ", ".join(trend_filter_settings["assets"])
            + f" (moving_average_days={trend_filter_settings['moving_average_days']}, "
            + f"reduction_fraction={trend_filter_settings['reduction_fraction']:.2f})"
        )
    if bool(risk_switch_settings["enabled"]):
        report_notes.append(
            "Risk switch enabled: "
            + f"lookback_days={risk_switch_settings['lookback_days']}, "
            + f"annualized_volatility_threshold={risk_switch_settings['annualized_volatility_threshold']:.2%}, "
            + f"reduction_fraction={risk_switch_settings['reduction_fraction']:.2f}, "
            + "risk_assets="
            + (", ".join(risk_switch_settings["risk_assets"]) if risk_switch_settings["risk_assets"] else "none")
            + ", destination_assets="
            + (
                ", ".join(risk_switch_settings["destination_assets"])
                if risk_switch_settings["destination_assets"]
                else "none"
            )
        )
    if not risk_limit_breaches.empty:
        breach_labels = ", ".join(
            f"{row.portfolio}:{row.metric}"
            for row in risk_limit_breaches.itertuples()
        )
        report_notes.append(f"Risk-limit breaches observed: {breach_labels}")
        if args.fail_on_risk_limit_breach:
            raise RuntimeError("Risk-limit validation failed; breached checks: " + breach_labels)
    config_paths = {
        "universe": args.universe_config,
        "portfolio_templates": args.portfolio_config,
        "benchmarks": args.benchmark_config,
        "rebalance_rules": args.rebalance_config,
        "risk_limits": args.risk_limits_config,
        "scoring_rules": args.scoring_config,
    }
    input_hashes = {
        "config_files": hash_files(config_paths.values()),
        "raw_price_files": price_input_hashes,
    }
    run_configuration = build_run_configuration_summary(
        start=args.start,
        end=args.end,
        template_name=args.template_name,
        backtest_universe_mode=args.backtest_universe_mode,
        rolling_window=args.rolling_window,
        config_paths=config_paths,
    )
    write_run_configuration_output(run_configuration, args.output_dir)
    report_path = write_phase1_report(
        strategy_name=strategy_name,
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=strategy_result["annual_return_table"],
        benchmark_comparisons=strategy_result["benchmark_comparisons"],
        benchmark_annual_excess_returns=strategy_result["benchmark_annual_excess_returns"],
        benchmark_drawdown_comparisons=strategy_result["benchmark_drawdown_comparisons"],
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=data_quality_summary,
        covariance_matrix=risk_outputs["covariance_matrix"],
        correlation_matrix=risk_outputs["correlation_matrix"],
        correlation_pairs=risk_outputs["correlation_pairs"],
        chart_paths=chart_paths,
        output_path=Path(args.report_dir) / f"{strategy_name}_phase1_report.md",
        report_date=asset_returns.index.max().strftime("%Y-%m-%d"),
        trend_filter_summary=trend_filter_summary,
        risk_switch_summary=risk_switch_summary,
        rolling_metric_snapshot=rolling_metric_snapshot,
        rolling_correlation=rolling_outputs["rolling_correlation"],
        rebalance_reason_table=rebalance_reason_table,
        risk_limit_checks=risk_limit_checks,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        pipeline_health_summary=None,
        portfolio_risk_contribution=portfolio_risk_contribution,
        portfolio_score_summary=portfolio_score_summary,
        portfolio_evaluation_summary=portfolio_evaluation_summary,
        macro_regime_summary=macro_regime_summary,
        run_configuration=run_configuration,
        notes=report_notes,
    )
    html_report_path = write_phase1_html_report(
        strategy_name=strategy_name,
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=strategy_result["annual_return_table"],
        benchmark_comparisons=strategy_result["benchmark_comparisons"],
        benchmark_annual_excess_returns=strategy_result["benchmark_annual_excess_returns"],
        benchmark_drawdown_comparisons=strategy_result["benchmark_drawdown_comparisons"],
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=data_quality_summary,
        covariance_matrix=risk_outputs["covariance_matrix"],
        correlation_matrix=risk_outputs["correlation_matrix"],
        correlation_pairs=risk_outputs["correlation_pairs"],
        chart_paths=chart_paths,
        output_path=Path(args.report_dir) / f"{strategy_name}_phase1_report.html",
        report_date=asset_returns.index.max().strftime("%Y-%m-%d"),
        trend_filter_summary=trend_filter_summary,
        risk_switch_summary=risk_switch_summary,
        rolling_metric_snapshot=rolling_metric_snapshot,
        rolling_correlation=rolling_outputs["rolling_correlation"],
        rebalance_reason_table=rebalance_reason_table,
        risk_limit_checks=risk_limit_checks,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        pipeline_health_summary=None,
        portfolio_risk_contribution=portfolio_risk_contribution,
        portfolio_score_summary=portfolio_score_summary,
        portfolio_evaluation_summary=portfolio_evaluation_summary,
        macro_regime_summary=macro_regime_summary,
        run_configuration=run_configuration,
        notes=report_notes,
    )
    LOGGER.info("Saved Phase 1 report to %s", report_path)
    LOGGER.info("Saved Phase 1 HTML report to %s", html_report_path)

    manifest_path = Path(args.output_dir) / "pipeline_manifest.json"
    report_paths = [report_path, html_report_path]
    table_paths = collect_table_output_paths(args.output_dir)
    manifest = build_pipeline_manifest(
        start=args.start,
        end=args.end,
        enabled_tickers=tickers,
        liquid_tickers=liquid_tickers,
        backtest_tickers=backtest_tickers,
        strategy_name=strategy_name,
        template_name=args.template_name,
        backtest_universe_mode=args.backtest_universe_mode,
        rolling_window=args.rolling_window,
        performance_summary=performance_summary,
        table_paths=table_paths,
        report_paths=report_paths,
        chart_paths=chart_paths,
        config_paths=config_paths,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        metadata_dir=args.metadata_dir,
        processed_dir=args.processed_dir,
        macro_dir=args.macro_dir,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
        as_of_date=args.as_of_date,
        seed=args.seed,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        data_source_mode=data_source_mode,
        input_hashes=input_hashes,
    )
    write_pipeline_manifest(manifest, args.output_dir)
    output_inventory = build_output_inventory(
        table_paths=collect_table_output_paths(args.output_dir),
        report_paths=report_paths,
        chart_paths=chart_paths,
        manifest_path=manifest_path,
    )
    inventory_path = write_output_inventory(output_inventory, args.output_dir)
    missing_outputs = find_missing_output_inventory_entries(output_inventory)
    empty_outputs = find_empty_output_inventory_entries(output_inventory)
    pipeline_health_summary = build_pipeline_health_summary(
        missing_outputs=missing_outputs,
        empty_outputs=empty_outputs,
        risk_limit_breaches=risk_limit_breaches,
        fail_on_missing_outputs=args.fail_on_missing_outputs,
        fail_on_empty_outputs=args.fail_on_empty_outputs,
        fail_on_risk_limit_breach=args.fail_on_risk_limit_breach,
    )
    pipeline_health_summary_path = write_pipeline_health_summary(
        pipeline_health_summary, args.output_dir
    )
    report_path = write_phase1_report(
        strategy_name=strategy_name,
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=strategy_result["annual_return_table"],
        benchmark_comparisons=strategy_result["benchmark_comparisons"],
        benchmark_annual_excess_returns=strategy_result["benchmark_annual_excess_returns"],
        benchmark_drawdown_comparisons=strategy_result["benchmark_drawdown_comparisons"],
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=data_quality_summary,
        covariance_matrix=risk_outputs["covariance_matrix"],
        correlation_matrix=risk_outputs["correlation_matrix"],
        correlation_pairs=risk_outputs["correlation_pairs"],
        chart_paths=chart_paths,
        output_path=Path(args.report_dir) / f"{strategy_name}_phase1_report.md",
        report_date=asset_returns.index.max().strftime("%Y-%m-%d"),
        trend_filter_summary=trend_filter_summary,
        risk_switch_summary=risk_switch_summary,
        rolling_metric_snapshot=rolling_metric_snapshot,
        rolling_correlation=rolling_outputs["rolling_correlation"],
        rebalance_reason_table=rebalance_reason_table,
        risk_limit_checks=risk_limit_checks,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        pipeline_health_summary=pipeline_health_summary,
        portfolio_risk_contribution=portfolio_risk_contribution,
        portfolio_score_summary=portfolio_score_summary,
        portfolio_evaluation_summary=portfolio_evaluation_summary,
        macro_regime_summary=macro_regime_summary,
        run_configuration=run_configuration,
        notes=report_notes,
    )
    html_report_path = write_phase1_html_report(
        strategy_name=strategy_name,
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=strategy_result["annual_return_table"],
        benchmark_comparisons=strategy_result["benchmark_comparisons"],
        benchmark_annual_excess_returns=strategy_result["benchmark_annual_excess_returns"],
        benchmark_drawdown_comparisons=strategy_result["benchmark_drawdown_comparisons"],
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=data_quality_summary,
        covariance_matrix=risk_outputs["covariance_matrix"],
        correlation_matrix=risk_outputs["correlation_matrix"],
        correlation_pairs=risk_outputs["correlation_pairs"],
        chart_paths=chart_paths,
        output_path=Path(args.report_dir) / f"{strategy_name}_phase1_report.html",
        report_date=asset_returns.index.max().strftime("%Y-%m-%d"),
        trend_filter_summary=trend_filter_summary,
        risk_switch_summary=risk_switch_summary,
        rolling_metric_snapshot=rolling_metric_snapshot,
        rolling_correlation=rolling_outputs["rolling_correlation"],
        rebalance_reason_table=rebalance_reason_table,
        risk_limit_checks=risk_limit_checks,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        pipeline_health_summary=pipeline_health_summary,
        portfolio_risk_contribution=portfolio_risk_contribution,
        portfolio_score_summary=portfolio_score_summary,
        portfolio_evaluation_summary=portfolio_evaluation_summary,
        macro_regime_summary=macro_regime_summary,
        run_configuration=run_configuration,
        notes=report_notes,
    )
    LOGGER.info("Updated Phase 1 report with pipeline health summary at %s", report_path)
    LOGGER.info("Updated Phase 1 HTML report with pipeline health summary at %s", html_report_path)
    if not missing_outputs.empty:
        missing_labels = ", ".join(
            f"{row.output_type}:{row.name}"
            for row in missing_outputs.itertuples(index=False)
        )
        if args.fail_on_missing_outputs:
            raise RuntimeError(
                "Output inventory validation failed; missing artifacts: "
                + missing_labels
            )
        LOGGER.warning("Output inventory found missing artifacts: %s", missing_labels)
    if not empty_outputs.empty:
        empty_labels = ", ".join(
            f"{row.output_type}:{row.name}"
            for row in empty_outputs.itertuples(index=False)
        )
        if args.fail_on_empty_outputs:
            raise RuntimeError(
                "Output inventory validation failed; empty artifacts: "
                + empty_labels
            )
        LOGGER.warning("Output inventory found empty artifacts: %s", empty_labels)

    final_table_paths = collect_table_output_paths(args.output_dir)
    final_table_paths["etf_metadata_summary"] = etf_metadata_summary_path
    final_table_paths["macro_observation_summary"] = macro_summary_path
    final_table_paths["macro_regime_summary"] = macro_regime_summary_path
    final_table_paths["output_inventory"] = inventory_path
    final_table_paths["pipeline_health_summary"] = pipeline_health_summary_path
    final_table_paths["risk_limit_checks"] = risk_limit_path
    final_table_paths["risk_limit_breaches"] = risk_limit_breaches_path
    final_table_paths["risk_limit_breach_summary"] = risk_limit_breach_summary_path
    final_table_paths["portfolio_score_summary"] = portfolio_score_summary_path
    final_table_paths["portfolio_evaluation_summary"] = portfolio_evaluation_summary_path
    manifest = build_pipeline_manifest(
        start=args.start,
        end=args.end,
        enabled_tickers=tickers,
        liquid_tickers=liquid_tickers,
        backtest_tickers=backtest_tickers,
        strategy_name=strategy_name,
        template_name=args.template_name,
        backtest_universe_mode=args.backtest_universe_mode,
        rolling_window=args.rolling_window,
        performance_summary=performance_summary,
        table_paths=final_table_paths,
        report_paths=report_paths,
        chart_paths=chart_paths,
        config_paths=config_paths,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        metadata_dir=args.metadata_dir,
        processed_dir=args.processed_dir,
        macro_dir=args.macro_dir,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
        as_of_date=args.as_of_date,
        seed=args.seed,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        data_source_mode=data_source_mode,
        input_hashes=input_hashes,
    )
    write_pipeline_manifest(manifest, args.output_dir)

    LOGGER.info("Selected %s liquid tickers: %s", len(liquid_tickers), liquid_tickers)


if __name__ == "__main__":
    main()
