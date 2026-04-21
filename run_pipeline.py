"""Phase 1 pipeline runner for downloading, cleaning, and screening ETF data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.analytics.correlation import (
    build_adjusted_close_matrix,
    correlation_matrix,
    covariance_matrix,
    matrix_to_long_table,
    return_matrix_from_prices,
)
from src.analytics.drawdown import drawdown_from_returns
from src.analytics.risk import rolling_sharpe_ratio, rolling_volatility
from src.backtest.engine import run_fixed_weight_backtest
from src.data.clean_data import batch_clean_price_frames, build_data_quality_summary
from src.data.fetch_prices import fetch_prices
from src.dashboard.plots import write_phase1_chart_outputs
from src.dashboard.reporting import (
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
from src.portfolio.transaction_cost import load_one_way_transaction_cost_bps
from src.portfolio.weights import load_portfolio_template
from src.universe.etf_scoring import score_etf_universe
from src.universe.liquidity_filter import filter_liquid_universe
from src.universe.universe_builder import load_enabled_universe
from src.utils.dates import validate_date_range
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


def run_strategy_backtests(
    asset_returns: pd.DataFrame,
    portfolio_template_config: str | Path,
    benchmark_config: str | Path,
    rebalance_config: str | Path,
    template_name: str | None = None,
) -> tuple[str, dict[str, pd.Series | pd.DataFrame], dict[str, dict[str, pd.Series | pd.DataFrame]]]:
    """Run the default strategy portfolio and configured benchmarks."""
    strategy_name = template_name or "balanced"
    strategy_weights = load_portfolio_template(portfolio_template_config, strategy_name if template_name else None)
    benchmark_weights = load_benchmarks(benchmark_config)
    rebalance_frequency = load_standard_rebalance_frequency(rebalance_config)
    one_way_bps = load_one_way_transaction_cost_bps(rebalance_config)

    benchmark_results: dict[str, dict[str, pd.Series | pd.DataFrame]] = {}
    benchmark_return_series: dict[str, pd.Series] = {}

    for benchmark_name, config in benchmark_weights.items():
        result = run_fixed_weight_backtest(
            asset_returns,
            config["weights"],
            rebalance_frequency=rebalance_frequency,
            one_way_bps=one_way_bps,
        )
        benchmark_results[benchmark_name] = result
        benchmark_return_series[benchmark_name] = result["portfolio_returns"]

    strategy_result = run_fixed_weight_backtest(
        asset_returns,
        strategy_weights,
        rebalance_frequency=rebalance_frequency,
        one_way_bps=one_way_bps,
        benchmark_returns=benchmark_return_series,
    )
    return strategy_name, strategy_result, benchmark_results


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
    nav_table = build_nav_table(strategy_name, strategy_result, benchmark_results)
    return_table = build_return_table(strategy_name, strategy_result, benchmark_results)
    risk_outputs = build_risk_matrix_outputs(asset_returns)

    performance_summary.to_csv(output_path / "performance_summary.csv", index=True)
    turnover_summary.to_csv(output_path / "turnover_summary.csv", index=True)
    annual_return_table.to_csv(output_path / "annual_return_table.csv", index=True)
    benchmark_comparisons.to_csv(output_path / "benchmark_comparisons.csv", index=True)
    benchmark_annual_excess_returns.to_csv(output_path / "benchmark_annual_excess_returns.csv", index=True)
    benchmark_drawdown_comparisons.to_csv(output_path / "benchmark_drawdown_comparisons.csv", index=True)
    nav_table.to_csv(output_path / "nav_series.csv", index=True)
    return_table.to_csv(output_path / "return_series.csv", index=True)
    policy_validation.to_csv(output_path / "backtest_universe_validation.csv", index=True)
    policy_summary.to_csv(output_path / "backtest_universe_policy_summary.csv", index=True)
    risk_outputs["covariance_matrix"].to_csv(output_path / "covariance_matrix.csv", index=True)
    risk_outputs["correlation_matrix"].to_csv(output_path / "correlation_matrix.csv", index=True)
    risk_outputs["covariance_pairs"].to_csv(output_path / "covariance_pairs.csv", index=False)
    risk_outputs["correlation_pairs"].to_csv(output_path / "correlation_pairs.csv", index=False)
    risk_outputs["top_correlation_pairs"].to_csv(output_path / "top_correlation_pairs.csv", index=False)
    risk_outputs["asset_risk_snapshot"].to_csv(output_path / "asset_risk_snapshot.csv", index=True)

    LOGGER.info("Saved performance summary to %s", output_path / "performance_summary.csv")
    LOGGER.info("Saved turnover summary to %s", output_path / "turnover_summary.csv")
    LOGGER.info("Saved annual return table to %s", output_path / "annual_return_table.csv")
    LOGGER.info("Saved benchmark comparisons to %s", output_path / "benchmark_comparisons.csv")
    LOGGER.info("Saved benchmark annual excess returns to %s", output_path / "benchmark_annual_excess_returns.csv")
    LOGGER.info("Saved benchmark drawdown comparisons to %s", output_path / "benchmark_drawdown_comparisons.csv")
    LOGGER.info("Saved covariance matrix to %s", output_path / "covariance_matrix.csv")
    LOGGER.info("Saved correlation matrix to %s", output_path / "correlation_matrix.csv")
    LOGGER.info("Saved top correlation pairs to %s", output_path / "top_correlation_pairs.csv")
    LOGGER.info("Saved asset risk snapshot to %s", output_path / "asset_risk_snapshot.csv")


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
    LOGGER.info("Saved rolling volatility to %s", output_path / "rolling_volatility.csv")
    LOGGER.info("Saved rolling Sharpe to %s", output_path / "rolling_sharpe.csv")
    LOGGER.info("Saved drawdown series to %s", output_path / "drawdown_series.csv")


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


def find_missing_output_inventory_entries(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return output inventory rows where expected artifacts are missing."""
    if inventory.empty or "exists" not in inventory.columns:
        return pd.DataFrame(columns=inventory.columns)
    exists_mask = inventory["exists"].astype(bool)
    return inventory.loc[~exists_mask].reset_index(drop=True)


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
    processed_dir: str | Path,
    figure_dir: str | Path,
    report_dir: str | Path,
    run_completed_at: str | None = None,
) -> dict:
    """Build a reproducibility manifest for a completed Phase 1 pipeline run."""
    completed_at = run_completed_at or pd.Timestamp.utcnow().isoformat()
    strategy_row = performance_summary.loc[strategy_name]
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
        },
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
        "directories": {
            "raw": str(Path(raw_dir)),
            "processed": str(Path(processed_dir)),
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
    parser.add_argument("--template-name", default=None)
    parser.add_argument("--start", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="Exclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--report-dir", default="outputs/reports")
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
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """Execute the Phase 1 data pipeline."""
    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    args.start, args.end = validate_date_range(args.start, args.end)

    tickers = load_enabled_tickers(args.universe_config)
    LOGGER.info("Loaded %s enabled tickers from %s", len(tickers), args.universe_config)

    raw_frames = fetch_prices(
        tickers=tickers,
        start=args.start,
        end=args.end,
        output_dir=args.raw_dir,
        save_raw=True,
    )
    clean_frames = batch_clean_price_frames(raw_frames)
    save_processed_frames(clean_frames, args.processed_dir)
    data_quality_summary = write_data_quality_outputs(clean_frames, args.output_dir)

    liquid_tickers, liquidity_table = filter_liquid_universe(clean_frames)
    etf_summary = score_etf_universe(args.universe_config, liquidity_table)
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
    strategy_name, strategy_result, benchmark_results = run_strategy_backtests(
        asset_returns,
        portfolio_template_config=args.portfolio_config,
        benchmark_config=args.benchmark_config,
        rebalance_config=args.rebalance_config,
        template_name=args.template_name,
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
    write_rolling_metric_outputs(rolling_outputs, args.output_dir)
    chart_paths = write_phase1_chart_outputs(
        strategy_name,
        build_nav_table(strategy_name, strategy_result, benchmark_results),
        strategy_result["annual_return_table"],
        asset_returns,
        args.figure_dir,
        rolling_volatility_table=rolling_outputs["rolling_volatility"],
        rolling_sharpe_table=rolling_outputs["rolling_sharpe"],
    )
    LOGGER.info("Saved Phase 1 charts: %s", chart_paths)

    performance_summary = build_performance_summary(strategy_name, strategy_result, benchmark_results)
    turnover_summary = build_turnover_summary(strategy_name, strategy_result, benchmark_results)
    risk_outputs = build_risk_matrix_outputs(asset_returns)
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
    config_paths = {
        "universe": args.universe_config,
        "portfolio_templates": args.portfolio_config,
        "benchmarks": args.benchmark_config,
        "rebalance_rules": args.rebalance_config,
        "risk_limits": args.risk_limits_config,
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
        rolling_metric_snapshot=rolling_metric_snapshot,
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
        rolling_metric_snapshot=rolling_metric_snapshot,
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
        processed_dir=args.processed_dir,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
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

    final_table_paths = collect_table_output_paths(args.output_dir)
    final_table_paths["output_inventory"] = inventory_path
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
        processed_dir=args.processed_dir,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
    )
    write_pipeline_manifest(manifest, args.output_dir)

    LOGGER.info("Selected %s liquid tickers: %s", len(liquid_tickers), liquid_tickers)


if __name__ == "__main__":
    main()
