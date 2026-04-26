"""Run Layer-2 robustness sweeps from config-driven ETF inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analytics.correlation import build_adjusted_close_matrix, return_matrix_from_prices
from src.backtest.scenarios import run_robustness_scenarios, write_robustness_scenarios
from src.backtest.stress_test import run_start_date_robustness, write_start_date_robustness
from src.data.clean_data import batch_clean_price_frames
from src.data.fetch_prices import fetch_prices
from src.portfolio.weights import load_portfolio_template
from src.universe.universe_builder import load_enabled_universe
from src.utils.logger import configure_logging


def parse_csv_strings(value: str) -> list[str]:
    """Parse comma-separated values into a trimmed non-empty string list."""
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("CSV input cannot be empty.")
    return parsed


def parse_csv_floats(value: str) -> list[float]:
    """Parse comma-separated values into a float list."""
    values = parse_csv_strings(value)
    return [float(item) for item in values]


def build_default_stress_start_dates(asset_returns: pd.DataFrame, max_starts: int = 3) -> list[pd.Timestamp]:
    """Build default start dates from earliest available trading day in each year."""
    if asset_returns.empty:
        raise ValueError("Asset return matrix cannot be empty.")
    if max_starts < 1:
        raise ValueError("max_starts must be >= 1.")

    index = pd.DatetimeIndex(asset_returns.index).sort_values()
    by_year = pd.Series(index, index=index).groupby(index.year).min()
    starts = [pd.Timestamp(value) for value in by_year.tolist()]
    return starts[:max_starts]


def run_robustness_workflow(
    *,
    universe_config: str | Path,
    portfolio_config: str | Path,
    template_name: str | None,
    start: str,
    end: str | None,
    raw_dir: str | Path,
    output_dir: str | Path,
    rebalance_frequencies: list[str],
    one_way_bps_values: list[float],
    stress_start_dates: list[str] | None,
    stress_rebalance_frequency: str,
    stress_one_way_bps: float,
    download_retries: int,
    download_retry_delay: float,
) -> dict[str, Path]:
    """Run robustness sweeps and persist scenario tables."""
    tickers = load_enabled_universe(universe_config).index.tolist()
    raw_frames = fetch_prices(
        tickers=tickers,
        start=start,
        end=end,
        output_dir=raw_dir,
        save_raw=True,
        max_retries=download_retries,
        retry_delay_seconds=download_retry_delay,
    )
    clean_frames = batch_clean_price_frames(raw_frames)
    asset_returns = return_matrix_from_prices(build_adjusted_close_matrix(clean_frames))
    target_weights = load_portfolio_template(portfolio_config, template_name)

    scenario_table = run_robustness_scenarios(
        asset_returns=asset_returns,
        target_weights=target_weights,
        rebalance_frequencies=rebalance_frequencies,
        one_way_bps_values=one_way_bps_values,
    )
    scenario_path = write_robustness_scenarios(scenario_table, output_dir=output_dir)

    if stress_start_dates:
        start_dates = [pd.Timestamp(value) for value in stress_start_dates]
    else:
        start_dates = build_default_stress_start_dates(asset_returns, max_starts=3)

    stress_table = run_start_date_robustness(
        asset_returns=asset_returns,
        target_weights=target_weights,
        start_dates=start_dates,
        rebalance_frequency=stress_rebalance_frequency,
        one_way_bps=stress_one_way_bps,
    )
    stress_path = write_start_date_robustness(stress_table, output_dir=output_dir)
    return {
        "robustness_scenarios": scenario_path,
        "start_date_robustness": stress_path,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser for robustness runs."""
    parser = argparse.ArgumentParser(description="Run robustness sweeps for backtest assumptions.")
    parser.add_argument("--universe-config", default="config/etf_universe.yaml")
    parser.add_argument("--portfolio-config", default="config/portfolio_templates.yaml")
    parser.add_argument("--template-name", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--rebalance-frequencies", default="monthly,quarterly")
    parser.add_argument("--one-way-bps-values", default="0,5,10")
    parser.add_argument("--stress-start-dates", default="")
    parser.add_argument("--stress-rebalance-frequency", default="quarterly")
    parser.add_argument("--stress-one-way-bps", type=float, default=5.0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-retry-delay", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """Run robustness scenario and stress sweeps from the CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    rebalance_frequencies = parse_csv_strings(args.rebalance_frequencies)
    one_way_bps_values = parse_csv_floats(args.one_way_bps_values)
    stress_start_dates = parse_csv_strings(args.stress_start_dates) if args.stress_start_dates.strip() else None

    run_robustness_workflow(
        universe_config=args.universe_config,
        portfolio_config=args.portfolio_config,
        template_name=args.template_name,
        start=args.start,
        end=args.end,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        rebalance_frequencies=rebalance_frequencies,
        one_way_bps_values=one_way_bps_values,
        stress_start_dates=stress_start_dates,
        stress_rebalance_frequency=args.stress_rebalance_frequency,
        stress_one_way_bps=args.stress_one_way_bps,
        download_retries=args.download_retries,
        download_retry_delay=args.download_retry_delay,
    )


if __name__ == "__main__":
    main()
