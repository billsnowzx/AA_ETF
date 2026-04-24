"""Validate that required Phase 1 output artifacts exist and are non-empty."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_TABLES = [
    "performance_summary.csv",
    "annual_return_table.csv",
    "benchmark_comparisons.csv",
    "benchmark_annual_excess_returns.csv",
    "benchmark_drawdown_comparisons.csv",
    "trend_filter_summary.csv",
    "rebalance_reason.csv",
    "rebalance_reason_summary.csv",
    "macro_observation_summary.csv",
    "risk_limit_checks.csv",
    "risk_limit_breaches.csv",
    "risk_limit_breach_summary.csv",
    "portfolio_risk_contribution.csv",
    "rolling_volatility.csv",
    "rolling_sharpe.csv",
    "output_inventory.csv",
    "pipeline_health_summary.csv",
    "pipeline_manifest.json",
]

REQUIRED_FIGURES = [
    "balanced_nav.png",
    "balanced_drawdown.png",
    "balanced_annual_returns.png",
    "balanced_rolling_volatility.png",
    "balanced_rolling_sharpe.png",
    "correlation_heatmap.png",
    "balanced_risk_contribution.png",
    "balanced_mctr.png",
]

REQUIRED_REPORTS = [
    "balanced_phase1_report.md",
    "balanced_phase1_report.html",
    "dashboard.html",
]


def _validate_non_empty_files(base_dir: Path, relative_files: list[str], label: str) -> list[str]:
    errors: list[str] = []
    for filename in relative_files:
        path = base_dir / filename
        if not path.exists():
            errors.append(f"{label} missing: {path}")
            continue
        size = path.stat().st_size
        if size <= 0:
            errors.append(f"{label} empty: {path}")
    return errors


def validate_required_outputs(
    *,
    table_dir: str | Path,
    figure_dir: str | Path,
    report_dir: str | Path,
) -> None:
    """Raise ValueError when required artifacts are missing or empty."""
    errors: list[str] = []
    errors.extend(_validate_non_empty_files(Path(table_dir), REQUIRED_TABLES, "table"))
    errors.extend(_validate_non_empty_files(Path(figure_dir), REQUIRED_FIGURES, "figure"))
    errors.extend(_validate_non_empty_files(Path(report_dir), REQUIRED_REPORTS, "report"))
    if errors:
        raise ValueError("Required output validation failed:\n- " + "\n- ".join(errors))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate required non-empty Phase 1 output artifacts.")
    parser.add_argument("--table-dir", default="outputs/tables")
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--report-dir", default="outputs/reports")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    validate_required_outputs(
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
