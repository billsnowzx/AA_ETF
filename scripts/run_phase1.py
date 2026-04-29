"""One-command Phase 1 runner: pipeline + static dashboard generation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Phase 1 workflow with sane defaults.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--template-name", default=None)
    parser.add_argument("--backtest-universe-mode", default="configured", choices=["configured", "liquidity_filtered"])
    parser.add_argument("--rolling-window", type=int, default=63)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scoring-config", default="config/scoring_rules.yaml")
    parser.add_argument("--open-dashboard", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--reuse-raw-data", action="store_true")
    parser.add_argument("--metadata-dir", default="data/raw/metadata")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--macro-dir", default="data/macro")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-retry-delay", type=float, default=1.0)
    parser.add_argument("--run-robustness", action="store_true")
    parser.add_argument("--robustness-rebalance-frequencies", default="monthly,quarterly")
    parser.add_argument("--robustness-one-way-bps-values", default="0,5,10")
    parser.add_argument("--robustness-stress-start-dates", default="")
    parser.add_argument("--robustness-stress-rebalance-frequency", default="quarterly")
    parser.add_argument("--robustness-stress-one-way-bps", type=float, default=5.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_pipeline_command(
    python_executable: str,
    repo_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build run_pipeline.py command from parsed run_phase1 args."""
    command = [
        python_executable,
        str(repo_root / "run_pipeline.py"),
        "--start",
        args.start,
        "--backtest-universe-mode",
        args.backtest_universe_mode,
        "--rolling-window",
        str(args.rolling_window),
        "--scoring-config",
        args.scoring_config,
        "--raw-dir",
        args.raw_dir,
        "--metadata-dir",
        args.metadata_dir,
        "--processed-dir",
        args.processed_dir,
        "--macro-dir",
        args.macro_dir,
        "--output-dir",
        args.output_dir,
        "--figure-dir",
        args.figure_dir,
        "--report-dir",
        args.report_dir,
        "--download-retries",
        str(args.download_retries),
        "--download-retry-delay",
        str(args.download_retry_delay),
        "--fail-on-missing-outputs",
        "--fail-on-empty-outputs",
        "--log-level",
        args.log_level,
    ]
    if args.end:
        command.extend(["--end", args.end])
    if args.template_name:
        command.extend(["--template-name", args.template_name])
    if args.as_of_date:
        command.extend(["--as-of-date", args.as_of_date])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.reuse_raw_data:
        command.append("--reuse-raw-data")
    return command


def build_robustness_command(
    python_executable: str,
    repo_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build run_robustness.py command from parsed run_phase1 args."""
    command = [
        python_executable,
        str(repo_root / "scripts" / "run_robustness.py"),
        "--start",
        args.start,
        "--raw-dir",
        args.raw_dir,
        "--output-dir",
        args.output_dir,
        "--rebalance-frequencies",
        args.robustness_rebalance_frequencies,
        "--one-way-bps-values",
        args.robustness_one_way_bps_values,
        "--stress-rebalance-frequency",
        args.robustness_stress_rebalance_frequency,
        "--stress-one-way-bps",
        str(args.robustness_stress_one_way_bps),
        "--download-retries",
        str(args.download_retries),
        "--download-retry-delay",
        str(args.download_retry_delay),
        "--fail-on-missing-outputs",
        "--fail-on-empty-outputs",
        "--log-level",
        args.log_level,
    ]
    if args.end:
        command.extend(["--end", args.end])
    if args.template_name:
        command.extend(["--template-name", args.template_name])
    if args.robustness_stress_start_dates:
        command.extend(["--stress-start-dates", args.robustness_stress_start_dates])
    if args.reuse_raw_data:
        command.append("--reuse-raw-data")
    return command


def build_dashboard_command(
    python_executable: str,
    args: argparse.Namespace,
) -> list[str]:
    """Build base dashboard command."""
    return [
        python_executable,
        "-m",
        "src.dashboard.app",
        "--output-dir",
        args.output_dir,
        "--figure-dir",
        args.figure_dir,
        "--report-dir",
        args.report_dir,
        "--dashboard-path",
        str(Path(args.report_dir) / "dashboard.html"),
    ]


def build_dashboard_server_command(
    python_executable: str,
    args: argparse.Namespace,
) -> list[str]:
    """Build dashboard server command for interactive mode."""
    return [
        python_executable,
        "-m",
        "src.dashboard.app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output-dir",
        args.output_dir,
        "--figure-dir",
        args.figure_dir,
        "--report-dir",
        args.report_dir,
        "--dashboard-path",
        str(Path(args.report_dir) / "dashboard.html"),
    ]


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    python_executable = sys.executable

    pipeline_cmd = build_pipeline_command(python_executable, repo_root, args)
    subprocess.run(pipeline_cmd, check=True, cwd=repo_root)

    if args.run_robustness:
        robustness_cmd = build_robustness_command(python_executable, repo_root, args)
        subprocess.run(robustness_cmd, check=True, cwd=repo_root)

    dashboard_cmd = build_dashboard_command(python_executable, args)

    if args.open_dashboard:
        subprocess.run(dashboard_cmd + ["--open"], check=True, cwd=repo_root)
    else:
        subprocess.run(dashboard_cmd + ["--no-server"], check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
