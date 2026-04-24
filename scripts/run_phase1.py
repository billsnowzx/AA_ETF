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
    parser.add_argument("--open-dashboard", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--macro-dir", default="data/macro")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--figure-dir", default="outputs/figures")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    python_executable = sys.executable

    pipeline_cmd = [
        python_executable,
        str(repo_root / "run_pipeline.py"),
        "--start",
        args.start,
        "--backtest-universe-mode",
        args.backtest_universe_mode,
        "--rolling-window",
        str(args.rolling_window),
        "--raw-dir",
        args.raw_dir,
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
        "--fail-on-missing-outputs",
        "--fail-on-empty-outputs",
        "--log-level",
        args.log_level,
    ]
    if args.end:
        pipeline_cmd.extend(["--end", args.end])
    if args.template_name:
        pipeline_cmd.extend(["--template-name", args.template_name])
    if args.as_of_date:
        pipeline_cmd.extend(["--as-of-date", args.as_of_date])
    if args.seed is not None:
        pipeline_cmd.extend(["--seed", str(args.seed)])

    subprocess.run(pipeline_cmd, check=True, cwd=repo_root)

    dashboard_cmd = [
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

    if args.open_dashboard:
        subprocess.run(dashboard_cmd + ["--open"], check=True, cwd=repo_root)
    else:
        subprocess.run(dashboard_cmd + ["--no-server"], check=True, cwd=repo_root)

    if args.open_dashboard:
        subprocess.run(
            [
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
            ],
            check=True,
            cwd=repo_root,
        )


if __name__ == "__main__":
    main()
