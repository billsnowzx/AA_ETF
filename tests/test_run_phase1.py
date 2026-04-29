import sys
from pathlib import Path

from scripts.run_phase1 import (
    build_argument_parser,
    build_dashboard_command,
    build_dashboard_server_command,
    build_pipeline_command,
    build_robustness_command,
    main,
)


def test_argument_parser_exposes_robustness_options() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(["--run-robustness"])

    assert args.run_robustness is True
    assert args.reuse_raw_data is False
    assert args.robustness_rebalance_frequencies == "monthly,quarterly"
    assert args.robustness_one_way_bps_values == "0,5,10"
    assert args.robustness_stress_start_dates == ""
    assert args.robustness_stress_rebalance_frequency == "quarterly"
    assert args.robustness_stress_one_way_bps == 5.0


def test_build_command_helpers_include_expected_flags() -> None:
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "--start",
            "2020-01-01",
            "--end",
            "2020-12-31",
            "--template-name",
            "balanced",
            "--seed",
            "7",
            "--run-robustness",
            "--robustness-rebalance-frequencies",
            "monthly,quarterly",
            "--robustness-one-way-bps-values",
            "0,5",
            "--robustness-stress-start-dates",
            "2020-01-01,2020-07-01",
            "--reuse-raw-data",
        ]
    )
    repo_root = Path("D:/AI/AAETF")
    python_executable = "python"

    pipeline_cmd = build_pipeline_command(python_executable, repo_root, args)
    robustness_cmd = build_robustness_command(python_executable, repo_root, args)
    dashboard_cmd = build_dashboard_command(python_executable, args)
    dashboard_server_cmd = build_dashboard_server_command(python_executable, args)

    assert str(repo_root / "run_pipeline.py") in pipeline_cmd
    assert "--fail-on-missing-outputs" in pipeline_cmd
    assert "--seed" in pipeline_cmd and "7" in pipeline_cmd
    assert "--reuse-raw-data" in pipeline_cmd

    assert str(repo_root / "scripts" / "run_robustness.py") in robustness_cmd
    assert "--stress-start-dates" in robustness_cmd
    assert "--fail-on-empty-outputs" in robustness_cmd
    assert "--reuse-raw-data" in robustness_cmd

    assert dashboard_cmd[:3] == ["python", "-m", "src.dashboard.app"]
    assert "--dashboard-path" in dashboard_cmd
    assert "--host" in dashboard_server_cmd and "--port" in dashboard_server_cmd


def test_main_runs_pipeline_then_dashboard_without_robustness(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check, cwd):
        calls.append(command)
        return None

    monkeypatch.setattr("scripts.run_phase1.subprocess.run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_phase1.py", "--start", "2020-01-01"])

    main()

    assert len(calls) == 2
    assert "run_pipeline.py" in calls[0][1]
    assert "run_robustness.py" not in " ".join(calls[0])
    assert calls[1][:3] == [sys.executable, "-m", "src.dashboard.app"]
    assert "--no-server" in calls[1]


def test_main_runs_robustness_when_enabled(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check, cwd):
        calls.append(command)
        return None

    monkeypatch.setattr("scripts.run_phase1.subprocess.run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase1.py",
            "--start",
            "2020-01-01",
            "--run-robustness",
            "--open-dashboard",
        ],
    )

    main()

    assert len(calls) == 3
    assert "run_pipeline.py" in calls[0][1]
    assert "run_robustness.py" in calls[1][1]
    assert "--open" in calls[2]
    assert "--no-server" not in calls[2]
