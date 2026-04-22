import shutil
import sys
import uuid
from pathlib import Path

import json
import pandas as pd

from src.dashboard.app import (
    _build_manifest_summary,
    _format_dashboard_tables,
    build_dashboard_html,
    main,
    open_dashboard_html,
    write_dashboard_html,
)


def test_build_dashboard_html_contains_tables_and_figures() -> None:
    output_dir = Path("data/cache") / f"dashboard_source_{uuid.uuid4().hex}"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        pd.DataFrame({"annualized_return": [0.1]}, index=pd.Index(["balanced"], name="portfolio")).to_csv(
            table_dir / "performance_summary.csv"
        )
        pd.DataFrame({"tracking_error": [0.02]}, index=pd.Index(["benchmark_a"], name="benchmark")).to_csv(
            table_dir / "benchmark_comparisons.csv"
        )
        pd.DataFrame({"benchmark_a": [0.01]}, index=pd.Index([2024], name="year")).to_csv(
            table_dir / "benchmark_annual_excess_returns.csv"
        )
        pd.DataFrame(
            {"strategy_max_drawdown": [-0.2], "benchmark_max_drawdown": [-0.15], "max_drawdown_gap": [-0.05]},
            index=pd.Index(["benchmark_a"], name="benchmark"),
        ).to_csv(table_dir / "benchmark_drawdown_comparisons.csv")
        pd.DataFrame({"pair": ["VTI vs VEA"], "correlation": ["0.8604"]}).to_csv(
            table_dir / "top_correlation_pairs.csv", index=False
        )
        pd.DataFrame({"avg_correlation": ["0.4389"], "variance": ["0.000129"]}, index=pd.Index(["VTI"])).to_csv(
            table_dir / "asset_risk_snapshot.csv"
        )
        pd.DataFrame({"balanced": [0.10]}, index=pd.Index(["2024-01-01"], name="date")).to_csv(
            table_dir / "rolling_volatility.csv"
        )
        pd.DataFrame({"balanced": [0.5]}, index=pd.Index(["2024-01-01"], name="date")).to_csv(
            table_dir / "rolling_sharpe.csv"
        )
        pd.DataFrame({"asset_class": ["us_equity"]}, index=pd.Index(["VTI"])).to_csv(table_dir / "etf_summary.csv")
        pd.DataFrame(
            {
                "start_date": ["2024-01-02"],
                "end_date": ["2024-01-03"],
                "observations": [2],
                "missing_volume": [0],
            },
            index=pd.Index(["VTI"], name="ticker"),
        ).to_csv(table_dir / "data_quality_summary.csv")
        pd.DataFrame(
            {
                "observations": [3],
                "trend_active_days": [2],
                "trend_active_ratio": [2.0 / 3.0],
                "avg_reduced_assets": [2.0 / 3.0],
                "max_reduced_assets": [1],
            },
            index=pd.Index(["balanced"], name="portfolio"),
        ).to_csv(table_dir / "trend_filter_summary.csv")
        pd.DataFrame(
            {
                "balanced": ["none", "calendar", "calendar+drift"],
                "benchmark_a": ["calendar", "none", "drift"],
            },
            index=pd.Index(["2024-01-02", "2024-01-03", "2024-01-04"], name="date"),
        ).to_csv(table_dir / "rebalance_reason.csv")
        pd.DataFrame(
            {
                "total_days": [3],
                "rebalance_days": [2],
                "rebalance_ratio": [2.0 / 3.0],
                "top_reason": ["calendar"],
            },
            index=pd.Index(["balanced"], name="portfolio"),
        ).to_csv(table_dir / "rebalance_reason_summary.csv")
        pd.DataFrame(
            {"value": ["2024-01-01", "liquidity_filtered", "config\\risk_limits.yaml"]},
            index=pd.Index(["start", "backtest_universe_mode", "config_risk_limits"]),
        ).to_csv(table_dir / "run_configuration.csv")
        pd.DataFrame(
            {
                "output_type": ["table"],
                "name": ["performance_summary"],
                "path": ["outputs\\tables\\performance_summary.csv"],
                "exists": [True],
                "size_bytes": [1234],
            }
        ).to_csv(table_dir / "output_inventory.csv", index=False)
        (table_dir / "pipeline_manifest.json").write_text(
            json.dumps(
                {
                    "run_completed_at": "2026-04-20T00:00:00+00:00",
                    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
                    "parameters": {"backtest_universe_mode": "liquidity_filtered", "rolling_window": 63},
                    "strategy": {"name": "balanced", "ending_nav": 1.25},
                    "config_files": {"universe": "config/etf_universe.yaml"},
                    "universes": {
                        "enabled_tickers": ["VTI", "AGG"],
                        "liquid_tickers": ["AGG", "VTI"],
                        "backtest_tickers": ["VTI", "AGG"],
                    },
                }
            ),
            encoding="utf-8",
        )
        (figure_dir / "balanced_nav.png").write_bytes(b"fake")
        (report_dir / "balanced_phase1_report.html").write_text("<html></html>", encoding="utf-8")

        html = build_dashboard_html(output_dir=table_dir, figure_dir=figure_dir, report_dir=report_dir)

        assert "AA ETF Research Dashboard" in html
        assert "Performance Summary" in html
        assert "Benchmark Annual Excess Returns" in html
        assert "balanced_nav.png" in html
        assert "balanced_phase1_report.html" in html
        assert "10.00%" in html
        assert "Latest Rolling Volatility" in html
        assert "Run Manifest" in html
        assert "Run Configuration" in html
        assert "config\\risk_limits.yaml" in html
        assert "Output Inventory" in html
        assert "performance_summary" in html
        assert "liquidity_filtered" in html
        assert "config/etf_universe.yaml" in html
        assert "Data Quality Summary" in html
        assert "Trend Filter Summary" in html
        assert "66.67%" in html
        assert "Recent Rebalance Reasons" in html
        assert "Rebalance Reason Summary" in html
        assert "calendar+drift" in html
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_format_dashboard_tables_humanizes_numeric_fields() -> None:
    tables = _format_dashboard_tables(
        {
            "performance_summary": pd.DataFrame(
                {"annualized_return": [0.1], "ending_nav": [1.5]},
                index=pd.Index(["balanced"]),
            ),
            "benchmark_comparisons": pd.DataFrame(
                {"tracking_error": [0.02], "information_ratio": [0.5]},
                index=pd.Index(["benchmark_a"]),
            ),
            "benchmark_annual_excess_returns": pd.DataFrame({"benchmark_a": [0.01]}, index=pd.Index([2024])),
            "benchmark_drawdown_comparisons": pd.DataFrame(
                {"max_drawdown_gap": [-0.05]},
                index=pd.Index(["benchmark_a"]),
            ),
            "top_correlation_pairs": pd.DataFrame({"pair": ["VTI vs VEA"], "correlation": [0.8604]}),
            "asset_risk_snapshot": pd.DataFrame({"avg_correlation": [0.4389], "variance": [0.000129]}),
            "rolling_volatility": pd.DataFrame({"balanced": [0.10]}),
            "rolling_sharpe": pd.DataFrame({"balanced": [0.5]}),
            "manifest_summary": pd.DataFrame({"value": [1.25]}, index=pd.Index(["ending_nav"])),
            "run_configuration": pd.DataFrame({"value": ["config\\risk_limits.yaml"]}, index=pd.Index(["config_risk_limits"])),
            "output_inventory": pd.DataFrame({"name": ["performance_summary"], "size_bytes": [1234]}),
            "data_quality_summary": pd.DataFrame({"observations": [2], "missing_volume": [0]}),
            "trend_filter_summary": pd.DataFrame(
                {
                    "observations": [3],
                    "trend_active_days": [2],
                    "trend_active_ratio": [2.0 / 3.0],
                    "avg_reduced_assets": [2.0 / 3.0],
                    "max_reduced_assets": [1],
                }
            ),
            "rebalance_reason": pd.DataFrame({"balanced": ["calendar+drift"]}, index=pd.Index(["2024-01-04"])),
            "rebalance_reason_summary": pd.DataFrame(
                {"rebalance_ratio": [2.0 / 3.0], "top_reason": ["calendar"]},
                index=pd.Index(["balanced"]),
            ),
            "etf_summary": pd.DataFrame(
                {
                    "average_dollar_volume": [1_000_000.0],
                    "recent_pass_ratio": [1.0],
                    "phase1_score_pct": [1.0],
                    "observations": [2839],
                },
                index=pd.Index(["VTI"]),
            ),
        }
    )

    assert tables["performance_summary"].loc["balanced", "annualized_return"] == "10.00%"
    assert tables["benchmark_comparisons"].loc["benchmark_a", "tracking_error"] == "2.00%"
    assert tables["benchmark_annual_excess_returns"].loc[2024, "benchmark_a"] == "1.00%"
    assert tables["benchmark_drawdown_comparisons"].loc["benchmark_a", "max_drawdown_gap"] == "-5.00%"
    assert tables["top_correlation_pairs"].iloc[0]["correlation"] == "0.8604"
    assert tables["asset_risk_snapshot"].iloc[0]["variance"] == "0.000129"
    assert tables["rolling_volatility"].iloc[0]["balanced"] == "10.00%"
    assert tables["rolling_sharpe"].iloc[0]["balanced"] == "0.5000"
    assert tables["manifest_summary"].loc["ending_nav", "value"] == "1.2500"
    assert tables["run_configuration"].loc["config_risk_limits", "value"] == "config\\risk_limits.yaml"
    assert tables["output_inventory"].iloc[0]["size_bytes"] == "1234"
    assert tables["data_quality_summary"].iloc[0]["observations"] == "2"
    assert tables["trend_filter_summary"].iloc[0]["trend_active_ratio"] == "66.67%"
    assert tables["trend_filter_summary"].iloc[0]["max_reduced_assets"] == "1"
    assert tables["rebalance_reason"].iloc[0]["balanced"] == "calendar+drift"
    assert tables["rebalance_reason_summary"].loc["balanced", "rebalance_ratio"] == "66.67%"
    assert tables["etf_summary"].loc["VTI", "average_dollar_volume"] == "1000000"


def test_build_manifest_summary_flattens_key_run_context() -> None:
    manifest = {
        "run_completed_at": "2026-04-20T00:00:00+00:00",
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "parameters": {"backtest_universe_mode": "liquidity_filtered", "rolling_window": 63},
        "strategy": {"name": "balanced", "ending_nav": 1.25},
        "config_files": {"universe": "config/etf_universe.yaml"},
        "universes": {
            "enabled_tickers": ["VTI", "AGG"],
            "liquid_tickers": ["AGG", "VTI"],
            "backtest_tickers": ["VTI", "AGG"],
        },
    }

    summary = _build_manifest_summary(manifest)

    assert summary.loc["date_range", "value"] == "2024-01-01 to 2024-12-31"
    assert summary.loc["backtest_universe_mode", "value"] == "liquidity_filtered"
    assert summary.loc["backtest_tickers", "value"] == "VTI, AGG"
    assert summary.loc["config_universe", "value"] == "config/etf_universe.yaml"


def test_write_dashboard_html_creates_file() -> None:
    output_dir = Path("data/cache") / f"dashboard_write_{uuid.uuid4().hex}"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / "dashboard.html"

    try:
        pd.DataFrame({"annualized_return": [0.1]}, index=pd.Index(["balanced"], name="portfolio")).to_csv(
            table_dir / "performance_summary.csv"
        )
        result = write_dashboard_html(
            output_path=destination,
            output_dir=table_dir,
            figure_dir=figure_dir,
            report_dir=report_dir,
        )
        assert result.exists()
        assert "AA ETF Research Dashboard" in result.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_open_dashboard_html_opens_file_url(monkeypatch) -> None:
    opened_urls: list[str] = []

    def fake_open(url: str) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr("src.dashboard.app.webbrowser.open", fake_open)

    url = open_dashboard_html("outputs/reports/dashboard.html")

    assert url.startswith("file:///")
    assert url.endswith("/outputs/reports/dashboard.html")
    assert opened_urls == [url]


def test_main_no_server_skips_server_start(monkeypatch) -> None:
    calls: dict[str, object] = {"wrote": False, "served": False}

    def fake_write_dashboard_html(**kwargs):
        calls["wrote"] = kwargs
        return Path(kwargs["output_path"])

    def fake_run_dashboard_server(**kwargs):
        calls["served"] = True

    monkeypatch.setattr("src.dashboard.app.write_dashboard_html", fake_write_dashboard_html)
    monkeypatch.setattr("src.dashboard.app.run_dashboard_server", fake_run_dashboard_server)
    monkeypatch.setattr(sys, "argv", ["app.py", "--no-server", "--dashboard-path", "outputs/reports/test_dashboard.html"])

    main()

    assert calls["wrote"] is not False
    assert calls["served"] is False


def test_main_no_server_with_open_calls_open_helper(monkeypatch) -> None:
    calls: dict[str, object] = {"open_path": None, "served": False}

    def fake_write_dashboard_html(**kwargs):
        return Path(kwargs["output_path"])

    def fake_open_dashboard_html(path: str):
        calls["open_path"] = path
        return "file:///dummy"

    def fake_run_dashboard_server(**kwargs):
        calls["served"] = True

    monkeypatch.setattr("src.dashboard.app.write_dashboard_html", fake_write_dashboard_html)
    monkeypatch.setattr("src.dashboard.app.open_dashboard_html", fake_open_dashboard_html)
    monkeypatch.setattr("src.dashboard.app.run_dashboard_server", fake_run_dashboard_server)
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.py", "--no-server", "--open", "--dashboard-path", "outputs/reports/static.html"],
    )

    main()

    assert calls["open_path"] == "outputs/reports/static.html"
    assert calls["served"] is False


def test_main_default_runs_server_with_dashboard_root(monkeypatch) -> None:
    calls: dict[str, object] = {"served": None}

    def fake_write_dashboard_html(**kwargs):
        return Path(kwargs["output_path"])

    def fake_run_dashboard_server(**kwargs):
        calls["served"] = kwargs

    monkeypatch.setattr("src.dashboard.app.write_dashboard_html", fake_write_dashboard_html)
    monkeypatch.setattr("src.dashboard.app.run_dashboard_server", fake_run_dashboard_server)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app.py",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--dashboard-path",
            "outputs/reports/custom_dashboard.html",
        ],
    )

    main()

    assert calls["served"] == {
        "host": "0.0.0.0",
        "port": 9001,
        "directory": Path("outputs"),
    }
