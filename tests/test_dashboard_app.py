import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.dashboard.app import build_dashboard_html, write_dashboard_html


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
        pd.DataFrame({"asset_class": ["us_equity"]}, index=pd.Index(["VTI"])).to_csv(table_dir / "etf_summary.csv")
        (figure_dir / "balanced_nav.png").write_bytes(b"fake")
        (report_dir / "balanced_phase1_report.html").write_text("<html></html>", encoding="utf-8")

        html = build_dashboard_html(output_dir=table_dir, figure_dir=figure_dir, report_dir=report_dir)

        assert "AA ETF Research Dashboard" in html
        assert "Performance Summary" in html
        assert "Benchmark Annual Excess Returns" in html
        assert "balanced_nav.png" in html
        assert "balanced_phase1_report.html" in html
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


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
