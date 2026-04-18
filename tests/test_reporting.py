import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.dashboard.reporting import build_phase1_report_markdown, write_phase1_report


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annualized_return": [0.10],
            "annualized_volatility": [0.12],
            "downside_volatility": [0.08],
            "sharpe_ratio": [0.8],
            "sortino_ratio": [1.2],
            "max_drawdown": [-0.2],
            "calmar_ratio": [0.5],
            "ending_nav": [1.5],
            "total_turnover": [2.0],
            "total_transaction_cost_drag": [0.001],
        },
        index=pd.Index(["balanced"], name="portfolio"),
    )


def test_build_phase1_report_markdown_contains_key_sections() -> None:
    performance_summary = _sample_frame()
    turnover_summary = pd.DataFrame(
        {
            "total_turnover": [2.0],
            "average_turnover": [0.01],
            "rebalance_count": [4],
            "total_transaction_cost_drag": [0.001],
        },
        index=pd.Index(["balanced"], name="portfolio"),
    )
    annual_return_table = pd.DataFrame({"portfolio": [0.10]}, index=pd.Index([2024], name="year"))
    benchmark_comparisons = pd.DataFrame({"tracking_error": [0.02]}, index=pd.Index(["benchmark_a"]))
    liquidity_table = pd.DataFrame(
        {"passes_liquidity_filter": [True, False]},
        index=pd.Index(["VTI", "IAGG"], name="ticker"),
    )
    etf_summary = pd.DataFrame(
        {
            "asset_class": ["us_equity"],
            "average_dollar_volume": [1_000_000.0],
            "recent_pass_ratio": [1.0],
            "passes_liquidity_filter": [True],
            "phase1_total_score": [40.0],
            "phase1_rank": [1],
        },
        index=pd.Index(["VTI"], name="ticker"),
    )
    chart_paths = {"nav_chart": Path("outputs/figures/balanced_nav.png")}

    report = build_phase1_report_markdown(
        strategy_name="balanced",
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=annual_return_table,
        benchmark_comparisons=benchmark_comparisons,
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        chart_paths=chart_paths,
        report_date="2026-04-18",
        notes=["IAGG failed the liquidity filter"],
    )

    assert "# Phase 1 Pipeline Report" in report
    assert "## Executive Summary" in report
    assert "IAGG failed the liquidity filter" in report
    assert "balanced_nav.png" in report


def test_write_phase1_report_creates_markdown_file() -> None:
    output_dir = Path("data/cache") / f"test_report_{uuid.uuid4().hex}"
    output_path = output_dir / "phase1_report.md"

    performance_summary = _sample_frame()
    turnover_summary = pd.DataFrame(
        {
            "total_turnover": [2.0],
            "average_turnover": [0.01],
            "rebalance_count": [4],
            "total_transaction_cost_drag": [0.001],
        },
        index=pd.Index(["balanced"], name="portfolio"),
    )
    annual_return_table = pd.DataFrame({"portfolio": [0.10]}, index=pd.Index([2024], name="year"))
    benchmark_comparisons = pd.DataFrame({"tracking_error": [0.02]}, index=pd.Index(["benchmark_a"]))
    liquidity_table = pd.DataFrame(
        {"passes_liquidity_filter": [True]},
        index=pd.Index(["VTI"], name="ticker"),
    )
    etf_summary = pd.DataFrame(
        {
            "asset_class": ["us_equity"],
            "average_dollar_volume": [1_000_000.0],
            "recent_pass_ratio": [1.0],
            "passes_liquidity_filter": [True],
            "phase1_total_score": [40.0],
            "phase1_rank": [1],
        },
        index=pd.Index(["VTI"], name="ticker"),
    )

    try:
        result = write_phase1_report(
            strategy_name="balanced",
            performance_summary=performance_summary,
            turnover_summary=turnover_summary,
            annual_return_table=annual_return_table,
            benchmark_comparisons=benchmark_comparisons,
            liquidity_table=liquidity_table,
            etf_summary=etf_summary,
            chart_paths={"nav_chart": Path("outputs/figures/balanced_nav.png")},
            output_path=output_path,
            report_date="2026-04-18",
            notes=None,
        )
        assert result.exists()
        assert "Phase 1 Pipeline Report" in result.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
