import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.dashboard.reporting import (
    build_rebalance_reason_summary,
    build_asset_risk_snapshot,
    build_latest_rolling_metric_snapshot,
    build_phase1_report_html,
    build_phase1_report_markdown,
    build_phase1_risk_summary_tables,
    build_run_configuration_summary,
    build_top_correlation_summary,
    write_phase1_html_report,
    write_phase1_report,
)


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


def _sample_data_quality_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_date": ["2024-01-02"],
            "end_date": ["2024-01-03"],
            "observations": [2],
            "missing_adj_close": [0],
            "missing_volume": [0],
            "zero_volume": [0],
            "missing_dollar_volume": [0],
            "has_duplicate_dates": [False],
        },
        index=pd.Index(["VTI"], name="ticker"),
    )


def _sample_run_configuration() -> pd.DataFrame:
    return build_run_configuration_summary(
        start="2024-01-01",
        end="2024-12-31",
        template_name=None,
        backtest_universe_mode="liquidity_filtered",
        rolling_window=63,
        config_paths={"universe": "config/etf_universe.yaml"},
    )


def _sample_trend_filter_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observations": [3],
            "trend_active_days": [2],
            "trend_active_ratio": [2.0 / 3.0],
            "avg_reduced_assets": [2.0 / 3.0],
            "max_reduced_assets": [1],
        },
        index=pd.Index(["balanced"], name="portfolio"),
    )


def _sample_risk_switch_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observations": [4],
            "risk_switch_active_days": [1],
            "risk_switch_active_ratio": [0.25],
            "avg_reduced_assets": [0.25],
            "max_reduced_assets": [1],
        },
        index=pd.Index(["balanced"], name="portfolio"),
    )


def _sample_rebalance_reason_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "balanced": ["none", "calendar", "calendar+drift", "none"],
            "benchmark_a": ["calendar", "none", "drift", "none"],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )


def _sample_risk_limit_checks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio": ["balanced"],
            "metric": ["max_drawdown"],
            "threshold": [0.25],
            "observed": [-0.20],
            "comparison_value": [0.20],
            "limit_enabled": [True],
            "breach": [False],
        },
        index=pd.Index(["balanced:max_drawdown"], name="rule_id"),
    )


def _sample_risk_limit_breaches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio": ["balanced"],
            "metric": ["max_drawdown"],
            "threshold": [0.25],
            "observed": [-0.30],
            "comparison_value": [0.30],
            "limit_enabled": [True],
            "breach": [True],
        },
        index=pd.Index(["balanced:max_drawdown"], name="rule_id"),
    )


def _sample_risk_limit_breach_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_enabled_checks": [2, 2, 4],
            "breached_checks": [1, 1, 2],
            "breach_ratio": [0.5, 0.5, 0.5],
        },
        index=pd.Index(["balanced", "benchmark_a", "overall"], name="portfolio"),
    )


def _sample_pipeline_health_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "missing_outputs": [0],
            "empty_outputs": [0],
            "risk_limit_breaches": [1],
            "fail_on_missing_outputs": [False],
            "fail_on_empty_outputs": [False],
            "fail_on_risk_limit_breach": [False],
            "would_fail_missing_outputs": [False],
            "would_fail_empty_outputs": [False],
            "would_fail_risk_limit_breach": [False],
            "run_passed_quality_gates": [True],
        },
        index=pd.Index(["pipeline"], name="summary"),
    )


def _sample_portfolio_risk_contribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weight": [0.6, 0.4],
            "marginal_contribution_to_risk": [0.12, 0.04],
            "absolute_risk_contribution": [0.072, 0.016],
            "percent_risk_contribution": [0.8182, 0.1818],
            "portfolio_volatility": [0.088, 0.088],
        },
        index=pd.Index(["VTI", "AGG"], name="asset"),
    )


def _sample_macro_regime_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of_date": "2024-12-31",
                "metric": "vix",
                "latest_value": 18.0,
                "reference_value": 25.0,
                "signal": "risk_on",
                "rule": "risk_off when latest VIX >= 25",
            },
            {
                "as_of_date": "2024-12-31",
                "metric": "composite_regime",
                "latest_value": 1.0,
                "reference_value": 4.0,
                "signal": "mixed",
                "rule": "risk_off if >=2 component risk_off signals; risk_on if >=2 risk_on; else mixed",
            },
        ]
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
    benchmark_annual_excess_returns = pd.DataFrame({"benchmark_a": [0.01]}, index=pd.Index([2024], name="year"))
    benchmark_drawdown_comparisons = pd.DataFrame(
        {
            "strategy_max_drawdown": [-0.2],
            "benchmark_max_drawdown": [-0.15],
            "max_drawdown_gap": [-0.05],
        },
        index=pd.Index(["benchmark_a"], name="benchmark"),
    )
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
    covariance_matrix = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0002]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_pairs = pd.DataFrame(
        [
            {"left": "VTI", "right": "VTI", "correlation": 1.0},
            {"left": "VTI", "right": "AGG", "correlation": 0.25},
            {"left": "AGG", "right": "AGG", "correlation": 1.0},
        ]
    )
    chart_paths = {"nav_chart": Path("outputs/figures/balanced_nav.png")}
    rolling_metric_snapshot = pd.DataFrame(
        {"balanced": [0.12, 0.8], "benchmark_a": [0.10, 0.7]},
        index=["latest_rolling_volatility", "latest_rolling_sharpe"],
    )
    rolling_correlation = pd.DataFrame(
        {"VTI_AGG_rolling_correlation": [0.25, 0.20, 0.15]},
        index=pd.Index(["2024-01-02", "2024-01-03", "2024-01-04"], name="date"),
    )

    report = build_phase1_report_markdown(
        strategy_name="balanced",
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=annual_return_table,
        benchmark_comparisons=benchmark_comparisons,
        benchmark_annual_excess_returns=benchmark_annual_excess_returns,
        benchmark_drawdown_comparisons=benchmark_drawdown_comparisons,
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=_sample_data_quality_summary(),
        covariance_matrix=covariance_matrix,
        correlation_matrix=correlation_matrix,
        correlation_pairs=correlation_pairs,
        chart_paths=chart_paths,
        report_date="2026-04-18",
        trend_filter_summary=_sample_trend_filter_summary(),
        risk_switch_summary=_sample_risk_switch_summary(),
        rolling_metric_snapshot=rolling_metric_snapshot,
        rolling_correlation=rolling_correlation,
        rebalance_reason_table=_sample_rebalance_reason_table(),
        risk_limit_checks=_sample_risk_limit_checks(),
        risk_limit_breaches=_sample_risk_limit_breaches(),
        risk_limit_breach_summary=_sample_risk_limit_breach_summary(),
        pipeline_health_summary=_sample_pipeline_health_summary(),
        portfolio_risk_contribution=_sample_portfolio_risk_contribution(),
        macro_regime_summary=_sample_macro_regime_summary(),
        run_configuration=_sample_run_configuration(),
        notes=["IAGG failed the liquidity filter"],
    )

    assert "# Phase 1 Pipeline Report" in report
    assert "## Executive Summary" in report
    assert "IAGG failed the liquidity filter" in report
    assert "balanced_nav.png" in report
    assert "## Correlation Highlights" in report
    assert "VTI vs AGG" in report
    assert "## Benchmark Annual Excess Returns" in report
    assert "## Benchmark Drawdown Comparisons" in report
    assert "## Latest Rolling Metrics" in report
    assert "## Rolling Correlation (VTI vs AGG)" in report
    assert "## Trend Filter Summary" in report
    assert "## Risk Switch Summary" in report
    assert "## Rebalance Reason Summary" in report
    assert "## Recent Rebalance Events" in report
    assert "## Risk Limit Checks" in report
    assert "## Risk Limit Breaches" in report
    assert "## Risk Limit Breach Summary" in report
    assert "## Pipeline Health Summary" in report
    assert "## Macro Regime Summary" in report
    assert "## Portfolio Risk Contribution" in report
    assert "## Data Quality Summary" in report
    assert "## Run Configuration" in report
    assert "config\\etf_universe.yaml" in report
    assert "missing_volume" in report
    assert "12.00%" in report
    assert "0.2500" in report
    assert "25.00%" in report
    assert "25.00%" in report
    assert "30.00%" in report
    assert "50.00%" in report
    assert "calendar+drift" in report
    assert "run_passed_quality_gates" in report
    assert "percent_risk_contribution" in report
    assert "composite_regime" in report


def test_build_phase1_report_html_contains_key_sections() -> None:
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
    benchmark_annual_excess_returns = pd.DataFrame({"benchmark_a": [0.01]}, index=pd.Index([2024], name="year"))
    benchmark_drawdown_comparisons = pd.DataFrame(
        {
            "strategy_max_drawdown": [-0.2],
            "benchmark_max_drawdown": [-0.15],
            "max_drawdown_gap": [-0.05],
        },
        index=pd.Index(["benchmark_a"], name="benchmark"),
    )
    liquidity_table = pd.DataFrame(
        {"passes_liquidity_filter": [True, False]},
        index=pd.Index(["VTI", "BNDX"], name="ticker"),
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
    covariance_matrix = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0002]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_pairs = pd.DataFrame(
        [
            {"left": "VTI", "right": "VTI", "correlation": 1.0},
            {"left": "VTI", "right": "AGG", "correlation": 0.25},
        ]
    )

    report = build_phase1_report_html(
        strategy_name="balanced",
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=annual_return_table,
        benchmark_comparisons=benchmark_comparisons,
        benchmark_annual_excess_returns=benchmark_annual_excess_returns,
        benchmark_drawdown_comparisons=benchmark_drawdown_comparisons,
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=_sample_data_quality_summary(),
        covariance_matrix=covariance_matrix,
        correlation_matrix=correlation_matrix,
        correlation_pairs=correlation_pairs,
        chart_paths={"nav_chart": Path("outputs/figures/balanced_nav.png")},
        report_date="2026-04-19",
        trend_filter_summary=_sample_trend_filter_summary(),
        risk_switch_summary=_sample_risk_switch_summary(),
        run_configuration=_sample_run_configuration(),
        rolling_metric_snapshot=pd.DataFrame(
            {"balanced": [0.12, 0.8]},
            index=["latest_rolling_volatility", "latest_rolling_sharpe"],
        ),
        rolling_correlation=pd.DataFrame(
            {"VTI_AGG_rolling_correlation": [0.25, 0.20]},
            index=pd.Index(["2024-01-02", "2024-01-03"], name="date"),
        ),
        rebalance_reason_table=_sample_rebalance_reason_table(),
        risk_limit_checks=_sample_risk_limit_checks(),
        risk_limit_breaches=_sample_risk_limit_breaches(),
        risk_limit_breach_summary=_sample_risk_limit_breach_summary(),
        pipeline_health_summary=_sample_pipeline_health_summary(),
        portfolio_risk_contribution=_sample_portfolio_risk_contribution(),
        macro_regime_summary=_sample_macro_regime_summary(),
        notes=["Backtest universe mode: liquidity_filtered"],
    )

    assert "<html" in report
    assert "<h2>Correlation Highlights</h2>" in report
    assert "balanced_nav.png" in report
    assert "<h2>Benchmark Annual Excess Returns</h2>" in report
    assert "<h2>Benchmark Drawdown Comparisons</h2>" in report
    assert "<h2>Latest Rolling Metrics</h2>" in report
    assert "<h2>Rolling Correlation (VTI vs AGG)</h2>" in report
    assert "<h2>Trend Filter Summary</h2>" in report
    assert "<h2>Risk Switch Summary</h2>" in report
    assert "<h2>Rebalance Reason Summary</h2>" in report
    assert "<h2>Recent Rebalance Events</h2>" in report
    assert "<h2>Risk Limit Checks</h2>" in report
    assert "<h2>Risk Limit Breaches</h2>" in report
    assert "<h2>Risk Limit Breach Summary</h2>" in report
    assert "<h2>Pipeline Health Summary</h2>" in report
    assert "<h2>Macro Regime Summary</h2>" in report
    assert "<h2>Portfolio Risk Contribution</h2>" in report
    assert "<h2>Data Quality Summary</h2>" in report
    assert "<h2>Run Configuration</h2>" in report
    assert "config\\etf_universe.yaml" in report
    assert "missing_volume" in report
    assert "12.00%" in report
    assert "0.2500" in report
    assert "25.00%" in report
    assert "25.00%" in report
    assert "30.00%" in report
    assert "50.00%" in report
    assert "calendar+drift" in report
    assert "run_passed_quality_gates" in report
    assert "percent_risk_contribution" in report
    assert "composite_regime" in report


def test_build_rebalance_reason_summary_counts_trigger_types() -> None:
    summary = build_rebalance_reason_summary(_sample_rebalance_reason_table())

    assert int(summary.loc["balanced", "total_days"]) == 4
    assert int(summary.loc["balanced", "rebalance_days"]) == 2
    assert float(summary.loc["balanced", "rebalance_ratio"]) == 0.5
    assert int(summary.loc["balanced", "calendar_days"]) == 2
    assert int(summary.loc["balanced", "drift_days"]) == 1
    assert int(summary.loc["balanced", "calendar_and_drift_days"]) == 1


def test_risk_summary_helpers_build_expected_tables() -> None:
    correlation_pairs = pd.DataFrame(
        [
            {"left": "VTI", "right": "VTI", "correlation": 1.0},
            {"left": "VTI", "right": "AGG", "correlation": 0.25},
            {"left": "VTI", "right": "GLD", "correlation": -0.4},
        ]
    )
    correlation_summary = build_top_correlation_summary(correlation_pairs, top_n=1)
    assert correlation_summary.iloc[0]["pair"] == "VTI vs GLD"

    correlation_matrix = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    covariance_matrix = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0002]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    snapshot = build_asset_risk_snapshot(correlation_matrix, covariance_matrix)
    assert snapshot.loc["VTI", "avg_correlation"] == "0.2500"
    assert snapshot.loc["AGG", "variance"] == "0.000200"

    tables = build_phase1_risk_summary_tables(
        correlation_pairs=correlation_pairs,
        correlation_matrix=correlation_matrix,
        covariance_matrix=covariance_matrix,
    )
    assert "top_correlation_pairs" in tables
    assert "asset_risk_snapshot" in tables


def test_build_latest_rolling_metric_snapshot_uses_latest_non_empty_rows() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    rolling_volatility = pd.DataFrame(
        {"balanced": [None, 0.10, 0.11, 0.12]},
        index=index,
    )
    rolling_sharpe = pd.DataFrame(
        {"balanced": [None, None, 0.7, 0.8]},
        index=index,
    )

    snapshot = build_latest_rolling_metric_snapshot(rolling_volatility, rolling_sharpe)

    assert snapshot.loc["latest_rolling_volatility", "balanced"] == 0.12
    assert snapshot.loc["latest_rolling_sharpe", "balanced"] == 0.8


def test_build_run_configuration_summary_contains_parameters_and_configs() -> None:
    summary = _sample_run_configuration()

    assert summary.loc["template_name", "value"] == "default"
    assert summary.loc["backtest_universe_mode", "value"] == "liquidity_filtered"
    assert summary.loc["rolling_window", "value"] == 63
    assert summary.loc["config_universe", "value"] == "config\\etf_universe.yaml"


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
    benchmark_annual_excess_returns = pd.DataFrame({"benchmark_a": [0.01]}, index=pd.Index([2024], name="year"))
    benchmark_drawdown_comparisons = pd.DataFrame(
        {
            "strategy_max_drawdown": [-0.2],
            "benchmark_max_drawdown": [-0.15],
            "max_drawdown_gap": [-0.05],
        },
        index=pd.Index(["benchmark_a"], name="benchmark"),
    )
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
    covariance_matrix = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0002]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_pairs = pd.DataFrame(
        [
            {"left": "VTI", "right": "VTI", "correlation": 1.0},
            {"left": "VTI", "right": "AGG", "correlation": 0.25},
        ]
    )

    try:
        result = write_phase1_report(
            strategy_name="balanced",
            performance_summary=performance_summary,
            turnover_summary=turnover_summary,
            annual_return_table=annual_return_table,
            benchmark_comparisons=benchmark_comparisons,
            benchmark_annual_excess_returns=benchmark_annual_excess_returns,
            benchmark_drawdown_comparisons=benchmark_drawdown_comparisons,
            liquidity_table=liquidity_table,
            etf_summary=etf_summary,
            data_quality_summary=_sample_data_quality_summary(),
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            correlation_pairs=correlation_pairs,
            chart_paths={"nav_chart": Path("outputs/figures/balanced_nav.png")},
            output_path=output_path,
            report_date="2026-04-18",
            trend_filter_summary=_sample_trend_filter_summary(),
            risk_switch_summary=_sample_risk_switch_summary(),
            run_configuration=_sample_run_configuration(),
            risk_limit_checks=_sample_risk_limit_checks(),
            risk_limit_breaches=_sample_risk_limit_breaches(),
            risk_limit_breach_summary=_sample_risk_limit_breach_summary(),
            pipeline_health_summary=_sample_pipeline_health_summary(),
            portfolio_risk_contribution=_sample_portfolio_risk_contribution(),
            notes=None,
        )
        assert result.exists()
        assert "Phase 1 Pipeline Report" in result.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_write_phase1_html_report_creates_html_file() -> None:
    output_dir = Path("data/cache") / f"test_html_report_{uuid.uuid4().hex}"
    output_path = output_dir / "phase1_report.html"

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
    benchmark_annual_excess_returns = pd.DataFrame({"benchmark_a": [0.01]}, index=pd.Index([2024], name="year"))
    benchmark_drawdown_comparisons = pd.DataFrame(
        {
            "strategy_max_drawdown": [-0.2],
            "benchmark_max_drawdown": [-0.15],
            "max_drawdown_gap": [-0.05],
        },
        index=pd.Index(["benchmark_a"], name="benchmark"),
    )
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
    covariance_matrix = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0002]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )
    correlation_pairs = pd.DataFrame(
        [
            {"left": "VTI", "right": "VTI", "correlation": 1.0},
            {"left": "VTI", "right": "AGG", "correlation": 0.25},
        ]
    )

    try:
        result = write_phase1_html_report(
            strategy_name="balanced",
            performance_summary=performance_summary,
            turnover_summary=turnover_summary,
            annual_return_table=annual_return_table,
            benchmark_comparisons=benchmark_comparisons,
            benchmark_annual_excess_returns=benchmark_annual_excess_returns,
            benchmark_drawdown_comparisons=benchmark_drawdown_comparisons,
            liquidity_table=liquidity_table,
            etf_summary=etf_summary,
            data_quality_summary=_sample_data_quality_summary(),
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            correlation_pairs=correlation_pairs,
            chart_paths={"nav_chart": Path("outputs/figures/balanced_nav.png")},
            output_path=output_path,
            report_date="2026-04-19",
            trend_filter_summary=_sample_trend_filter_summary(),
            risk_switch_summary=_sample_risk_switch_summary(),
            run_configuration=_sample_run_configuration(),
            risk_limit_checks=_sample_risk_limit_checks(),
            risk_limit_breaches=_sample_risk_limit_breaches(),
            risk_limit_breach_summary=_sample_risk_limit_breach_summary(),
            pipeline_health_summary=_sample_pipeline_health_summary(),
            portfolio_risk_contribution=_sample_portfolio_risk_contribution(),
            notes=None,
        )
        assert result.exists()
        assert "<!DOCTYPE html>" in result.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
