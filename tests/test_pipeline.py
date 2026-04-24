import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

from run_pipeline import (
    build_argument_parser,
    main,
    save_processed_frames,
    write_data_quality_outputs,
    write_liquidity_outputs,
    write_macro_outputs,
)


def test_argument_parser_exposes_rolling_window_default_and_override() -> None:
    parser = build_argument_parser()

    default_args = parser.parse_args(["--start", "2024-01-01"])
    override_args = parser.parse_args(["--start", "2024-01-01", "--rolling-window", "21"])
    fail_args = parser.parse_args(["--start", "2024-01-01", "--fail-on-missing-outputs"])
    deterministic_args = parser.parse_args(["--start", "2024-01-01", "--as-of-date", "2024-12-31", "--seed", "7"])
    risk_limit_args = parser.parse_args(["--start", "2024-01-01", "--fail-on-risk-limit-breach"])

    assert default_args.rolling_window == 63
    assert default_args.risk_limits_config == "config/risk_limits.yaml"
    assert default_args.macro_dir == "data/macro"
    assert default_args.download_retries == 3
    assert default_args.download_retry_delay == 1.0
    assert default_args.fail_on_missing_outputs is False
    assert default_args.fail_on_empty_outputs is False
    assert override_args.rolling_window == 21
    assert fail_args.fail_on_missing_outputs is True
    assert deterministic_args.as_of_date == "2024-12-31"
    assert deterministic_args.seed == 7
    assert risk_limit_args.fail_on_risk_limit_breach is True


def test_save_processed_frames_writes_per_ticker_csv() -> None:
    output_dir = Path("data/cache") / f"test_processed_frames_{uuid.uuid4().hex}"
    frame = pd.DataFrame(
        {"adj_close": [100.0], "volume": [1000], "dollar_volume": [100000.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    frame.index.name = "date"

    try:
        save_processed_frames({"VTI": frame}, output_dir)
        assert (output_dir / "VTI.csv").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_write_data_quality_outputs_writes_summary_file() -> None:
    output_dir = Path("data/cache") / f"test_data_quality_outputs_{uuid.uuid4().hex}"
    frame = pd.DataFrame(
        {"adj_close": [100.0], "volume": [1000.0], "dollar_volume": [100000.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    frame.index.name = "date"

    try:
        summary = write_data_quality_outputs({"VTI": frame}, output_dir)
        assert (output_dir / "data_quality_summary.csv").exists()
        assert summary.loc["VTI", "observations"] == 1
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_write_macro_outputs_writes_summary_file() -> None:
    output_dir = Path("data/cache") / f"test_macro_outputs_{uuid.uuid4().hex}"
    macro = pd.DataFrame(
        {"us10y_yield": [4.2], "vix": [15.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    macro.index.name = "date"

    try:
        path = write_macro_outputs(macro, output_dir)
        assert path.exists()
        loaded = pd.read_csv(path, index_col=0)
        assert "us10y_yield" in loaded.columns
        assert "vix" in loaded.columns
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_write_liquidity_outputs_writes_expected_files() -> None:
    output_dir = Path("data/cache") / f"test_liquidity_outputs_{uuid.uuid4().hex}"
    summary = pd.DataFrame(
        {"passes_liquidity_filter": [True]},
        index=pd.Index(["VTI"], name="ticker"),
    )
    etf_summary = pd.DataFrame(
        {"phase1_total_score": [40.0]},
        index=pd.Index(["VTI"], name="ticker"),
    )

    try:
        write_liquidity_outputs(["VTI"], summary, etf_summary, output_dir)
        assert (output_dir / "liquidity_summary.csv").exists()
        assert (output_dir / "etf_summary.csv").exists()
        assert (output_dir / "investable_universe.csv").read_text(encoding="utf-8") == "ticker\nVTI\n"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_main_fail_on_missing_outputs_raises_runtime_error(monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    asset_returns = pd.DataFrame({"VTI": [0.0, 0.0]}, index=index)
    clean_frame = pd.DataFrame(
        {
            "adj_close": [100.0, 101.0],
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_000_000],
            "dollar_volume": [100_000_000.0, 101_000_000.0],
        },
        index=index,
    )
    clean_frame.index.name = "date"
    clean_frames = {"VTI": clean_frame}

    strategy_result = {
        "portfolio_returns": pd.Series([0.0, 0.0], index=index),
        "portfolio_nav": pd.Series([1.0, 1.0], index=index),
        "turnover": pd.Series([0.0, 0.0], index=index),
        "transaction_costs": pd.Series([0.0, 0.0], index=index),
        "summary": pd.Series({"annualized_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0}),
        "annual_return_table": pd.DataFrame({"portfolio": [0.0]}, index=pd.Index([2024], name="year")),
        "benchmark_comparisons": pd.DataFrame(),
        "benchmark_annual_excess_returns": pd.DataFrame(),
        "benchmark_drawdown_comparisons": pd.DataFrame(),
    }

    monkeypatch.setattr("run_pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.validate_date_range", lambda start, end: (start, end))
    monkeypatch.setattr("run_pipeline.load_enabled_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.fetch_prices", lambda *args, **kwargs: {"VTI": clean_frame})
    monkeypatch.setattr("run_pipeline.batch_clean_price_frames", lambda frames: clean_frames)
    monkeypatch.setattr("run_pipeline.save_processed_frames", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.write_data_quality_outputs",
        lambda *args, **kwargs: pd.DataFrame({"observations": [2]}, index=pd.Index(["VTI"], name="ticker")),
    )
    monkeypatch.setattr(
        "run_pipeline.fetch_macro_series",
        lambda *args, **kwargs: pd.DataFrame({"vix": [15.0]}, index=index[:1]),
    )
    monkeypatch.setattr("run_pipeline.save_macro_series_per_symbol", lambda *args, **kwargs: {})
    monkeypatch.setattr("run_pipeline.write_macro_outputs", lambda *args, **kwargs: Path("outputs/tables/macro_observation_summary.csv"))
    monkeypatch.setattr(
        "run_pipeline.filter_liquid_universe",
        lambda *args, **kwargs: (
            ["VTI"],
            pd.DataFrame({"passes_liquidity_filter": [True]}, index=pd.Index(["VTI"], name="ticker")),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.score_etf_universe",
        lambda *args, **kwargs: pd.DataFrame({"phase1_total_score": [40.0]}, index=pd.Index(["VTI"], name="ticker")),
    )
    monkeypatch.setattr("run_pipeline.write_liquidity_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.collect_required_backtest_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.warn_on_non_liquid_required_assets", lambda *args, **kwargs: [])
    monkeypatch.setattr("run_pipeline.resolve_backtest_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.build_asset_return_matrix", lambda *args, **kwargs: asset_returns)
    monkeypatch.setattr("run_pipeline.build_adjusted_close_matrix", lambda *args, **kwargs: pd.DataFrame({"VTI": [100.0, 101.0]}, index=index))
    monkeypatch.setattr(
        "run_pipeline.build_trend_filter_overlay_settings",
        lambda *args, **kwargs: {"enabled": False, "moving_average_days": 210, "reduction_fraction": 0.5, "assets": []},
    )
    monkeypatch.setattr("run_pipeline.run_strategy_backtests", lambda *args, **kwargs: ("balanced", strategy_result, {}))
    monkeypatch.setattr(
        "run_pipeline.build_backtest_policy_tables",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr("run_pipeline.write_backtest_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.build_return_table",
        lambda *args, **kwargs: pd.DataFrame({"balanced": [0.0, 0.0]}, index=index),
    )
    monkeypatch.setattr(
        "run_pipeline.build_rolling_metric_outputs",
        lambda *args, **kwargs: {
            "rolling_volatility": pd.DataFrame({"balanced": [0.0, 0.0]}, index=index),
            "rolling_sharpe": pd.DataFrame({"balanced": [0.0, 0.0]}, index=index),
            "drawdown_series": pd.DataFrame({"balanced": [0.0, 0.0]}, index=index),
        },
    )
    monkeypatch.setattr("run_pipeline.write_rolling_metric_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.build_nav_table",
        lambda *args, **kwargs: pd.DataFrame({"balanced": [1.0, 1.0]}, index=index),
    )
    monkeypatch.setattr(
        "run_pipeline.write_phase1_chart_outputs",
        lambda *args, **kwargs: {"nav_chart": Path("outputs/figures/balanced_nav.png")},
    )
    monkeypatch.setattr(
        "run_pipeline.build_performance_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "annualized_return": [0.0],
                "annualized_volatility": [0.0],
                "max_drawdown": [0.0],
                "ending_nav": [1.0],
            },
            index=pd.Index(["balanced"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.build_turnover_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"total_turnover": [0.0], "total_transaction_cost_drag": [0.0]},
            index=pd.Index(["balanced"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.build_risk_matrix_outputs",
        lambda *args, **kwargs: {
            "covariance_matrix": pd.DataFrame({"VTI": [0.0]}, index=["VTI"]),
            "correlation_matrix": pd.DataFrame({"VTI": [1.0]}, index=["VTI"]),
            "correlation_pairs": pd.DataFrame([{"left": "VTI", "right": "VTI", "correlation": 1.0}]),
        },
    )
    monkeypatch.setattr(
        "run_pipeline.build_latest_rolling_metric_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"balanced": [0.0]}, index=["latest_rolling_volatility"]),
    )
    monkeypatch.setattr(
        "run_pipeline.load_risk_limits",
        lambda *args, **kwargs: {"annualized_volatility_warning": None, "max_drawdown_warning": None},
    )
    monkeypatch.setattr(
        "run_pipeline.build_portfolio_risk_limit_checks",
        lambda *args, **kwargs: pd.DataFrame(
            {"portfolio": ["balanced"], "metric": ["max_drawdown"], "breach": [False]},
            index=pd.Index(["balanced:max_drawdown"], name="rule_id"),
        ),
    )
    monkeypatch.setattr("run_pipeline.write_risk_limit_output", lambda *args, **kwargs: Path("outputs/tables/risk_limit_checks.csv"))
    monkeypatch.setattr("run_pipeline.write_risk_limit_breaches_output", lambda *args, **kwargs: Path("outputs/tables/risk_limit_breaches.csv"))
    monkeypatch.setattr(
        "run_pipeline.build_risk_limit_breach_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"total_enabled_checks": [1], "breached_checks": [0], "breach_ratio": [0.0]},
            index=pd.Index(["overall"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.write_risk_limit_breach_summary_output",
        lambda *args, **kwargs: Path("outputs/tables/risk_limit_breach_summary.csv"),
    )
    monkeypatch.setattr("run_pipeline.find_risk_limit_breaches", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("run_pipeline.validate_risk_limit_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.build_run_configuration_summary",
        lambda *args, **kwargs: pd.DataFrame({"value": ["2024-01-01"]}, index=["start"]),
    )
    monkeypatch.setattr("run_pipeline.write_run_configuration_output", lambda *args, **kwargs: Path("outputs/tables/run_configuration.csv"))
    monkeypatch.setattr("run_pipeline.write_phase1_report", lambda *args, **kwargs: Path("outputs/reports/balanced_phase1_report.md"))
    monkeypatch.setattr("run_pipeline.write_phase1_html_report", lambda *args, **kwargs: Path("outputs/reports/balanced_phase1_report.html"))
    monkeypatch.setattr(
        "run_pipeline.collect_table_output_paths",
        lambda *args, **kwargs: {"performance_summary": Path("outputs/tables/performance_summary.csv")},
    )
    monkeypatch.setattr("run_pipeline.build_pipeline_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr("run_pipeline.write_pipeline_manifest", lambda *args, **kwargs: Path("outputs/tables/pipeline_manifest.json"))
    monkeypatch.setattr("run_pipeline.build_output_inventory", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("run_pipeline.write_output_inventory", lambda *args, **kwargs: Path("outputs/tables/output_inventory.csv"))
    monkeypatch.setattr(
        "run_pipeline.build_pipeline_health_summary",
        lambda *args, **kwargs: pd.DataFrame({"run_passed_quality_gates": [False]}, index=pd.Index(["health"])),
    )
    monkeypatch.setattr(
        "run_pipeline.write_pipeline_health_summary",
        lambda *args, **kwargs: Path("outputs/tables/pipeline_health_summary.csv"),
    )
    monkeypatch.setattr(
        "run_pipeline.find_missing_output_inventory_entries",
        lambda *_args, **_kwargs: pd.DataFrame(
            [{"output_type": "report", "name": "balanced_phase1_report", "exists": False, "size_bytes": 0}]
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--start", "2024-01-01", "--fail-on-missing-outputs"])

    try:
        main()
    except RuntimeError as exc:
        assert "Output inventory validation failed" in str(exc)
        assert "report:balanced_phase1_report" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when --fail-on-missing-outputs is set and outputs are missing.")


def test_main_passes_rebalance_reason_table_to_reports(monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    asset_returns = pd.DataFrame({"VTI": [0.0, 0.0]}, index=index)
    clean_frame = pd.DataFrame(
        {
            "adj_close": [100.0, 101.0],
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_000_000],
            "dollar_volume": [100_000_000.0, 101_000_000.0],
        },
        index=index,
    )
    clean_frame.index.name = "date"
    clean_frames = {"VTI": clean_frame}

    strategy_result = {
        "portfolio_returns": pd.Series([0.0, 0.0], index=index),
        "portfolio_nav": pd.Series([1.0, 1.0], index=index),
        "turnover": pd.Series([0.0, 0.0], index=index),
        "transaction_costs": pd.Series([0.0, 0.0], index=index),
        "summary": pd.Series({"annualized_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0}),
        "annual_return_table": pd.DataFrame({"portfolio": [0.0], "benchmark_a": [0.0]}, index=pd.Index([2024], name="year")),
        "benchmark_comparisons": pd.DataFrame({"tracking_error": [0.0]}, index=pd.Index(["benchmark_a"])),
        "benchmark_annual_excess_returns": pd.DataFrame({"benchmark_a": [0.0]}, index=pd.Index([2024], name="year")),
        "benchmark_drawdown_comparisons": pd.DataFrame(
            {"strategy_max_drawdown": [0.0], "benchmark_max_drawdown": [0.0], "max_drawdown_gap": [0.0]},
            index=pd.Index(["benchmark_a"]),
        ),
        "rebalance_reasons": pd.Series(["initial", "calendar"], index=index),
    }
    benchmark_results = {
        "benchmark_a": {
            "portfolio_returns": pd.Series([0.0, 0.0], index=index),
            "portfolio_nav": pd.Series([1.0, 1.0], index=index),
            "turnover": pd.Series([0.0, 0.0], index=index),
            "transaction_costs": pd.Series([0.0, 0.0], index=index),
            "summary": pd.Series({"annualized_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0}),
            "rebalance_reasons": pd.Series(["none", "drift"], index=index),
        }
    }

    captured_markdown_kwargs: dict[str, object] = {}
    captured_html_kwargs: dict[str, object] = {}

    monkeypatch.setattr("run_pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.validate_date_range", lambda start, end: (start, end))
    monkeypatch.setattr("run_pipeline.load_enabled_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.fetch_prices", lambda *args, **kwargs: {"VTI": clean_frame})
    monkeypatch.setattr("run_pipeline.batch_clean_price_frames", lambda frames: clean_frames)
    monkeypatch.setattr("run_pipeline.save_processed_frames", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.write_data_quality_outputs",
        lambda *args, **kwargs: pd.DataFrame({"observations": [2]}, index=pd.Index(["VTI"], name="ticker")),
    )
    monkeypatch.setattr(
        "run_pipeline.fetch_macro_series",
        lambda *args, **kwargs: pd.DataFrame({"vix": [15.0]}, index=index[:1]),
    )
    monkeypatch.setattr("run_pipeline.save_macro_series_per_symbol", lambda *args, **kwargs: {})
    monkeypatch.setattr("run_pipeline.write_macro_outputs", lambda *args, **kwargs: Path("outputs/tables/macro_observation_summary.csv"))
    monkeypatch.setattr(
        "run_pipeline.filter_liquid_universe",
        lambda *args, **kwargs: (
            ["VTI"],
            pd.DataFrame({"passes_liquidity_filter": [True]}, index=pd.Index(["VTI"], name="ticker")),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.score_etf_universe",
        lambda *args, **kwargs: pd.DataFrame({"phase1_total_score": [40.0]}, index=pd.Index(["VTI"], name="ticker")),
    )
    monkeypatch.setattr("run_pipeline.write_liquidity_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.collect_required_backtest_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.warn_on_non_liquid_required_assets", lambda *args, **kwargs: [])
    monkeypatch.setattr("run_pipeline.resolve_backtest_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.build_asset_return_matrix", lambda *args, **kwargs: asset_returns)
    monkeypatch.setattr("run_pipeline.build_adjusted_close_matrix", lambda *args, **kwargs: pd.DataFrame({"VTI": [100.0, 101.0]}, index=index))
    monkeypatch.setattr(
        "run_pipeline.build_trend_filter_overlay_settings",
        lambda *args, **kwargs: {"enabled": False, "moving_average_days": 210, "reduction_fraction": 0.5, "assets": []},
    )
    monkeypatch.setattr(
        "run_pipeline.run_strategy_backtests",
        lambda *args, **kwargs: ("balanced", strategy_result, benchmark_results),
    )
    monkeypatch.setattr("run_pipeline.build_backtest_policy_tables", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr("run_pipeline.write_backtest_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.build_return_table",
        lambda *args, **kwargs: pd.DataFrame({"balanced": [0.0, 0.0], "benchmark_a": [0.0, 0.0]}, index=index),
    )
    monkeypatch.setattr(
        "run_pipeline.build_rolling_metric_outputs",
        lambda *args, **kwargs: {
            "rolling_volatility": pd.DataFrame({"balanced": [0.0, 0.0], "benchmark_a": [0.0, 0.0]}, index=index),
            "rolling_sharpe": pd.DataFrame({"balanced": [0.0, 0.0], "benchmark_a": [0.0, 0.0]}, index=index),
            "drawdown_series": pd.DataFrame({"balanced": [0.0, 0.0], "benchmark_a": [0.0, 0.0]}, index=index),
        },
    )
    monkeypatch.setattr("run_pipeline.write_rolling_metric_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.build_nav_table",
        lambda *args, **kwargs: pd.DataFrame({"balanced": [1.0, 1.0], "benchmark_a": [1.0, 1.0]}, index=index),
    )
    monkeypatch.setattr(
        "run_pipeline.write_phase1_chart_outputs",
        lambda *args, **kwargs: {"nav_chart": Path("outputs/figures/balanced_nav.png")},
    )
    monkeypatch.setattr(
        "run_pipeline.build_performance_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "annualized_return": [0.0, 0.0],
                "annualized_volatility": [0.0, 0.0],
                "max_drawdown": [0.0, 0.0],
                "ending_nav": [1.0, 1.0],
            },
            index=pd.Index(["balanced", "benchmark_a"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.build_turnover_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"total_turnover": [0.0], "total_transaction_cost_drag": [0.0]},
            index=pd.Index(["balanced"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.build_risk_matrix_outputs",
        lambda *args, **kwargs: {
            "covariance_matrix": pd.DataFrame({"VTI": [0.0]}, index=["VTI"]),
            "correlation_matrix": pd.DataFrame({"VTI": [1.0]}, index=["VTI"]),
            "correlation_pairs": pd.DataFrame([{"left": "VTI", "right": "VTI", "correlation": 1.0}]),
        },
    )
    monkeypatch.setattr(
        "run_pipeline.build_latest_rolling_metric_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"balanced": [0.0]}, index=["latest_rolling_volatility"]),
    )
    monkeypatch.setattr(
        "run_pipeline.load_risk_limits",
        lambda *args, **kwargs: {"annualized_volatility_warning": None, "max_drawdown_warning": None},
    )
    monkeypatch.setattr(
        "run_pipeline.build_portfolio_risk_limit_checks",
        lambda *args, **kwargs: pd.DataFrame(
            {"portfolio": ["balanced"], "metric": ["max_drawdown"], "breach": [False]},
            index=pd.Index(["balanced:max_drawdown"], name="rule_id"),
        ),
    )
    monkeypatch.setattr("run_pipeline.write_risk_limit_output", lambda *args, **kwargs: Path("outputs/tables/risk_limit_checks.csv"))
    monkeypatch.setattr("run_pipeline.write_risk_limit_breaches_output", lambda *args, **kwargs: Path("outputs/tables/risk_limit_breaches.csv"))
    monkeypatch.setattr(
        "run_pipeline.build_risk_limit_breach_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"total_enabled_checks": [1], "breached_checks": [0], "breach_ratio": [0.0]},
            index=pd.Index(["overall"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.write_risk_limit_breach_summary_output",
        lambda *args, **kwargs: Path("outputs/tables/risk_limit_breach_summary.csv"),
    )
    monkeypatch.setattr("run_pipeline.find_risk_limit_breaches", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("run_pipeline.validate_risk_limit_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.build_run_configuration_summary",
        lambda *args, **kwargs: pd.DataFrame({"value": ["2024-01-01"]}, index=["start"]),
    )
    monkeypatch.setattr("run_pipeline.write_run_configuration_output", lambda *args, **kwargs: Path("outputs/tables/run_configuration.csv"))
    monkeypatch.setattr(
        "run_pipeline.write_phase1_report",
        lambda *args, **kwargs: captured_markdown_kwargs.update(kwargs) or Path("outputs/reports/balanced_phase1_report.md"),
    )
    monkeypatch.setattr(
        "run_pipeline.write_phase1_html_report",
        lambda *args, **kwargs: captured_html_kwargs.update(kwargs) or Path("outputs/reports/balanced_phase1_report.html"),
    )
    monkeypatch.setattr(
        "run_pipeline.collect_table_output_paths",
        lambda *args, **kwargs: {"performance_summary": Path("outputs/tables/performance_summary.csv")},
    )
    monkeypatch.setattr("run_pipeline.build_pipeline_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr("run_pipeline.write_pipeline_manifest", lambda *args, **kwargs: Path("outputs/tables/pipeline_manifest.json"))
    monkeypatch.setattr("run_pipeline.build_output_inventory", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("run_pipeline.write_output_inventory", lambda *args, **kwargs: Path("outputs/tables/output_inventory.csv"))
    monkeypatch.setattr(
        "run_pipeline.build_pipeline_health_summary",
        lambda *args, **kwargs: pd.DataFrame({"run_passed_quality_gates": [True]}, index=pd.Index(["health"])),
    )
    monkeypatch.setattr(
        "run_pipeline.write_pipeline_health_summary",
        lambda *args, **kwargs: Path("outputs/tables/pipeline_health_summary.csv"),
    )
    monkeypatch.setattr("run_pipeline.find_missing_output_inventory_entries", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--start", "2024-01-01"])

    main()

    markdown_table = captured_markdown_kwargs["rebalance_reason_table"]
    html_table = captured_html_kwargs["rebalance_reason_table"]
    markdown_risk_limits = captured_markdown_kwargs["risk_limit_checks"]
    markdown_risk_limit_breaches = captured_markdown_kwargs["risk_limit_breaches"]
    markdown_risk_limit_breach_summary = captured_markdown_kwargs["risk_limit_breach_summary"]
    markdown_pipeline_health_summary = captured_markdown_kwargs["pipeline_health_summary"]
    html_risk_limits = captured_html_kwargs["risk_limit_checks"]
    html_risk_limit_breaches = captured_html_kwargs["risk_limit_breaches"]
    html_risk_limit_breach_summary = captured_html_kwargs["risk_limit_breach_summary"]
    html_pipeline_health_summary = captured_html_kwargs["pipeline_health_summary"]
    assert isinstance(markdown_table, pd.DataFrame)
    assert isinstance(html_table, pd.DataFrame)
    assert isinstance(markdown_risk_limits, pd.DataFrame)
    assert isinstance(markdown_risk_limit_breaches, pd.DataFrame)
    assert isinstance(markdown_risk_limit_breach_summary, pd.DataFrame)
    assert isinstance(html_risk_limits, pd.DataFrame)
    assert isinstance(html_risk_limit_breaches, pd.DataFrame)
    assert isinstance(html_risk_limit_breach_summary, pd.DataFrame)
    assert isinstance(markdown_pipeline_health_summary, pd.DataFrame)
    assert isinstance(html_pipeline_health_summary, pd.DataFrame)
    assert markdown_table.columns.tolist() == ["balanced", "benchmark_a"]
    assert markdown_table["balanced"].tolist() == ["initial", "calendar"]
    assert markdown_table["benchmark_a"].tolist() == ["none", "drift"]
    pd.testing.assert_frame_equal(markdown_table, html_table)
    pd.testing.assert_frame_equal(markdown_risk_limits, html_risk_limits)
    pd.testing.assert_frame_equal(markdown_risk_limit_breaches, html_risk_limit_breaches)
    pd.testing.assert_frame_equal(markdown_risk_limit_breach_summary, html_risk_limit_breach_summary)
    pd.testing.assert_frame_equal(markdown_pipeline_health_summary, html_pipeline_health_summary)


def test_main_fail_on_risk_limit_breach_raises_runtime_error(monkeypatch) -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    asset_returns = pd.DataFrame({"VTI": [0.0, 0.0]}, index=index)
    clean_frame = pd.DataFrame(
        {
            "adj_close": [100.0, 101.0],
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_000_000],
            "dollar_volume": [100_000_000.0, 101_000_000.0],
        },
        index=index,
    )
    clean_frame.index.name = "date"

    strategy_result = {
        "portfolio_returns": pd.Series([0.0, 0.0], index=index),
        "portfolio_nav": pd.Series([1.0, 1.0], index=index),
        "turnover": pd.Series([0.0, 0.0], index=index),
        "transaction_costs": pd.Series([0.0, 0.0], index=index),
        "summary": pd.Series({"annualized_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0}),
        "annual_return_table": pd.DataFrame({"portfolio": [0.0]}, index=pd.Index([2024], name="year")),
        "benchmark_comparisons": pd.DataFrame(),
        "benchmark_annual_excess_returns": pd.DataFrame(),
        "benchmark_drawdown_comparisons": pd.DataFrame(),
    }

    monkeypatch.setattr("run_pipeline.configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.validate_date_range", lambda start, end: (start, end))
    monkeypatch.setattr("run_pipeline.load_enabled_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.fetch_prices", lambda *args, **kwargs: {"VTI": clean_frame})
    monkeypatch.setattr("run_pipeline.batch_clean_price_frames", lambda frames: {"VTI": clean_frame})
    monkeypatch.setattr("run_pipeline.save_processed_frames", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "run_pipeline.write_data_quality_outputs",
        lambda *args, **kwargs: pd.DataFrame({"observations": [2]}, index=pd.Index(["VTI"], name="ticker")),
    )
    monkeypatch.setattr(
        "run_pipeline.fetch_macro_series",
        lambda *args, **kwargs: pd.DataFrame({"vix": [15.0]}, index=index[:1]),
    )
    monkeypatch.setattr("run_pipeline.save_macro_series_per_symbol", lambda *args, **kwargs: {})
    monkeypatch.setattr("run_pipeline.write_macro_outputs", lambda *args, **kwargs: Path("outputs/tables/macro_observation_summary.csv"))
    monkeypatch.setattr(
        "run_pipeline.filter_liquid_universe",
        lambda *args, **kwargs: (
            ["VTI"],
            pd.DataFrame({"passes_liquidity_filter": [True]}, index=pd.Index(["VTI"], name="ticker")),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.score_etf_universe",
        lambda *args, **kwargs: pd.DataFrame({"phase1_total_score": [40.0]}, index=pd.Index(["VTI"], name="ticker")),
    )
    monkeypatch.setattr("run_pipeline.write_liquidity_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.collect_required_backtest_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.warn_on_non_liquid_required_assets", lambda *args, **kwargs: [])
    monkeypatch.setattr("run_pipeline.resolve_backtest_tickers", lambda *args, **kwargs: ["VTI"])
    monkeypatch.setattr("run_pipeline.build_asset_return_matrix", lambda *args, **kwargs: asset_returns)
    monkeypatch.setattr("run_pipeline.build_adjusted_close_matrix", lambda *args, **kwargs: pd.DataFrame({"VTI": [100.0, 101.0]}, index=index))
    monkeypatch.setattr(
        "run_pipeline.build_trend_filter_overlay_settings",
        lambda *args, **kwargs: {"enabled": False, "moving_average_days": 210, "reduction_fraction": 0.5, "assets": []},
    )
    monkeypatch.setattr("run_pipeline.run_strategy_backtests", lambda *args, **kwargs: ("balanced", strategy_result, {}))
    monkeypatch.setattr("run_pipeline.build_backtest_policy_tables", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr("run_pipeline.write_backtest_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.build_return_table", lambda *args, **kwargs: pd.DataFrame({"balanced": [0.0, 0.0]}, index=index))
    monkeypatch.setattr(
        "run_pipeline.build_rolling_metric_outputs",
        lambda *args, **kwargs: {
            "rolling_volatility": pd.DataFrame({"balanced": [0.0, 0.0]}, index=index),
            "rolling_sharpe": pd.DataFrame({"balanced": [0.0, 0.0]}, index=index),
            "drawdown_series": pd.DataFrame({"balanced": [0.0, 0.0]}, index=index),
        },
    )
    monkeypatch.setattr("run_pipeline.write_rolling_metric_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.build_nav_table", lambda *args, **kwargs: pd.DataFrame({"balanced": [1.0, 1.0]}, index=index))
    monkeypatch.setattr("run_pipeline.write_phase1_chart_outputs", lambda *args, **kwargs: {"nav_chart": Path("outputs/figures/balanced_nav.png")})
    monkeypatch.setattr(
        "run_pipeline.build_performance_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"annualized_return": [0.0], "annualized_volatility": [0.3], "max_drawdown": [-0.4], "ending_nav": [1.0]},
            index=pd.Index(["balanced"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.build_turnover_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"total_turnover": [0.0], "total_transaction_cost_drag": [0.0]},
            index=pd.Index(["balanced"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.build_risk_matrix_outputs",
        lambda *args, **kwargs: {
            "covariance_matrix": pd.DataFrame({"VTI": [0.0]}, index=["VTI"]),
            "correlation_matrix": pd.DataFrame({"VTI": [1.0]}, index=["VTI"]),
            "correlation_pairs": pd.DataFrame([{"left": "VTI", "right": "VTI", "correlation": 1.0}]),
        },
    )
    monkeypatch.setattr(
        "run_pipeline.build_latest_rolling_metric_snapshot",
        lambda *args, **kwargs: pd.DataFrame({"balanced": [0.0]}, index=["latest_rolling_volatility"]),
    )
    monkeypatch.setattr(
        "run_pipeline.load_risk_limits",
        lambda *args, **kwargs: {"annualized_volatility_warning": 0.2, "max_drawdown_warning": 0.25},
    )
    monkeypatch.setattr(
        "run_pipeline.build_portfolio_risk_limit_checks",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "portfolio": ["balanced"],
                "metric": ["max_drawdown"],
                "threshold": [0.25],
                "observed": [-0.40],
                "comparison_value": [0.40],
                "limit_enabled": [True],
                "breach": [True],
            },
            index=pd.Index(["balanced:max_drawdown"], name="rule_id"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.find_risk_limit_breaches",
        lambda checks: checks.loc[checks["breach"]],
    )
    monkeypatch.setattr("run_pipeline.validate_risk_limit_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_pipeline.write_risk_limit_output", lambda *args, **kwargs: Path("outputs/tables/risk_limit_checks.csv"))
    monkeypatch.setattr("run_pipeline.write_risk_limit_breaches_output", lambda *args, **kwargs: Path("outputs/tables/risk_limit_breaches.csv"))
    monkeypatch.setattr(
        "run_pipeline.build_risk_limit_breach_summary",
        lambda *args, **kwargs: pd.DataFrame(
            {"total_enabled_checks": [1], "breached_checks": [1], "breach_ratio": [1.0]},
            index=pd.Index(["overall"], name="portfolio"),
        ),
    )
    monkeypatch.setattr(
        "run_pipeline.write_risk_limit_breach_summary_output",
        lambda *args, **kwargs: Path("outputs/tables/risk_limit_breach_summary.csv"),
    )
    monkeypatch.setattr(
        "run_pipeline.build_run_configuration_summary",
        lambda *args, **kwargs: pd.DataFrame({"value": ["2024-01-01"]}, index=["start"]),
    )
    monkeypatch.setattr("run_pipeline.write_run_configuration_output", lambda *args, **kwargs: Path("outputs/tables/run_configuration.csv"))
    monkeypatch.setattr("run_pipeline.write_phase1_report", lambda *args, **kwargs: Path("outputs/reports/balanced_phase1_report.md"))
    monkeypatch.setattr("run_pipeline.write_phase1_html_report", lambda *args, **kwargs: Path("outputs/reports/balanced_phase1_report.html"))
    monkeypatch.setattr("run_pipeline.collect_table_output_paths", lambda *args, **kwargs: {"performance_summary": Path("outputs/tables/performance_summary.csv")})
    monkeypatch.setattr("run_pipeline.build_pipeline_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr("run_pipeline.write_pipeline_manifest", lambda *args, **kwargs: Path("outputs/tables/pipeline_manifest.json"))
    monkeypatch.setattr("run_pipeline.build_output_inventory", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("run_pipeline.write_output_inventory", lambda *args, **kwargs: Path("outputs/tables/output_inventory.csv"))
    monkeypatch.setattr(
        "run_pipeline.build_pipeline_health_summary",
        lambda *args, **kwargs: pd.DataFrame({"run_passed_quality_gates": [False]}, index=pd.Index(["health"])),
    )
    monkeypatch.setattr(
        "run_pipeline.write_pipeline_health_summary",
        lambda *args, **kwargs: Path("outputs/tables/pipeline_health_summary.csv"),
    )
    monkeypatch.setattr("run_pipeline.find_missing_output_inventory_entries", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("run_pipeline.find_empty_output_inventory_entries", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "--start", "2024-01-01", "--fail-on-risk-limit-breach"],
    )

    try:
        main()
    except RuntimeError as exc:
        assert "Risk-limit validation failed" in str(exc)
        assert "balanced:max_drawdown" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when --fail-on-risk-limit-breach is set and breaches exist.")
