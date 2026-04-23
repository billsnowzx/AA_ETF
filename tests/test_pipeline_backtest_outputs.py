import shutil
import uuid
from pathlib import Path

import json
import math
import pandas as pd

from run_pipeline import (
    build_asset_return_matrix,
    build_nav_table,
    build_rebalance_reason_table,
    build_trend_filter_summary,
    build_trend_filter_overlay_settings,
    build_backtest_policy_tables,
    find_missing_output_inventory_entries,
    find_empty_output_inventory_entries,
    build_output_inventory,
    build_performance_summary,
    build_pipeline_manifest,
    build_risk_matrix_outputs,
    build_rolling_metric_outputs,
    build_return_table,
    build_turnover_summary,
    collect_table_output_paths,
    collect_required_backtest_tickers,
    resolve_backtest_tickers,
    run_strategy_backtests,
    write_backtest_outputs,
    write_output_inventory,
    write_pipeline_manifest,
    write_run_configuration_output,
    write_rolling_metric_outputs,
)
from src.dashboard.reporting import build_rebalance_reason_summary


def _build_clean_frame(prices: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    frame = pd.DataFrame(
        {
            "adj_close": prices,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1_000_000] * len(prices),
            "dollar_volume": [price * 1_000_000 for price in prices],
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


def test_build_asset_return_matrix_from_clean_frames() -> None:
    clean_frames = {
        "VTI": _build_clean_frame([100.0, 110.0, 121.0]),
        "AGG": _build_clean_frame([50.0, 51.0, 52.02]),
    }

    asset_returns = build_asset_return_matrix(clean_frames, tickers=["VTI", "AGG"])

    assert list(asset_returns.columns) == ["VTI", "AGG"]
    assert len(asset_returns) == 2
    assert math.isclose(asset_returns.iloc[0]["VTI"], 0.10, rel_tol=1e-9)


def test_run_strategy_backtests_uses_repo_configs() -> None:
    clean_frames = {
        "VTI": _build_clean_frame([100.0, 101.0, 102.0]),
        "VEA": _build_clean_frame([100.0, 100.5, 101.0]),
        "IEMG": _build_clean_frame([100.0, 100.2, 100.4]),
        "AGG": _build_clean_frame([100.0, 100.1, 100.2]),
        "BNDX": _build_clean_frame([100.0, 100.05, 100.10]),
        "GLD": _build_clean_frame([100.0, 100.3, 100.6]),
        "VNQ": _build_clean_frame([100.0, 100.4, 100.8]),
    }
    asset_returns = build_asset_return_matrix(clean_frames)

    strategy_name, strategy_result, benchmark_results = run_strategy_backtests(
        asset_returns,
        portfolio_template_config="config/portfolio_templates.yaml",
        benchmark_config="config/benchmark_config.yaml",
        rebalance_config="config/rebalance_rules.yaml",
    )

    assert strategy_name == "balanced"
    assert "benchmark_a" in benchmark_results
    assert "benchmark_b" in benchmark_results
    assert "benchmark_a" in strategy_result["annual_return_table"].columns


def test_collect_required_backtest_tickers_includes_strategy_and_benchmarks() -> None:
    tickers = collect_required_backtest_tickers(
        "config/portfolio_templates.yaml",
        "config/benchmark_config.yaml",
    )

    assert tickers == ["VTI", "VEA", "IEMG", "AGG", "BNDX", "GLD", "VNQ"]


def test_build_trend_filter_overlay_settings_selects_equity_like_assets() -> None:
    settings = build_trend_filter_overlay_settings(
        rebalance_config="config/rebalance_rules.yaml",
        universe_config="config/etf_universe.yaml",
        backtest_tickers=["VTI", "VEA", "IEMG", "AGG", "GLD", "VNQ"],
    )

    assert settings["enabled"] is False
    assert settings["moving_average_days"] == 210
    assert settings["assets"] == []


def test_build_trend_filter_overlay_settings_enabled_uses_equity_and_reit_assets() -> None:
    rebalance_config_path = Path("data/cache") / f"test_rebalance_trend_{uuid.uuid4().hex}.yaml"
    rebalance_config_path.parent.mkdir(parents=True, exist_ok=True)
    rebalance_config_path.write_text(
        "\n".join(
            [
                "standard_rebalance:",
                "  frequency: quarterly",
                "weight_drift_rule:",
                "  enabled: true",
                "  relative_deviation_threshold: 0.20",
                "trend_filter:",
                "  enabled: true",
                "  moving_average_months: 6",
                "  reduction_fraction: 0.30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        settings = build_trend_filter_overlay_settings(
            rebalance_config=rebalance_config_path,
            universe_config="config/etf_universe.yaml",
            backtest_tickers=["VTI", "VEA", "IEMG", "AGG", "GLD", "VNQ"],
        )
        assert settings["enabled"] is True
        assert settings["moving_average_days"] == 126
        assert settings["reduction_fraction"] == 0.30
        assert settings["assets"] == ["VTI", "VEA", "IEMG", "VNQ"]
    finally:
        rebalance_config_path.unlink(missing_ok=True)


def test_build_trend_filter_summary_counts_active_days_and_reduced_assets() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    strategy_result = {
        "trend_filter_active": pd.Series([False, True, True], index=index, name="trend_filter_active"),
        "trend_filter_scales": pd.DataFrame(
            {
                "VTI": [1.0, 0.5, 0.5],
                "AGG": [1.0, 1.0, 1.0],
            },
            index=index,
        ),
    }

    summary = build_trend_filter_summary("balanced", strategy_result)

    assert int(summary.loc["balanced", "observations"]) == 3
    assert int(summary.loc["balanced", "trend_active_days"]) == 2
    assert math.isclose(float(summary.loc["balanced", "trend_active_ratio"]), 2.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(float(summary.loc["balanced", "avg_reduced_assets"]), 2.0 / 3.0, rel_tol=1e-9)
    assert int(summary.loc["balanced", "max_reduced_assets"]) == 1


def test_resolve_backtest_tickers_configured_mode_keeps_required_assets() -> None:
    tickers = resolve_backtest_tickers(
        required_tickers=["VTI", "IAGG"],
        liquid_tickers=["VTI"],
        mode="configured",
    )

    assert tickers == ["VTI", "IAGG"]


def test_resolve_backtest_tickers_liquidity_filtered_mode_requires_all_assets_to_pass() -> None:
    try:
        resolve_backtest_tickers(
            required_tickers=["VTI", "IAGG"],
            liquid_tickers=["VTI"],
            mode="liquidity_filtered",
        )
    except ValueError as exc:
        assert "failed the liquidity screen" in str(exc)
    else:
        raise AssertionError("Expected ValueError when a required asset fails the liquidity screen.")


def test_build_rebalance_reason_table_uses_fallback_none_when_reasons_missing() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    strategy_result = {
        "portfolio_returns": pd.Series([0.01, -0.01, 0.0], index=index),
    }
    benchmark_results = {
        "benchmark_a": {
            "portfolio_nav": pd.Series([1.0, 1.01, 1.01], index=index),
        }
    }

    table = build_rebalance_reason_table(
        strategy_name="balanced",
        strategy_result=strategy_result,
        benchmark_results=benchmark_results,
    )

    assert table.index.name == "date"
    assert table.columns.tolist() == ["balanced", "benchmark_a"]
    assert table["balanced"].tolist() == ["none", "none", "none"]
    assert table["benchmark_a"].tolist() == ["none", "none", "none"]


def test_build_rebalance_reason_table_prefers_explicit_rebalance_reason_series() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    strategy_result = {
        "rebalance_reasons": pd.Series(["initial", "calendar"], index=index),
        "portfolio_returns": pd.Series([0.0, 0.0], index=index),
    }
    benchmark_results = {
        "benchmark_a": {
            "rebalance_reasons": pd.Series(["none", "drift"], index=index),
            "portfolio_returns": pd.Series([0.0, 0.0], index=index),
        }
    }

    table = build_rebalance_reason_table(
        strategy_name="balanced",
        strategy_result=strategy_result,
        benchmark_results=benchmark_results,
    )

    assert table["balanced"].tolist() == ["initial", "calendar"]
    assert table["benchmark_a"].tolist() == ["none", "drift"]


def test_build_summary_tables_and_write_outputs() -> None:
    clean_frames = {
        "VTI": _build_clean_frame([100.0, 101.0, 102.0]),
        "VEA": _build_clean_frame([100.0, 100.5, 101.0]),
        "IEMG": _build_clean_frame([100.0, 100.2, 100.4]),
        "AGG": _build_clean_frame([100.0, 100.1, 100.2]),
        "BNDX": _build_clean_frame([100.0, 100.05, 100.10]),
        "GLD": _build_clean_frame([100.0, 100.3, 100.6]),
        "VNQ": _build_clean_frame([100.0, 100.4, 100.8]),
    }
    asset_returns = build_asset_return_matrix(clean_frames)
    strategy_name, strategy_result, benchmark_results = run_strategy_backtests(
        asset_returns,
        portfolio_template_config="config/portfolio_templates.yaml",
        benchmark_config="config/benchmark_config.yaml",
        rebalance_config="config/rebalance_rules.yaml",
    )
    policy_validation, policy_summary = build_backtest_policy_tables(
        strategy_name,
        "config/portfolio_templates.yaml",
        "config/benchmark_config.yaml",
        liquid_tickers=["VTI", "VEA", "IEMG", "AGG", "IAGG", "GLD", "VNQ"],
    )

    performance_summary = build_performance_summary(strategy_name, strategy_result, benchmark_results)
    turnover_summary = build_turnover_summary(strategy_name, strategy_result, benchmark_results)
    nav_table = build_nav_table(strategy_name, strategy_result, benchmark_results)
    return_table = build_return_table(strategy_name, strategy_result, benchmark_results)
    risk_outputs = build_risk_matrix_outputs(asset_returns)
    rolling_outputs = build_rolling_metric_outputs(return_table, window=2, periods_per_year=2)

    output_dir = Path("data/cache") / f"test_pipeline_outputs_{uuid.uuid4().hex}"
    try:
        write_backtest_outputs(
            strategy_name,
            strategy_result,
            benchmark_results,
            policy_validation,
            policy_summary,
            asset_returns,
            output_dir,
        )
        assert strategy_name in performance_summary.index
        assert "benchmark_a" in turnover_summary.index
        assert strategy_name in nav_table.columns
        assert "benchmark_b" in return_table.columns
        assert "covariance_matrix" in risk_outputs
        assert "correlation_pairs" in risk_outputs
        assert "top_correlation_pairs" in risk_outputs
        assert "asset_risk_snapshot" in risk_outputs
        assert "rolling_volatility" in rolling_outputs
        assert "rolling_sharpe" in rolling_outputs
        write_rolling_metric_outputs(rolling_outputs, output_dir)
        assert (output_dir / "performance_summary.csv").exists()
        assert (output_dir / "annual_return_table.csv").exists()
        assert (output_dir / "benchmark_annual_excess_returns.csv").exists()
        assert (output_dir / "benchmark_drawdown_comparisons.csv").exists()
        assert (output_dir / "nav_series.csv").exists()
        assert (output_dir / "backtest_universe_validation.csv").exists()
        assert (output_dir / "rebalance_reason.csv").exists()
        assert (output_dir / "rebalance_reason_summary.csv").exists()
        assert (output_dir / "covariance_matrix.csv").exists()
        assert (output_dir / "correlation_pairs.csv").exists()
        assert (output_dir / "top_correlation_pairs.csv").exists()
        assert (output_dir / "asset_risk_snapshot.csv").exists()
        assert (output_dir / "trend_filter_summary.csv").exists()
        assert (output_dir / "trend_filter_active.csv").exists()
        assert (output_dir / "trend_filter_scales.csv").exists()
        assert (output_dir / "rolling_volatility.csv").exists()
        assert (output_dir / "rolling_sharpe.csv").exists()
        assert (output_dir / "drawdown_series.csv").exists()

        rebalance_reason = pd.read_csv(output_dir / "rebalance_reason.csv", index_col=0)
        rebalance_reason_summary = pd.read_csv(output_dir / "rebalance_reason_summary.csv", index_col=0)
        expected_summary = build_rebalance_reason_summary(rebalance_reason)
        pd.testing.assert_frame_equal(rebalance_reason_summary, expected_summary)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_and_write_pipeline_manifest_records_run_context() -> None:
    performance_summary = pd.DataFrame(
        {
            "ending_nav": [1.25],
            "annualized_return": [0.10],
            "annualized_volatility": [0.12],
            "max_drawdown": [-0.20],
        },
        index=pd.Index(["balanced"], name="portfolio"),
    )
    manifest = build_pipeline_manifest(
        start="2024-01-01",
        end="2024-12-31",
        enabled_tickers=["VTI", "AGG"],
        liquid_tickers=["VTI", "AGG"],
        backtest_tickers=["VTI", "AGG"],
        strategy_name="balanced",
        template_name="balanced",
        backtest_universe_mode="liquidity_filtered",
        rolling_window=21,
        performance_summary=performance_summary,
        table_paths={
            "performance_summary": Path("outputs/tables/performance_summary.csv"),
            "output_inventory": Path("outputs/tables/output_inventory.csv"),
        },
        report_paths=[Path("outputs/reports/balanced_phase1_report.md")],
        chart_paths={"nav_chart": Path("outputs/figures/balanced_nav.png")},
        config_paths={
            "universe": "config/etf_universe.yaml",
            "portfolio_templates": "config/portfolio_templates.yaml",
            "benchmarks": "config/benchmark_config.yaml",
            "rebalance_rules": "config/rebalance_rules.yaml",
            "risk_limits": "config/risk_limits.yaml",
        },
        output_dir="outputs/tables",
        raw_dir="data/raw",
        processed_dir="data/processed",
        figure_dir="outputs/figures",
        report_dir="outputs/reports",
        run_completed_at="2026-04-20T00:00:00+00:00",
        as_of_date="2024-12-31",
        seed=7,
    )

    output_dir = Path("data/cache") / f"test_manifest_{uuid.uuid4().hex}"
    try:
        manifest_path = write_pipeline_manifest(manifest, output_dir)
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert loaded["parameters"]["rolling_window"] == 21
        assert loaded["parameters"]["template_name"] == "balanced"
        assert loaded["parameters"]["backtest_universe_mode"] == "liquidity_filtered"
        assert loaded["parameters"]["as_of_date"] == "2024-12-31"
        assert loaded["parameters"]["seed"] == 7
        assert loaded["config_files"]["universe"] == "config\\etf_universe.yaml"
        assert loaded["config_files"]["risk_limits"] == "config\\risk_limits.yaml"
        assert loaded["universes"]["backtest_tickers"] == ["VTI", "AGG"]
        assert loaded["strategy"]["ending_nav"] == 1.25
        assert loaded["outputs"]["tables"]["performance_summary"] == "outputs\\tables\\performance_summary.csv"
        assert loaded["outputs"]["tables"]["output_inventory"] == "outputs\\tables\\output_inventory.csv"
        assert loaded["outputs"]["figures"]["nav_chart"] == "outputs\\figures\\balanced_nav.png"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_collect_table_output_paths_returns_csv_outputs_by_stem() -> None:
    output_dir = Path("data/cache") / f"test_table_paths_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        (output_dir / "performance_summary.csv").write_text("x\n1\n", encoding="utf-8")
        (output_dir / "pipeline_manifest.json").write_text("{}", encoding="utf-8")

        paths = collect_table_output_paths(output_dir)

        assert paths == {"performance_summary": output_dir / "performance_summary.csv"}
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_write_run_configuration_output_creates_auditable_csv() -> None:
    output_dir = Path("data/cache") / f"test_run_config_{uuid.uuid4().hex}"
    run_configuration = pd.DataFrame.from_dict(
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "config_universe": "config\\etf_universe.yaml",
        },
        orient="index",
        columns=["value"],
    )

    try:
        output_path = write_run_configuration_output(run_configuration, output_dir)
        loaded = pd.read_csv(output_path, index_col=0)
        table_paths = collect_table_output_paths(output_dir)

        assert output_path == output_dir / "run_configuration.csv"
        assert loaded.loc["start", "value"] == "2024-01-01"
        assert loaded.loc["config_universe", "value"] == "config\\etf_universe.yaml"
        assert table_paths["run_configuration"] == output_path
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_and_write_output_inventory_records_existing_outputs() -> None:
    output_dir = Path("data/cache") / f"test_output_inventory_{uuid.uuid4().hex}"
    table_path = output_dir / "performance_summary.csv"
    report_path = output_dir / "balanced_phase1_report.md"
    figure_path = output_dir / "balanced_nav.png"
    manifest_path = output_dir / "pipeline_manifest.json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path.write_text("portfolio,ending_nav\nbalanced,1.25\n", encoding="utf-8")
        report_path.write_text("# report\n", encoding="utf-8")
        figure_path.write_bytes(b"png")
        manifest_path.write_text("{}\n", encoding="utf-8")

        inventory = build_output_inventory(
            table_paths={"performance_summary": table_path},
            report_paths=[report_path],
            chart_paths={"nav_chart": figure_path},
            manifest_path=manifest_path,
        )
        output_path = write_output_inventory(inventory, output_dir)
        loaded = pd.read_csv(output_path)

        table_row = loaded.loc[loaded["name"] == "performance_summary"].iloc[0]
        report_row = loaded.loc[loaded["name"] == "balanced_phase1_report"].iloc[0]
        figure_row = loaded.loc[loaded["name"] == "nav_chart"].iloc[0]
        manifest_row = loaded.loc[loaded["name"] == "pipeline_manifest"].iloc[0]

        assert output_path == output_dir / "output_inventory.csv"
        assert bool(table_row["exists"]) is True
        assert table_row["size_bytes"] == table_path.stat().st_size
        assert report_row["output_type"] == "report"
        assert figure_row["output_type"] == "figure"
        assert manifest_row["output_type"] == "manifest"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_output_inventory_marks_missing_artifacts() -> None:
    output_dir = Path("data/cache") / f"test_output_inventory_missing_{uuid.uuid4().hex}"
    existing_table_path = output_dir / "performance_summary.csv"
    missing_report_path = output_dir / "balanced_phase1_report.md"
    missing_figure_path = output_dir / "balanced_nav.png"
    missing_manifest_path = output_dir / "pipeline_manifest.json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_table_path.write_text("portfolio,ending_nav\nbalanced,1.25\n", encoding="utf-8")

        inventory = build_output_inventory(
            table_paths={"performance_summary": existing_table_path},
            report_paths=[missing_report_path],
            chart_paths={"nav_chart": missing_figure_path},
            manifest_path=missing_manifest_path,
        )

        report_row = inventory.loc[inventory["name"] == "balanced_phase1_report"].iloc[0]
        figure_row = inventory.loc[inventory["name"] == "nav_chart"].iloc[0]
        manifest_row = inventory.loc[inventory["name"] == "pipeline_manifest"].iloc[0]

        assert bool(report_row["exists"]) is False
        assert report_row["size_bytes"] == 0
        assert bool(figure_row["exists"]) is False
        assert figure_row["size_bytes"] == 0
        assert bool(manifest_row["exists"]) is False
        assert manifest_row["size_bytes"] == 0
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_find_missing_output_inventory_entries_returns_only_missing_rows() -> None:
    inventory = pd.DataFrame(
        [
            {"output_type": "table", "name": "performance_summary", "path": "a.csv", "exists": True, "size_bytes": 10},
            {"output_type": "report", "name": "balanced_phase1_report", "path": "b.md", "exists": False, "size_bytes": 0},
            {"output_type": "figure", "name": "nav_chart", "path": "c.png", "exists": False, "size_bytes": 0},
        ]
    )

    missing = find_missing_output_inventory_entries(inventory)

    assert len(missing) == 2
    assert missing["name"].tolist() == ["balanced_phase1_report", "nav_chart"]
    assert missing["size_bytes"].tolist() == [0, 0]


def test_find_empty_output_inventory_entries_returns_only_existing_empty_rows() -> None:
    inventory = pd.DataFrame(
        [
            {"output_type": "table", "name": "performance_summary", "path": "a.csv", "exists": True, "size_bytes": 10},
            {"output_type": "report", "name": "balanced_phase1_report", "path": "b.md", "exists": True, "size_bytes": 0},
            {"output_type": "figure", "name": "nav_chart", "path": "c.png", "exists": False, "size_bytes": 0},
        ]
    )

    empty = find_empty_output_inventory_entries(inventory)

    assert len(empty) == 1
    assert empty["name"].tolist() == ["balanced_phase1_report"]
    assert empty["size_bytes"].tolist() == [0]
