"""Phase 1 report assembly helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _format_percent(value: float | int | object) -> str:
    """Format a decimal value as a percent string when numeric."""
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _format_decimal(value: float | int | object, digits: int = 4) -> str:
    """Format a numeric value as a fixed-point decimal string when numeric."""
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a simple Markdown table."""
    table = frame.copy()
    headers = ["index", *table.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for index, row in table.iterrows():
        values = [str(index), *[str(value) for value in row.tolist()]]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a compact HTML table."""
    table = frame.copy()
    header_cells = "".join(f"<th>{escape(str(column))}</th>" for column in ["index", *table.columns.tolist()])
    body_rows = []

    for index, row in table.iterrows():
        value_cells = "".join(f"<td>{escape(str(value))}</td>" for value in [index, *row.tolist()])
        body_rows.append(f"<tr>{value_cells}</tr>")

    body = "".join(body_rows)
    return (
        "<table>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def build_top_correlation_summary(
    correlation_pairs: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a small summary of the strongest non-diagonal correlations."""
    if correlation_pairs.empty:
        return pd.DataFrame(columns=["pair", "correlation"])

    working = correlation_pairs.copy()
    if {"left", "right", "correlation"} - set(working.columns):
        raise ValueError("Correlation pair table must contain 'left', 'right', and 'correlation' columns.")

    working = working.loc[working["left"] != working["right"]].copy()
    if working.empty:
        return pd.DataFrame(columns=["pair", "correlation"])

    working["abs_correlation"] = working["correlation"].abs()
    working["pair"] = working["left"] + " vs " + working["right"]
    working = working.sort_values(["abs_correlation", "pair"], ascending=[False, True])
    summary = working[["pair", "correlation"]].head(top_n).copy()
    summary["correlation"] = summary["correlation"].map(_format_decimal)
    return summary


def build_asset_risk_snapshot(
    correlation_matrix: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-asset snapshot from correlation and covariance matrices."""
    if correlation_matrix.empty or covariance_matrix.empty:
        return pd.DataFrame(columns=["avg_correlation", "variance"])

    avg_correlation = {}
    for asset in correlation_matrix.columns:
        peer_values = correlation_matrix.loc[asset].drop(labels=[asset], errors="ignore")
        avg_correlation[asset] = float(peer_values.mean()) if not peer_values.empty else float("nan")

    variance = pd.Series(
        {asset: float(covariance_matrix.loc[asset, asset]) for asset in covariance_matrix.columns},
        name="variance",
    )
    snapshot = pd.DataFrame({"avg_correlation": pd.Series(avg_correlation), "variance": variance})
    snapshot["avg_correlation"] = snapshot["avg_correlation"].map(_format_decimal)
    snapshot["variance"] = snapshot["variance"].map(lambda value: _format_decimal(value, digits=6))
    return snapshot


def build_phase1_risk_summary_tables(
    correlation_pairs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build report-facing risk summary tables in reusable tabular form."""
    return {
        "top_correlation_pairs": build_top_correlation_summary(correlation_pairs),
        "asset_risk_snapshot": build_asset_risk_snapshot(correlation_matrix, covariance_matrix),
    }


def build_latest_rolling_metric_snapshot(
    rolling_volatility: pd.DataFrame,
    rolling_sharpe: pd.DataFrame,
) -> pd.DataFrame:
    """Build a latest-value snapshot from rolling strategy and benchmark metrics."""
    rows: dict[str, pd.Series] = {}

    if not rolling_volatility.empty:
        latest_volatility = rolling_volatility.dropna(how="all").tail(1)
        if not latest_volatility.empty:
            rows["latest_rolling_volatility"] = latest_volatility.iloc[0]

    if not rolling_sharpe.empty:
        latest_sharpe = rolling_sharpe.dropna(how="all").tail(1)
        if not latest_sharpe.empty:
            rows["latest_rolling_sharpe"] = latest_sharpe.iloc[0]

    return pd.DataFrame(rows).T


def _format_rolling_metric_snapshot(snapshot: pd.DataFrame | None) -> pd.DataFrame:
    """Format latest rolling metrics for report presentation."""
    if snapshot is None or snapshot.empty:
        return pd.DataFrame()

    formatted = snapshot.astype(object).copy()
    for index in formatted.index:
        formatter = _format_percent if "volatility" in str(index) else _format_decimal
        formatted.loc[index] = formatted.loc[index].map(formatter)
    return formatted


def _format_rolling_correlation(rolling_correlation: pd.DataFrame | None) -> pd.DataFrame:
    """Format rolling correlation values for report presentation."""
    if rolling_correlation is None or rolling_correlation.empty:
        return pd.DataFrame()
    formatted = rolling_correlation.astype(object).copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(_format_decimal)
    return formatted


def _format_data_quality_summary(data_quality_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format data-quality diagnostics for report presentation."""
    if data_quality_summary is None or data_quality_summary.empty:
        return pd.DataFrame()

    columns = [
        "start_date",
        "end_date",
        "observations",
        "missing_adj_close",
        "missing_volume",
        "zero_volume",
        "missing_dollar_volume",
        "has_duplicate_dates",
    ]
    available_columns = [column for column in columns if column in data_quality_summary.columns]
    formatted = data_quality_summary[available_columns].astype(object).copy()
    for column in [
        "observations",
        "missing_adj_close",
        "missing_volume",
        "zero_volume",
        "missing_dollar_volume",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: _format_decimal(value, digits=0))
    return formatted


def build_rebalance_reason_summary(rebalance_reason_table: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize rebalance trigger reasons by portfolio for audit reporting."""
    if rebalance_reason_table is None or rebalance_reason_table.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for portfolio in rebalance_reason_table.columns:
        reasons = (
            rebalance_reason_table[portfolio]
            .fillna("none")
            .astype(str)
            .str.strip()
            .replace("", "none")
        )
        total_days = int(len(reasons))
        non_none = reasons[reasons != "none"]
        rebalance_days = int(non_none.shape[0])
        calendar_days = int(non_none.str.contains("calendar").sum())
        drift_days = int(non_none.str.contains("drift").sum())
        both_days = int(non_none.str.contains("calendar").mul(non_none.str.contains("drift")).sum())
        if non_none.empty:
            top_reason = "none"
        else:
            top_reason = str(non_none.value_counts().index[0])

        rows.append(
            {
                "portfolio": portfolio,
                "total_days": total_days,
                "rebalance_days": rebalance_days,
                "rebalance_ratio": float(rebalance_days / total_days) if total_days > 0 else 0.0,
                "calendar_days": calendar_days,
                "drift_days": drift_days,
                "calendar_and_drift_days": both_days,
                "top_reason": top_reason,
            }
        )

    return pd.DataFrame(rows).set_index("portfolio")


def _format_rebalance_reason_summary(rebalance_reason_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format rebalance summary diagnostics for report presentation."""
    if rebalance_reason_summary is None or rebalance_reason_summary.empty:
        return pd.DataFrame()

    formatted = rebalance_reason_summary.astype(object).copy()
    for column in [
        "total_days",
        "rebalance_days",
        "calendar_days",
        "drift_days",
        "calendar_and_drift_days",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: _format_decimal(value, digits=0))
    if "rebalance_ratio" in formatted.columns:
        formatted["rebalance_ratio"] = formatted["rebalance_ratio"].map(_format_percent)
    return formatted


def _build_recent_rebalance_events(rebalance_reason_table: pd.DataFrame | None, limit: int = 20) -> pd.DataFrame:
    """Build a recent non-empty rebalance event table for reporting."""
    if rebalance_reason_table is None or rebalance_reason_table.empty:
        return pd.DataFrame()

    events = rebalance_reason_table.copy()
    if isinstance(events.index, pd.DatetimeIndex):
        events.index = events.index.strftime("%Y-%m-%d")
    events = events.fillna("none").replace("", "none")
    non_empty_rows = events.apply(lambda row: (row != "none").any(), axis=1)
    recent = events.loc[non_empty_rows]
    if recent.empty:
        return pd.DataFrame()
    return recent.tail(limit)


def _format_risk_limit_checks(risk_limit_checks: pd.DataFrame | None) -> pd.DataFrame:
    """Format risk-limit checks for report presentation."""
    if risk_limit_checks is None or risk_limit_checks.empty:
        return pd.DataFrame()

    formatted = risk_limit_checks.astype(object).copy()
    for column in ["threshold", "observed", "comparison_value"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_percent)
    return formatted


def _format_risk_limit_breaches(risk_limit_breaches: pd.DataFrame | None) -> pd.DataFrame:
    """Format risk-limit breach rows for report presentation."""
    return _format_risk_limit_checks(risk_limit_breaches)


def _format_risk_limit_breach_summary(risk_limit_breach_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format risk-limit breach summary rows for report presentation."""
    if risk_limit_breach_summary is None or risk_limit_breach_summary.empty:
        return pd.DataFrame()
    formatted = risk_limit_breach_summary.astype(object).copy()
    for column in ["total_enabled_checks", "breached_checks"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: _format_decimal(value, digits=0))
    if "breach_ratio" in formatted.columns:
        formatted["breach_ratio"] = formatted["breach_ratio"].map(_format_percent)
    return formatted


def _format_pipeline_health_summary(pipeline_health_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format pipeline-health summary rows for report presentation."""
    if pipeline_health_summary is None or pipeline_health_summary.empty:
        return pd.DataFrame()
    formatted = pipeline_health_summary.astype(object).copy()
    for column in [
        "missing_outputs",
        "empty_outputs",
        "risk_limit_breaches",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: _format_decimal(value, digits=0))
    return formatted


def _format_portfolio_risk_contribution(portfolio_risk_contribution: pd.DataFrame | None) -> pd.DataFrame:
    """Format portfolio risk contribution metrics for report presentation."""
    if portfolio_risk_contribution is None or portfolio_risk_contribution.empty:
        return pd.DataFrame()
    formatted = portfolio_risk_contribution.astype(object).copy()
    for column in [
        "weight",
        "absolute_risk_contribution",
        "percent_risk_contribution",
        "portfolio_volatility",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_percent)
    if "marginal_contribution_to_risk" in formatted.columns:
        formatted["marginal_contribution_to_risk"] = formatted["marginal_contribution_to_risk"].map(_format_decimal)
    return formatted


def _format_portfolio_score_summary(portfolio_score_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format portfolio score summary for report presentation."""
    if portfolio_score_summary is None or portfolio_score_summary.empty:
        return pd.DataFrame()
    formatted = portfolio_score_summary.astype(object).copy()
    for column in [
        "return_score",
        "risk_control_score",
        "risk_adjusted_score",
        "stability_score",
        "executability_score",
        "total_score",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_decimal)
    for column in ["score_pct", "monthly_win_rate", "annual_win_rate", "avg_turnover", "total_transaction_cost_drag"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_percent)
    if "rank" in formatted.columns:
        formatted["rank"] = formatted["rank"].map(lambda value: _format_decimal(value, digits=0))
    return formatted


def _format_portfolio_evaluation_summary(portfolio_evaluation_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format portfolio evaluation summary for report presentation."""
    if portfolio_evaluation_summary is None or portfolio_evaluation_summary.empty:
        return pd.DataFrame()
    formatted = portfolio_evaluation_summary.astype(object).copy()
    for column in ["monthly_win_rate", "annual_win_rate"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_percent)
    if "max_drawdown_recovery_days" in formatted.columns:
        formatted["max_drawdown_recovery_days"] = formatted["max_drawdown_recovery_days"].map(
            lambda value: _format_decimal(value, digits=0)
        )
    if "rolling_sharpe_stability" in formatted.columns:
        formatted["rolling_sharpe_stability"] = formatted["rolling_sharpe_stability"].map(_format_decimal)
    return formatted


def _format_macro_regime_summary(macro_regime_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format macro regime summary rows for report presentation."""
    if macro_regime_summary is None or macro_regime_summary.empty:
        return pd.DataFrame()

    formatted = macro_regime_summary.astype(object).copy()
    for column in ["latest_value", "reference_value"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_decimal)
    return formatted


def _format_risk_switch_summary(risk_switch_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Format risk-switch summary diagnostics for report presentation."""
    if risk_switch_summary is None or risk_switch_summary.empty:
        return pd.DataFrame()

    formatted = risk_switch_summary.astype(object).copy()
    for column in ["observations", "risk_switch_active_days", "max_reduced_assets"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: _format_decimal(value, digits=0))
    if "risk_switch_active_ratio" in formatted.columns:
        formatted["risk_switch_active_ratio"] = formatted["risk_switch_active_ratio"].map(_format_percent)
    if "avg_reduced_assets" in formatted.columns:
        formatted["avg_reduced_assets"] = formatted["avg_reduced_assets"].map(_format_decimal)
    return formatted


def build_run_configuration_summary(
    *,
    start: str,
    end: str | None,
    template_name: str | None,
    backtest_universe_mode: str,
    rolling_window: int,
    config_paths: dict[str, str | Path],
) -> pd.DataFrame:
    """Build a compact report table describing run parameters and config inputs."""
    rows = {
        "start": start,
        "end": end,
        "template_name": template_name if template_name is not None else "default",
        "backtest_universe_mode": backtest_universe_mode,
        "rolling_window": rolling_window,
    }
    rows.update(
        {
            f"config_{name}": str(Path(path))
            for name, path in sorted(config_paths.items())
        }
    )
    return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])


def build_phase1_report_markdown(
    strategy_name: str,
    performance_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_comparisons: pd.DataFrame,
    benchmark_annual_excess_returns: pd.DataFrame,
    benchmark_drawdown_comparisons: pd.DataFrame,
    liquidity_table: pd.DataFrame,
    etf_summary: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    chart_paths: dict[str, Path],
    report_date: str,
    data_quality_summary: pd.DataFrame | None = None,
    trend_filter_summary: pd.DataFrame | None = None,
    risk_switch_summary: pd.DataFrame | None = None,
    run_configuration: pd.DataFrame | None = None,
    rolling_metric_snapshot: pd.DataFrame | None = None,
    rolling_correlation: pd.DataFrame | None = None,
    rebalance_reason_table: pd.DataFrame | None = None,
    risk_limit_checks: pd.DataFrame | None = None,
    risk_limit_breaches: pd.DataFrame | None = None,
    risk_limit_breach_summary: pd.DataFrame | None = None,
    pipeline_health_summary: pd.DataFrame | None = None,
    portfolio_risk_contribution: pd.DataFrame | None = None,
    portfolio_score_summary: pd.DataFrame | None = None,
    portfolio_evaluation_summary: pd.DataFrame | None = None,
    macro_regime_summary: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> str:
    """Build a concise Markdown report from Phase 1 pipeline outputs."""
    strategy_row = performance_summary.loc[strategy_name]
    investable = liquidity_table.index[liquidity_table["passes_liquidity_filter"]].tolist()
    non_liquid = liquidity_table.index[~liquidity_table["passes_liquidity_filter"]].tolist()

    performance_view = performance_summary.copy()
    for column in [
        "annualized_return",
        "annualized_volatility",
        "downside_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "ending_nav",
        "total_turnover",
        "total_transaction_cost_drag",
    ]:
        if column in performance_view.columns:
            formatter = _format_percent if "return" in column or "volatility" in column or "drawdown" in column or "drag" in column else _format_decimal
            if column in {"ending_nav", "total_turnover", "sharpe_ratio", "sortino_ratio", "calmar_ratio"}:
                formatter = _format_decimal
            performance_view[column] = performance_view[column].map(formatter)

    turnover_view = turnover_summary.copy()
    for column in ["total_turnover", "average_turnover", "total_transaction_cost_drag"]:
        if column in turnover_view.columns:
            turnover_view[column] = turnover_view[column].map(_format_decimal)

    annual_view = annual_return_table.copy()
    for column in annual_view.columns:
        annual_view[column] = annual_view[column].map(_format_percent)

    benchmark_view = benchmark_comparisons.copy()
    for column in benchmark_view.columns:
        if "ratio" in column:
            benchmark_view[column] = benchmark_view[column].map(_format_decimal)
        else:
            benchmark_view[column] = benchmark_view[column].map(_format_percent)

    excess_view = benchmark_annual_excess_returns.copy()
    for column in excess_view.columns:
        excess_view[column] = excess_view[column].map(_format_percent)

    drawdown_view = benchmark_drawdown_comparisons.copy()
    for column in drawdown_view.columns:
        drawdown_view[column] = drawdown_view[column].map(_format_percent)

    etf_view = etf_summary[
        [
            "asset_class",
            "average_dollar_volume",
            "recent_pass_ratio",
            "passes_liquidity_filter",
            "phase1_total_score",
            "phase1_rank",
        ]
    ].copy()
    etf_view["average_dollar_volume"] = etf_view["average_dollar_volume"].map(lambda value: _format_decimal(value, digits=0))
    etf_view["recent_pass_ratio"] = etf_view["recent_pass_ratio"].map(_format_percent)
    etf_view["phase1_total_score"] = etf_view["phase1_total_score"].map(_format_decimal)

    risk_summary_tables = build_phase1_risk_summary_tables(
        correlation_pairs=correlation_pairs,
        correlation_matrix=correlation_matrix,
        covariance_matrix=covariance_matrix,
    )
    correlation_summary = risk_summary_tables["top_correlation_pairs"]
    asset_risk_snapshot = risk_summary_tables["asset_risk_snapshot"]
    rolling_view = _format_rolling_metric_snapshot(rolling_metric_snapshot)
    rolling_correlation_view = _format_rolling_correlation(rolling_correlation)
    data_quality_view = _format_data_quality_summary(data_quality_summary)
    rebalance_summary_view = _format_rebalance_reason_summary(build_rebalance_reason_summary(rebalance_reason_table))
    recent_rebalance_events = _build_recent_rebalance_events(rebalance_reason_table)
    risk_limit_view = _format_risk_limit_checks(risk_limit_checks)
    risk_limit_breach_view = _format_risk_limit_breaches(risk_limit_breaches)
    risk_limit_breach_summary_view = _format_risk_limit_breach_summary(risk_limit_breach_summary)
    pipeline_health_view = _format_pipeline_health_summary(pipeline_health_summary)
    portfolio_risk_contribution_view = _format_portfolio_risk_contribution(portfolio_risk_contribution)
    portfolio_score_view = _format_portfolio_score_summary(portfolio_score_summary)
    portfolio_evaluation_view = _format_portfolio_evaluation_summary(portfolio_evaluation_summary)
    macro_regime_view = _format_macro_regime_summary(macro_regime_summary)
    trend_view = trend_filter_summary if trend_filter_summary is not None else pd.DataFrame()
    risk_switch_view = _format_risk_switch_summary(risk_switch_summary)
    run_config_view = run_configuration if run_configuration is not None else pd.DataFrame()

    note_lines = "\n".join(f"- {note}" for note in notes) if notes else "- None"
    figure_lines = "\n".join(f"- `{name}`: `{path.as_posix()}`" for name, path in chart_paths.items())

    report = f"""# Phase 1 Pipeline Report

Generated: {report_date}

## Executive Summary

- Strategy: `{strategy_name}`
- Ending NAV: {_format_decimal(strategy_row["ending_nav"])}
- Annualized return: {_format_percent(strategy_row["annualized_return"])}
- Annualized volatility: {_format_percent(strategy_row["annualized_volatility"])}
- Max drawdown: {_format_percent(strategy_row["max_drawdown"])}
- Investable liquidity-passing ETFs: {", ".join(investable) if investable else "None"}
- Liquidity-screen failures: {", ".join(non_liquid) if non_liquid else "None"}

## Notes

{note_lines}

## Run Configuration

{dataframe_to_markdown_table(run_config_view) if not run_config_view.empty else "No run configuration generated."}

## Performance Summary

{dataframe_to_markdown_table(performance_view)}

## Turnover Summary

{dataframe_to_markdown_table(turnover_view)}

## Annual Return Table

{dataframe_to_markdown_table(annual_view)}

## Benchmark Comparisons

{dataframe_to_markdown_table(benchmark_view) if not benchmark_view.empty else "No benchmark comparisons generated."}

## Benchmark Annual Excess Returns

{dataframe_to_markdown_table(excess_view) if not excess_view.empty else "No benchmark annual excess returns generated."}

## Benchmark Drawdown Comparisons

{dataframe_to_markdown_table(drawdown_view) if not drawdown_view.empty else "No benchmark drawdown comparisons generated."}

## Latest Rolling Metrics

{dataframe_to_markdown_table(rolling_view) if not rolling_view.empty else "No rolling metrics generated."}

## Rolling Correlation (VTI vs AGG)

{dataframe_to_markdown_table(rolling_correlation_view.tail(20)) if not rolling_correlation_view.empty else "No rolling correlation generated."}

## Trend Filter Summary

{dataframe_to_markdown_table(trend_view) if not trend_view.empty else "No trend filter summary generated."}

## Risk Switch Summary

{dataframe_to_markdown_table(risk_switch_view) if not risk_switch_view.empty else "No risk switch summary generated."}

## Rebalance Reason Summary

{dataframe_to_markdown_table(rebalance_summary_view) if not rebalance_summary_view.empty else "No rebalance reason summary generated."}

## Recent Rebalance Events

{dataframe_to_markdown_table(recent_rebalance_events) if not recent_rebalance_events.empty else "No recent rebalance events generated."}

## Risk Limit Checks

{dataframe_to_markdown_table(risk_limit_view) if not risk_limit_view.empty else "No risk limit checks generated."}

## Risk Limit Breaches

{dataframe_to_markdown_table(risk_limit_breach_view) if not risk_limit_breach_view.empty else "No risk limit breaches generated."}

## Risk Limit Breach Summary

{dataframe_to_markdown_table(risk_limit_breach_summary_view) if not risk_limit_breach_summary_view.empty else "No risk limit breach summary generated."}

## Pipeline Health Summary

{dataframe_to_markdown_table(pipeline_health_view) if not pipeline_health_view.empty else "No pipeline health summary generated."}

## Macro Regime Summary

{dataframe_to_markdown_table(macro_regime_view) if not macro_regime_view.empty else "No macro regime summary generated."}

## Portfolio Risk Contribution

{dataframe_to_markdown_table(portfolio_risk_contribution_view) if not portfolio_risk_contribution_view.empty else "No portfolio risk contribution generated."}

## Portfolio Score Summary

{dataframe_to_markdown_table(portfolio_score_view) if not portfolio_score_view.empty else "No portfolio score summary generated."}

## Portfolio Evaluation Summary

{dataframe_to_markdown_table(portfolio_evaluation_view) if not portfolio_evaluation_view.empty else "No portfolio evaluation summary generated."}

## ETF Summary

{dataframe_to_markdown_table(etf_view)}

## Data Quality Summary

{dataframe_to_markdown_table(data_quality_view) if not data_quality_view.empty else "No data quality summary generated."}

## Correlation Highlights

{dataframe_to_markdown_table(correlation_summary) if not correlation_summary.empty else "No non-diagonal correlation pairs available."}

## Asset Risk Snapshot

{dataframe_to_markdown_table(asset_risk_snapshot) if not asset_risk_snapshot.empty else "No asset risk snapshot available."}

## Figures

{figure_lines}
"""
    return report


def build_phase1_report_html(
    strategy_name: str,
    performance_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_comparisons: pd.DataFrame,
    benchmark_annual_excess_returns: pd.DataFrame,
    benchmark_drawdown_comparisons: pd.DataFrame,
    liquidity_table: pd.DataFrame,
    etf_summary: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    chart_paths: dict[str, Path],
    report_date: str,
    data_quality_summary: pd.DataFrame | None = None,
    trend_filter_summary: pd.DataFrame | None = None,
    risk_switch_summary: pd.DataFrame | None = None,
    run_configuration: pd.DataFrame | None = None,
    rolling_metric_snapshot: pd.DataFrame | None = None,
    rolling_correlation: pd.DataFrame | None = None,
    rebalance_reason_table: pd.DataFrame | None = None,
    risk_limit_checks: pd.DataFrame | None = None,
    risk_limit_breaches: pd.DataFrame | None = None,
    risk_limit_breach_summary: pd.DataFrame | None = None,
    pipeline_health_summary: pd.DataFrame | None = None,
    portfolio_risk_contribution: pd.DataFrame | None = None,
    portfolio_score_summary: pd.DataFrame | None = None,
    portfolio_evaluation_summary: pd.DataFrame | None = None,
    macro_regime_summary: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> str:
    """Build a shareable HTML report from Phase 1 pipeline outputs."""
    strategy_row = performance_summary.loc[strategy_name]
    investable = liquidity_table.index[liquidity_table["passes_liquidity_filter"]].tolist()
    non_liquid = liquidity_table.index[~liquidity_table["passes_liquidity_filter"]].tolist()

    performance_view = performance_summary.copy()
    for column in [
        "annualized_return",
        "annualized_volatility",
        "downside_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "ending_nav",
        "total_turnover",
        "total_transaction_cost_drag",
    ]:
        if column in performance_view.columns:
            formatter = _format_percent if "return" in column or "volatility" in column or "drawdown" in column or "drag" in column else _format_decimal
            if column in {"ending_nav", "total_turnover", "sharpe_ratio", "sortino_ratio", "calmar_ratio"}:
                formatter = _format_decimal
            performance_view[column] = performance_view[column].map(formatter)

    turnover_view = turnover_summary.copy()
    for column in ["total_turnover", "average_turnover", "total_transaction_cost_drag"]:
        if column in turnover_view.columns:
            turnover_view[column] = turnover_view[column].map(_format_decimal)

    annual_view = annual_return_table.copy()
    for column in annual_view.columns:
        annual_view[column] = annual_view[column].map(_format_percent)

    benchmark_view = benchmark_comparisons.copy()
    for column in benchmark_view.columns:
        if "ratio" in column:
            benchmark_view[column] = benchmark_view[column].map(_format_decimal)
        else:
            benchmark_view[column] = benchmark_view[column].map(_format_percent)

    excess_view = benchmark_annual_excess_returns.copy()
    for column in excess_view.columns:
        excess_view[column] = excess_view[column].map(_format_percent)

    drawdown_view = benchmark_drawdown_comparisons.copy()
    for column in drawdown_view.columns:
        drawdown_view[column] = drawdown_view[column].map(_format_percent)

    etf_view = etf_summary[
        [
            "asset_class",
            "average_dollar_volume",
            "recent_pass_ratio",
            "passes_liquidity_filter",
            "phase1_total_score",
            "phase1_rank",
        ]
    ].copy()
    etf_view["average_dollar_volume"] = etf_view["average_dollar_volume"].map(lambda value: _format_decimal(value, digits=0))
    etf_view["recent_pass_ratio"] = etf_view["recent_pass_ratio"].map(_format_percent)
    etf_view["phase1_total_score"] = etf_view["phase1_total_score"].map(_format_decimal)

    risk_summary_tables = build_phase1_risk_summary_tables(
        correlation_pairs=correlation_pairs,
        correlation_matrix=correlation_matrix,
        covariance_matrix=covariance_matrix,
    )
    correlation_summary = risk_summary_tables["top_correlation_pairs"]
    asset_risk_snapshot = risk_summary_tables["asset_risk_snapshot"]
    rolling_view = _format_rolling_metric_snapshot(rolling_metric_snapshot)
    rolling_correlation_view = _format_rolling_correlation(rolling_correlation)
    data_quality_view = _format_data_quality_summary(data_quality_summary)
    rebalance_summary_view = _format_rebalance_reason_summary(build_rebalance_reason_summary(rebalance_reason_table))
    recent_rebalance_events = _build_recent_rebalance_events(rebalance_reason_table)
    risk_limit_view = _format_risk_limit_checks(risk_limit_checks)
    risk_limit_breach_view = _format_risk_limit_breaches(risk_limit_breaches)
    risk_limit_breach_summary_view = _format_risk_limit_breach_summary(risk_limit_breach_summary)
    pipeline_health_view = _format_pipeline_health_summary(pipeline_health_summary)
    portfolio_risk_contribution_view = _format_portfolio_risk_contribution(portfolio_risk_contribution)
    portfolio_score_view = _format_portfolio_score_summary(portfolio_score_summary)
    portfolio_evaluation_view = _format_portfolio_evaluation_summary(portfolio_evaluation_summary)
    macro_regime_view = _format_macro_regime_summary(macro_regime_summary)
    trend_view = trend_filter_summary if trend_filter_summary is not None else pd.DataFrame()
    risk_switch_view = _format_risk_switch_summary(risk_switch_summary)
    run_config_view = run_configuration if run_configuration is not None else pd.DataFrame()

    note_items = "".join(f"<li>{escape(note)}</li>" for note in notes) if notes else "<li>None</li>"
    figure_items = "".join(
        f"<li><strong>{escape(name)}</strong>: {escape(path.as_posix())}</li>"
        for name, path in chart_paths.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phase 1 Pipeline Report</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 32px; color: #1f2933; background: #f7f5ef; }}
    h1, h2 {{ color: #102a43; }}
    .summary {{ background: #fffdf7; border: 1px solid #d9d0bb; padding: 16px 20px; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th, td {{ border: 1px solid #d9d0bb; padding: 8px 10px; text-align: left; }}
    th {{ background: #efe7d3; }}
    ul {{ margin-top: 8px; }}
    .meta {{ color: #486581; }}
  </style>
</head>
<body>
  <h1>Phase 1 Pipeline Report</h1>
  <p class="meta">Generated: {escape(report_date)}</p>

  <section class="summary">
    <h2>Executive Summary</h2>
    <ul>
      <li>Strategy: <code>{escape(strategy_name)}</code></li>
      <li>Ending NAV: {_format_decimal(strategy_row["ending_nav"])}</li>
      <li>Annualized return: {_format_percent(strategy_row["annualized_return"])}</li>
      <li>Annualized volatility: {_format_percent(strategy_row["annualized_volatility"])}</li>
      <li>Max drawdown: {_format_percent(strategy_row["max_drawdown"])}</li>
      <li>Investable liquidity-passing ETFs: {escape(", ".join(investable) if investable else "None")}</li>
      <li>Liquidity-screen failures: {escape(", ".join(non_liquid) if non_liquid else "None")}</li>
    </ul>
  </section>

  <section>
    <h2>Notes</h2>
    <ul>{note_items}</ul>
  </section>

  <section><h2>Run Configuration</h2>{dataframe_to_html_table(run_config_view) if not run_config_view.empty else "<p>No run configuration generated.</p>"}</section>
  <section><h2>Performance Summary</h2>{dataframe_to_html_table(performance_view)}</section>
  <section><h2>Turnover Summary</h2>{dataframe_to_html_table(turnover_view)}</section>
  <section><h2>Annual Return Table</h2>{dataframe_to_html_table(annual_view)}</section>
  <section><h2>Benchmark Comparisons</h2>{dataframe_to_html_table(benchmark_view) if not benchmark_view.empty else "<p>No benchmark comparisons generated.</p>"}</section>
  <section><h2>Benchmark Annual Excess Returns</h2>{dataframe_to_html_table(excess_view) if not excess_view.empty else "<p>No benchmark annual excess returns generated.</p>"}</section>
  <section><h2>Benchmark Drawdown Comparisons</h2>{dataframe_to_html_table(drawdown_view) if not drawdown_view.empty else "<p>No benchmark drawdown comparisons generated.</p>"}</section>
  <section><h2>Latest Rolling Metrics</h2>{dataframe_to_html_table(rolling_view) if not rolling_view.empty else "<p>No rolling metrics generated.</p>"}</section>
  <section><h2>Rolling Correlation (VTI vs AGG)</h2>{dataframe_to_html_table(rolling_correlation_view.tail(20)) if not rolling_correlation_view.empty else "<p>No rolling correlation generated.</p>"}</section>
  <section><h2>Trend Filter Summary</h2>{dataframe_to_html_table(trend_view) if not trend_view.empty else "<p>No trend filter summary generated.</p>"}</section>
  <section><h2>Risk Switch Summary</h2>{dataframe_to_html_table(risk_switch_view) if not risk_switch_view.empty else "<p>No risk switch summary generated.</p>"}</section>
  <section><h2>Rebalance Reason Summary</h2>{dataframe_to_html_table(rebalance_summary_view) if not rebalance_summary_view.empty else "<p>No rebalance reason summary generated.</p>"}</section>
  <section><h2>Recent Rebalance Events</h2>{dataframe_to_html_table(recent_rebalance_events) if not recent_rebalance_events.empty else "<p>No recent rebalance events generated.</p>"}</section>
  <section><h2>Risk Limit Checks</h2>{dataframe_to_html_table(risk_limit_view) if not risk_limit_view.empty else "<p>No risk limit checks generated.</p>"}</section>
  <section><h2>Risk Limit Breaches</h2>{dataframe_to_html_table(risk_limit_breach_view) if not risk_limit_breach_view.empty else "<p>No risk limit breaches generated.</p>"}</section>
  <section><h2>Risk Limit Breach Summary</h2>{dataframe_to_html_table(risk_limit_breach_summary_view) if not risk_limit_breach_summary_view.empty else "<p>No risk limit breach summary generated.</p>"}</section>
  <section><h2>Pipeline Health Summary</h2>{dataframe_to_html_table(pipeline_health_view) if not pipeline_health_view.empty else "<p>No pipeline health summary generated.</p>"}</section>
  <section><h2>Macro Regime Summary</h2>{dataframe_to_html_table(macro_regime_view) if not macro_regime_view.empty else "<p>No macro regime summary generated.</p>"}</section>
  <section><h2>Portfolio Risk Contribution</h2>{dataframe_to_html_table(portfolio_risk_contribution_view) if not portfolio_risk_contribution_view.empty else "<p>No portfolio risk contribution generated.</p>"}</section>
  <section><h2>Portfolio Score Summary</h2>{dataframe_to_html_table(portfolio_score_view) if not portfolio_score_view.empty else "<p>No portfolio score summary generated.</p>"}</section>
  <section><h2>Portfolio Evaluation Summary</h2>{dataframe_to_html_table(portfolio_evaluation_view) if not portfolio_evaluation_view.empty else "<p>No portfolio evaluation summary generated.</p>"}</section>
  <section><h2>ETF Summary</h2>{dataframe_to_html_table(etf_view)}</section>
  <section><h2>Data Quality Summary</h2>{dataframe_to_html_table(data_quality_view) if not data_quality_view.empty else "<p>No data quality summary generated.</p>"}</section>
  <section><h2>Correlation Highlights</h2>{dataframe_to_html_table(correlation_summary) if not correlation_summary.empty else "<p>No non-diagonal correlation pairs available.</p>"}</section>
  <section><h2>Asset Risk Snapshot</h2>{dataframe_to_html_table(asset_risk_snapshot) if not asset_risk_snapshot.empty else "<p>No asset risk snapshot available.</p>"}</section>
  <section><h2>Figures</h2><ul>{figure_items}</ul></section>
</body>
</html>
"""


def write_phase1_report(
    strategy_name: str,
    performance_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_comparisons: pd.DataFrame,
    benchmark_annual_excess_returns: pd.DataFrame,
    benchmark_drawdown_comparisons: pd.DataFrame,
    liquidity_table: pd.DataFrame,
    etf_summary: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    chart_paths: dict[str, Path],
    output_path: str | Path,
    report_date: str,
    data_quality_summary: pd.DataFrame | None = None,
    trend_filter_summary: pd.DataFrame | None = None,
    risk_switch_summary: pd.DataFrame | None = None,
    run_configuration: pd.DataFrame | None = None,
    rolling_metric_snapshot: pd.DataFrame | None = None,
    rolling_correlation: pd.DataFrame | None = None,
    rebalance_reason_table: pd.DataFrame | None = None,
    risk_limit_checks: pd.DataFrame | None = None,
    risk_limit_breaches: pd.DataFrame | None = None,
    risk_limit_breach_summary: pd.DataFrame | None = None,
    pipeline_health_summary: pd.DataFrame | None = None,
    portfolio_risk_contribution: pd.DataFrame | None = None,
    portfolio_score_summary: pd.DataFrame | None = None,
    portfolio_evaluation_summary: pd.DataFrame | None = None,
    macro_regime_summary: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write the Phase 1 Markdown report to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_phase1_report_markdown(
        strategy_name=strategy_name,
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=annual_return_table,
        benchmark_comparisons=benchmark_comparisons,
        benchmark_annual_excess_returns=benchmark_annual_excess_returns,
        benchmark_drawdown_comparisons=benchmark_drawdown_comparisons,
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=data_quality_summary,
        covariance_matrix=covariance_matrix,
        correlation_matrix=correlation_matrix,
        correlation_pairs=correlation_pairs,
        chart_paths=chart_paths,
        report_date=report_date,
        trend_filter_summary=trend_filter_summary,
        risk_switch_summary=risk_switch_summary,
        rolling_metric_snapshot=rolling_metric_snapshot,
        rolling_correlation=rolling_correlation,
        rebalance_reason_table=rebalance_reason_table,
        risk_limit_checks=risk_limit_checks,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        pipeline_health_summary=pipeline_health_summary,
        portfolio_risk_contribution=portfolio_risk_contribution,
        portfolio_score_summary=portfolio_score_summary,
        portfolio_evaluation_summary=portfolio_evaluation_summary,
        macro_regime_summary=macro_regime_summary,
        run_configuration=run_configuration,
        notes=notes,
    )
    output.write_text(report, encoding="utf-8")
    return output


def write_phase1_html_report(
    strategy_name: str,
    performance_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    benchmark_comparisons: pd.DataFrame,
    benchmark_annual_excess_returns: pd.DataFrame,
    benchmark_drawdown_comparisons: pd.DataFrame,
    liquidity_table: pd.DataFrame,
    etf_summary: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    chart_paths: dict[str, Path],
    output_path: str | Path,
    report_date: str,
    data_quality_summary: pd.DataFrame | None = None,
    trend_filter_summary: pd.DataFrame | None = None,
    risk_switch_summary: pd.DataFrame | None = None,
    run_configuration: pd.DataFrame | None = None,
    rolling_metric_snapshot: pd.DataFrame | None = None,
    rolling_correlation: pd.DataFrame | None = None,
    rebalance_reason_table: pd.DataFrame | None = None,
    risk_limit_checks: pd.DataFrame | None = None,
    risk_limit_breaches: pd.DataFrame | None = None,
    risk_limit_breach_summary: pd.DataFrame | None = None,
    pipeline_health_summary: pd.DataFrame | None = None,
    portfolio_risk_contribution: pd.DataFrame | None = None,
    portfolio_score_summary: pd.DataFrame | None = None,
    portfolio_evaluation_summary: pd.DataFrame | None = None,
    macro_regime_summary: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write the Phase 1 HTML report to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_phase1_report_html(
        strategy_name=strategy_name,
        performance_summary=performance_summary,
        turnover_summary=turnover_summary,
        annual_return_table=annual_return_table,
        benchmark_comparisons=benchmark_comparisons,
        benchmark_annual_excess_returns=benchmark_annual_excess_returns,
        benchmark_drawdown_comparisons=benchmark_drawdown_comparisons,
        liquidity_table=liquidity_table,
        etf_summary=etf_summary,
        data_quality_summary=data_quality_summary,
        covariance_matrix=covariance_matrix,
        correlation_matrix=correlation_matrix,
        correlation_pairs=correlation_pairs,
        chart_paths=chart_paths,
        report_date=report_date,
        trend_filter_summary=trend_filter_summary,
        risk_switch_summary=risk_switch_summary,
        rolling_metric_snapshot=rolling_metric_snapshot,
        rolling_correlation=rolling_correlation,
        rebalance_reason_table=rebalance_reason_table,
        risk_limit_checks=risk_limit_checks,
        risk_limit_breaches=risk_limit_breaches,
        risk_limit_breach_summary=risk_limit_breach_summary,
        pipeline_health_summary=pipeline_health_summary,
        portfolio_risk_contribution=portfolio_risk_contribution,
        portfolio_score_summary=portfolio_score_summary,
        portfolio_evaluation_summary=portfolio_evaluation_summary,
        macro_regime_summary=macro_regime_summary,
        run_configuration=run_configuration,
        notes=notes,
    )
    output.write_text(report, encoding="utf-8")
    return output
