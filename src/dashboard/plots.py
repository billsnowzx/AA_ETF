"""Phase 1 plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.correlation import correlation_matrix
from src.analytics.drawdown import drawdown_series


def save_nav_chart(nav_table: pd.DataFrame, output_path: str | Path) -> None:
    """Save a NAV line chart for the strategy and benchmarks."""
    figure, axis = plt.subplots(figsize=(10, 6))
    nav_table.plot(ax=axis, linewidth=2)
    axis.set_title("Portfolio NAV")
    axis.set_xlabel("Date")
    axis.set_ylabel("NAV")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_drawdown_chart(nav_table: pd.DataFrame, output_path: str | Path) -> None:
    """Save a drawdown chart derived from NAV series."""
    drawdowns = drawdown_series(nav_table)
    figure, axis = plt.subplots(figsize=(10, 6))
    drawdowns.plot(ax=axis, linewidth=2)
    axis.set_title("Drawdown")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_annual_return_bar_chart(annual_return_table: pd.DataFrame, output_path: str | Path) -> None:
    """Save a grouped annual return bar chart."""
    figure, axis = plt.subplots(figsize=(10, 6))
    annual_return_table.plot(kind="bar", ax=axis)
    axis.set_title("Annual Returns")
    axis.set_xlabel("Year")
    axis.set_ylabel("Return")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_correlation_heatmap(returns: pd.DataFrame, output_path: str | Path) -> None:
    """Save a simple correlation heatmap from an asset return matrix."""
    matrix = correlation_matrix(returns)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_title("Correlation Heatmap")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_yticks(range(len(matrix.index)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right")
    axis.set_yticklabels(matrix.index)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_rolling_metric_chart(
    metric_table: pd.DataFrame,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """Save a rolling metric line chart."""
    figure, axis = plt.subplots(figsize=(10, 6))
    metric_table.plot(ax=axis, linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_risk_contribution_chart(
    risk_contribution_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a bar chart of asset percent risk contributions."""
    figure, axis = plt.subplots(figsize=(10, 6))
    risk_contribution_table["percent_risk_contribution"].plot(kind="bar", ax=axis)
    axis.set_title("Risk Contribution")
    axis.set_xlabel("Asset")
    axis.set_ylabel("Percent of Portfolio Risk")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_marginal_contribution_to_risk_chart(
    risk_contribution_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a bar chart of marginal contribution to risk by asset."""
    figure, axis = plt.subplots(figsize=(10, 6))
    risk_contribution_table["marginal_contribution_to_risk"].plot(kind="bar", ax=axis)
    axis.set_title("Marginal Contribution to Risk")
    axis.set_xlabel("Asset")
    axis.set_ylabel("MCTR")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_phase1_chart_outputs(
    strategy_name: str,
    nav_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    asset_returns: pd.DataFrame,
    output_dir: str | Path,
    rolling_volatility_table: pd.DataFrame | None = None,
    rolling_sharpe_table: pd.DataFrame | None = None,
    rolling_correlation_table: pd.DataFrame | None = None,
    risk_contribution_table: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write the required Phase 1 charts to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chart_paths = {
        "nav_chart": output_path / f"{strategy_name}_nav.png",
        "drawdown_chart": output_path / f"{strategy_name}_drawdown.png",
        "annual_return_chart": output_path / f"{strategy_name}_annual_returns.png",
        "correlation_heatmap": output_path / "correlation_heatmap.png",
    }
    if rolling_volatility_table is not None and not rolling_volatility_table.empty:
        chart_paths["rolling_volatility_chart"] = output_path / f"{strategy_name}_rolling_volatility.png"
    if rolling_sharpe_table is not None and not rolling_sharpe_table.empty:
        chart_paths["rolling_sharpe_chart"] = output_path / f"{strategy_name}_rolling_sharpe.png"
    if rolling_correlation_table is not None and not rolling_correlation_table.empty:
        chart_paths["rolling_correlation_chart"] = output_path / f"{strategy_name}_rolling_correlation.png"
    if risk_contribution_table is not None and not risk_contribution_table.empty:
        chart_paths["risk_contribution_chart"] = output_path / f"{strategy_name}_risk_contribution.png"
        chart_paths["mctr_chart"] = output_path / f"{strategy_name}_mctr.png"

    save_nav_chart(nav_table, chart_paths["nav_chart"])
    save_drawdown_chart(nav_table, chart_paths["drawdown_chart"])
    save_annual_return_bar_chart(annual_return_table, chart_paths["annual_return_chart"])
    save_correlation_heatmap(asset_returns, chart_paths["correlation_heatmap"])
    if "rolling_volatility_chart" in chart_paths:
        save_rolling_metric_chart(
            rolling_volatility_table,
            chart_paths["rolling_volatility_chart"],
            title="Rolling Volatility",
            ylabel="Annualized Volatility",
        )
    if "rolling_sharpe_chart" in chart_paths:
        save_rolling_metric_chart(
            rolling_sharpe_table,
            chart_paths["rolling_sharpe_chart"],
            title="Rolling Sharpe Ratio",
            ylabel="Sharpe Ratio",
        )
    if "rolling_correlation_chart" in chart_paths:
        save_rolling_metric_chart(
            rolling_correlation_table,
            chart_paths["rolling_correlation_chart"],
            title="Rolling Correlation (VTI vs AGG)",
            ylabel="Correlation",
        )
    if "risk_contribution_chart" in chart_paths:
        save_risk_contribution_chart(risk_contribution_table, chart_paths["risk_contribution_chart"])
        save_marginal_contribution_to_risk_chart(risk_contribution_table, chart_paths["mctr_chart"])
    return chart_paths
