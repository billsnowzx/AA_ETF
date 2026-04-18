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


def write_phase1_chart_outputs(
    strategy_name: str,
    nav_table: pd.DataFrame,
    annual_return_table: pd.DataFrame,
    asset_returns: pd.DataFrame,
    output_dir: str | Path,
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

    save_nav_chart(nav_table, chart_paths["nav_chart"])
    save_drawdown_chart(nav_table, chart_paths["drawdown_chart"])
    save_annual_return_bar_chart(annual_return_table, chart_paths["annual_return_chart"])
    save_correlation_heatmap(asset_returns, chart_paths["correlation_heatmap"])
    return chart_paths
