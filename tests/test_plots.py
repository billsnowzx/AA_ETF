import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.dashboard.plots import write_phase1_chart_outputs


def test_write_phase1_chart_outputs_creates_required_pngs() -> None:
    output_dir = Path("data/cache") / f"test_plots_{uuid.uuid4().hex}"
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    nav_table = pd.DataFrame(
        {
            "balanced": [1.0, 1.01, 1.02, 1.03],
            "benchmark_a": [1.0, 1.005, 1.01, 1.015],
        },
        index=index,
    )
    annual_return_table = pd.DataFrame(
        {
            "portfolio": [0.10, 0.05],
            "benchmark_a": [0.08, 0.04],
        },
        index=pd.Index([2023, 2024], name="year"),
    )
    asset_returns = pd.DataFrame(
        {
            "VTI": [0.01, 0.02, -0.01, 0.00],
            "AGG": [0.00, 0.01, 0.00, 0.005],
        },
        index=index,
    )
    rolling_volatility = pd.DataFrame(
        {
            "balanced": [None, 0.10, 0.11, 0.12],
            "benchmark_a": [None, 0.08, 0.09, 0.10],
        },
        index=index,
    )
    rolling_sharpe = pd.DataFrame(
        {
            "balanced": [None, 0.5, 0.6, 0.7],
            "benchmark_a": [None, 0.4, 0.5, 0.6],
        },
        index=index,
    )
    rolling_correlation = pd.DataFrame(
        {
            "VTI_AGG_rolling_correlation": [None, 0.2, 0.1, -0.1],
        },
        index=index,
    )
    risk_contribution = pd.DataFrame(
        {
            "weight": [0.6, 0.4],
            "marginal_contribution_to_risk": [0.12, 0.04],
            "absolute_risk_contribution": [0.072, 0.016],
            "percent_risk_contribution": [0.8182, 0.1818],
            "portfolio_volatility": [0.088, 0.088],
        },
        index=pd.Index(["VTI", "AGG"], name="asset"),
    )

    try:
        chart_paths = write_phase1_chart_outputs(
            "balanced",
            nav_table,
            annual_return_table,
            asset_returns,
            output_dir,
            rolling_volatility_table=rolling_volatility,
            rolling_sharpe_table=rolling_sharpe,
            rolling_correlation_table=rolling_correlation,
            risk_contribution_table=risk_contribution,
        )
        assert chart_paths["nav_chart"].exists()
        assert chart_paths["drawdown_chart"].exists()
        assert chart_paths["annual_return_chart"].exists()
        assert chart_paths["correlation_heatmap"].exists()
        assert chart_paths["rolling_volatility_chart"].exists()
        assert chart_paths["rolling_sharpe_chart"].exists()
        assert chart_paths["rolling_correlation_chart"].exists()
        assert chart_paths["risk_contribution_chart"].exists()
        assert chart_paths["mctr_chart"].exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
