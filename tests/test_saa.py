import math

import pandas as pd
import pytest

from src.portfolio.saa import (
    align_weights_to_returns,
    static_weight_portfolio_nav,
    static_weight_portfolio_returns,
    static_weight_portfolio_summary,
    static_weight_return_contributions,
)


def test_align_weights_to_returns_preserves_return_matrix_order() -> None:
    asset_returns = pd.DataFrame(columns=["AGG", "VTI"])

    aligned = align_weights_to_returns(asset_returns, {"VTI": 0.6, "AGG": 0.4})

    assert aligned.index.tolist() == ["AGG", "VTI"]
    assert math.isclose(float(aligned["AGG"]), 0.4, rel_tol=1e-9)
    assert math.isclose(float(aligned["VTI"]), 0.6, rel_tol=1e-9)


def test_align_weights_to_returns_rejects_missing_assets() -> None:
    asset_returns = pd.DataFrame(columns=["VTI"])

    with pytest.raises(ValueError, match="missing columns required by weights"):
        align_weights_to_returns(asset_returns, {"VTI": 0.6, "AGG": 0.4})


def test_static_weight_return_contributions_and_portfolio_returns() -> None:
    asset_returns = pd.DataFrame(
        {
            "VTI": [0.01, 0.00],
            "AGG": [0.00, 0.02],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )

    contributions = static_weight_return_contributions(asset_returns, {"VTI": 0.6, "AGG": 0.4})
    portfolio_returns = static_weight_portfolio_returns(asset_returns, {"VTI": 0.6, "AGG": 0.4})

    assert math.isclose(contributions.iloc[0]["VTI"], 0.006, rel_tol=1e-9)
    assert math.isclose(contributions.iloc[1]["AGG"], 0.008, rel_tol=1e-9)
    assert portfolio_returns.tolist() == [0.006, 0.008]


def test_static_weight_portfolio_nav_compounds_returns() -> None:
    asset_returns = pd.DataFrame(
        {"VTI": [0.01, 0.01]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    nav = static_weight_portfolio_nav(asset_returns, {"VTI": 1.0})

    assert nav.name == "portfolio_nav"
    assert nav.tolist() == [1.01, 1.0201]


def test_static_weight_portfolio_summary_includes_benchmark_fields_when_provided() -> None:
    asset_returns = pd.DataFrame(
        {
            "VTI": [0.02, 0.01, -0.01, 0.03],
            "AGG": [0.00, 0.01, 0.00, 0.00],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    benchmark_returns = pd.Series([0.01, 0.005, -0.005, 0.01], index=asset_returns.index)

    summary = static_weight_portfolio_summary(
        asset_returns,
        {"VTI": 0.6, "AGG": 0.4},
        benchmark_returns=benchmark_returns,
        periods_per_year=4,
    )

    assert "annualized_return" in summary.index
    assert "tracking_error" in summary.index
