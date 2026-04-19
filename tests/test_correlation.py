import math

import pandas as pd

from src.analytics.correlation import (
    build_adjusted_close_matrix,
    correlation_matrix,
    covariance_matrix,
    matrix_to_long_table,
    return_matrix_from_prices,
    rolling_correlation,
)


def _build_frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"adj_close": values}, index=index)


def test_build_adjusted_close_matrix_preserves_ticker_columns() -> None:
    frames = {
        "VTI": _build_frame([100.0, 101.0, 102.0]),
        "AGG": _build_frame([50.0, 50.5, 51.0]),
    }

    prices = build_adjusted_close_matrix(frames)

    assert list(prices.columns) == ["VTI", "AGG"]
    assert prices.iloc[0].tolist() == [100.0, 50.0]


def test_return_matrix_from_prices_drops_initial_all_nan_row() -> None:
    prices = pd.DataFrame(
        {
            "VTI": [100.0, 110.0, 121.0],
            "AGG": [50.0, 55.0, 60.5],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    returns = return_matrix_from_prices(prices)

    assert len(returns) == 2
    assert math.isclose(returns.iloc[0]["VTI"], 0.10, rel_tol=1e-9)
    assert math.isclose(returns.iloc[1]["AGG"], 0.10, rel_tol=1e-9)


def test_covariance_and_correlation_matrices_have_expected_shape() -> None:
    returns = pd.DataFrame(
        {
            "VTI": [0.01, 0.02, -0.01, 0.0],
            "AGG": [0.005, 0.01, -0.005, 0.0],
        }
    )

    covariance = covariance_matrix(returns)
    correlation = correlation_matrix(returns)

    assert covariance.shape == (2, 2)
    assert correlation.shape == (2, 2)
    assert math.isclose(correlation.loc["VTI", "AGG"], 1.0, rel_tol=1e-9)


def test_rolling_correlation_requires_present_columns() -> None:
    returns = pd.DataFrame({"VTI": [0.01, 0.02, 0.03]})

    try:
        rolling_correlation(returns, left="VTI", right="AGG", window=2)
    except ValueError as exc:
        assert "Missing columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing rolling correlation column.")


def test_matrix_to_long_table_returns_upper_triangle_pairs() -> None:
    matrix = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["VTI", "AGG"],
        columns=["VTI", "AGG"],
    )

    pairs = matrix_to_long_table(matrix, "correlation", include_diagonal=False)

    assert pairs.to_dict(orient="records") == [
        {"left": "VTI", "right": "AGG", "correlation": 0.5},
    ]
