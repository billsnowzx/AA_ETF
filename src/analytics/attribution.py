"""Benchmark-relative performance and attribution helpers."""

from __future__ import annotations

from typing import Union

import pandas as pd

from src.analytics.returns import annualized_return

PandasLike = Union[pd.Series, pd.DataFrame]


def align_return_series(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Inner-join two return series on their shared index."""
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna(how="any")

    if aligned.empty:
        raise ValueError("Strategy and benchmark returns do not overlap after alignment.")

    return aligned


def excess_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Compute strategy excess returns relative to a benchmark."""
    aligned = align_return_series(strategy_returns, benchmark_returns)
    return aligned["strategy"] - aligned["benchmark"]


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized tracking error versus a benchmark."""
    excess = excess_returns(strategy_returns, benchmark_returns)
    return float(excess.std(ddof=1) * (periods_per_year**0.5))


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute information ratio from strategy and benchmark return series."""
    excess = excess_returns(strategy_returns, benchmark_returns)
    ann_excess_return = annualized_return(excess, periods_per_year=periods_per_year)
    te = tracking_error(
        strategy_returns,
        benchmark_returns,
        periods_per_year=periods_per_year,
    )
    if te == 0:
        return float("nan")
    return float(ann_excess_return / te)


def annual_return_table(returns: PandasLike) -> pd.DataFrame:
    """Compound daily or periodic returns into calendar-year returns."""
    if isinstance(returns, pd.Series):
        annual = returns.groupby(returns.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
        return annual.to_frame(name="annual_return")

    annual = returns.groupby(returns.index.year).apply(lambda frame: (1.0 + frame).prod() - 1.0)
    annual.index.name = "year"
    return annual


def benchmark_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> pd.Series:
    """Build a compact benchmark-relative summary for one strategy series."""
    aligned = align_return_series(strategy_returns, benchmark_returns)
    excess = aligned["strategy"] - aligned["benchmark"]

    return pd.Series(
        {
            "strategy_annualized_return": annualized_return(
                aligned["strategy"],
                periods_per_year=periods_per_year,
            ),
            "benchmark_annualized_return": annualized_return(
                aligned["benchmark"],
                periods_per_year=periods_per_year,
            ),
            "annualized_excess_return": annualized_return(
                excess,
                periods_per_year=periods_per_year,
            ),
            "tracking_error": tracking_error(
                aligned["strategy"],
                aligned["benchmark"],
                periods_per_year=periods_per_year,
            ),
            "information_ratio": information_ratio(
                aligned["strategy"],
                aligned["benchmark"],
                periods_per_year=periods_per_year,
            ),
        }
    )
