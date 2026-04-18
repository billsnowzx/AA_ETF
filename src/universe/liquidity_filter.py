"""Liquidity summary and screening logic for ETF universe construction."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

DEFAULT_ADV_THRESHOLD = 50_000_000.0
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_RECENT_WINDOW = 60
DEFAULT_RECENT_PASS_RATIO = 0.80


def summarize_liquidity(
    frames: Mapping[str, pd.DataFrame],
    adv_threshold: float = DEFAULT_ADV_THRESHOLD,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> pd.DataFrame:
    """Build an audit table of liquidity metrics for each ETF."""
    summaries: list[dict[str, object]] = []

    for ticker, frame in frames.items():
        if "dollar_volume" not in frame.columns:
            raise ValueError(f"Ticker '{ticker}' is missing required column 'dollar_volume'.")

        dollar_volume = frame["dollar_volume"].dropna()
        rolling_adv = dollar_volume.rolling(window=rolling_window, min_periods=rolling_window).mean()
        valid_rolling = rolling_adv.dropna()
        has_sufficient_history = len(valid_rolling) >= recent_window

        if has_sufficient_history:
            recent_slice = valid_rolling.tail(recent_window)
            recent_pass_ratio = float((recent_slice > adv_threshold).mean())
            latest_rolling_adv = float(recent_slice.iloc[-1])
        else:
            recent_pass_ratio = float("nan")
            latest_rolling_adv = float(valid_rolling.iloc[-1]) if not valid_rolling.empty else float("nan")

        average_dollar_volume = float(dollar_volume.mean()) if not dollar_volume.empty else float("nan")
        summaries.append(
            {
                "ticker": ticker,
                "observations": int(len(frame)),
                "average_dollar_volume": average_dollar_volume,
                "latest_rolling_average_dollar_volume": latest_rolling_adv,
                "recent_pass_ratio": recent_pass_ratio,
                "has_sufficient_history": has_sufficient_history,
                "passes_average_volume_threshold": bool(average_dollar_volume > adv_threshold)
                if pd.notna(average_dollar_volume)
                else False,
            }
        )

    summary_table = pd.DataFrame(summaries).set_index("ticker").sort_index()
    return summary_table


def filter_liquid_universe(
    frames: Mapping[str, pd.DataFrame],
    adv_threshold: float = DEFAULT_ADV_THRESHOLD,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    recent_pass_ratio_threshold: float = DEFAULT_RECENT_PASS_RATIO,
) -> tuple[list[str], pd.DataFrame]:
    """Filter the ETF universe based on full-sample and recent liquidity stability."""
    summary_table = summarize_liquidity(
        frames=frames,
        adv_threshold=adv_threshold,
        rolling_window=rolling_window,
        recent_window=recent_window,
    )

    summary_table["passes_recent_liquidity_threshold"] = (
        summary_table["recent_pass_ratio"] >= recent_pass_ratio_threshold
    )
    summary_table["passes_liquidity_filter"] = (
        summary_table["has_sufficient_history"]
        & summary_table["passes_average_volume_threshold"]
        & summary_table["passes_recent_liquidity_threshold"]
    )

    filtered_tickers = summary_table.index[summary_table["passes_liquidity_filter"]].tolist()
    return filtered_tickers, summary_table
