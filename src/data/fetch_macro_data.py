"""Macro market data download utilities for Phase 1 monitoring."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.fetch_prices import configure_yfinance_tz_cache
from src.utils.dates import validate_date_range

LOGGER = logging.getLogger(__name__)

DEFAULT_MACRO_SERIES_MAP: dict[str, str] = {
    "us10y_yield": "^TNX",
    "us2y_yield": "^IRX",
    "usd_index": "DX-Y.NYB",
    "vix": "^VIX",
    "gold_spot_proxy": "GLD",
    "oil_wti": "CL=F",
    "hyg": "HYG",
    "lqd": "LQD",
}


def _standardize_macro_download_frame(raw_frame: pd.DataFrame, symbol: str) -> pd.Series:
    """Extract a single adjusted-close-like series from yfinance output."""
    if raw_frame.empty:
        raise ValueError(f"No macro data returned for symbol '{symbol}'.")

    frame = raw_frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    value_column = "Adj Close" if "Adj Close" in frame.columns else "Close"
    if value_column not in frame.columns:
        raise ValueError(f"Symbol '{symbol}' is missing both 'Adj Close' and 'Close' columns.")

    series = frame[value_column].astype(float).copy()
    series.index = pd.to_datetime(series.index)
    series.name = symbol
    return series


def fetch_macro_series(
    start: str | None = None,
    end: str | None = None,
    series_map: dict[str, str] | None = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch macro observation series and return a wide DataFrame by canonical series name."""
    start, end = validate_date_range(start, end)
    active_map = series_map.copy() if series_map is not None else DEFAULT_MACRO_SERIES_MAP.copy()
    configure_yfinance_tz_cache()
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1.")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0.")

    result: dict[str, pd.Series] = {}
    for series_name, symbol in active_map.items():
        series: pd.Series | None = None
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            LOGGER.info(
                "Downloading macro data for %s (%s), attempt %s/%s",
                series_name,
                symbol,
                attempt,
                max_retries,
            )
            raw_frame = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                actions=False,
            )
            try:
                series = _standardize_macro_download_frame(raw_frame, symbol=symbol)
                break
            except ValueError as exc:
                last_error = exc
                if attempt >= max_retries:
                    LOGGER.warning("Skipping macro series %s due to download issue: %s", series_name, exc)
                    break
                LOGGER.warning(
                    "Download attempt failed for macro series %s (%s). Retrying in %.1f seconds.",
                    series_name,
                    exc,
                    retry_delay_seconds,
                )
                time.sleep(retry_delay_seconds)
        if series is None:
            if last_error is not None:
                LOGGER.debug("Macro series %s final failure: %s", series_name, last_error)
            continue
        series.name = series_name
        result[series_name] = series

    if not result:
        raise ValueError("No macro series were downloaded successfully.")

    macro = pd.concat(result.values(), axis=1).sort_index()
    if {"us10y_yield", "us2y_yield"}.issubset(set(macro.columns)):
        macro["us_2s10s_slope"] = macro["us10y_yield"] - macro["us2y_yield"]
    if {"hyg", "lqd"}.issubset(set(macro.columns)):
        macro["hyg_lqd_ratio"] = macro["hyg"] / macro["lqd"]
    macro.index.name = "date"
    return macro


def save_macro_series_per_symbol(
    macro_series: pd.DataFrame,
    output_dir: str | Path = "data/macro",
) -> dict[str, Path]:
    """Persist each macro series as an individual CSV for auditability."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for column in macro_series.columns:
        path = output_path / f"{column}.csv"
        macro_series[[column]].to_csv(path, index=True)
        saved[column] = path
    return saved
