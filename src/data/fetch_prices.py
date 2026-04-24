"""Price download utilities for ETF market data."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from src.utils.dates import validate_date_range

LOGGER = logging.getLogger(__name__)

REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
YFINANCE_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _standardize_download_frame(raw_frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standardize a single yfinance download frame."""
    if raw_frame.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")

    frame = raw_frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.rename(columns=YFINANCE_COLUMN_MAP)
    missing_columns = [column for column in REQUIRED_PRICE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Ticker '{ticker}' is missing required columns: {missing_columns}."
        )

    frame = frame[REQUIRED_PRICE_COLUMNS]
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame["ticker"] = ticker
    return frame


def configure_yfinance_tz_cache(cache_dir: str | Path = "data/cache/yfinance_tz") -> Path:
    """Configure yfinance timezone cache location to a writable workspace path."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_path))
    return cache_path


def fetch_price_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch and standardize daily OHLCV data for a single ticker."""
    start, end = validate_date_range(start, end)
    cache_path = configure_yfinance_tz_cache()
    LOGGER.debug("Configured yfinance timezone cache path: %s", cache_path)
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1.")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0.")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        LOGGER.info("Downloading price data for %s (attempt %s/%s)", ticker, attempt, max_retries)
        raw_frame = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            actions=False,
        )
        try:
            frame = _standardize_download_frame(raw_frame, ticker=ticker)
            break
        except ValueError as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            LOGGER.warning(
                "Download attempt failed for %s (%s). Retrying in %.1f seconds.",
                ticker,
                exc,
                retry_delay_seconds,
            )
            time.sleep(retry_delay_seconds)
    else:
        if last_error is not None:
            raise last_error
        raise ValueError(f"No price data returned for ticker '{ticker}'.")

    if frame["adj_close"].isna().any():
        missing_count = int(frame["adj_close"].isna().sum())
        LOGGER.warning("Ticker %s has %s rows with missing adjusted close.", ticker, missing_count)

    return frame


def save_price_frame(frame: pd.DataFrame, output_path: Path) -> None:
    """Persist a standardized price frame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=True)
    LOGGER.info("Saved raw price data to %s", output_path)


def fetch_prices(
    tickers: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    output_dir: str | Path = "data/raw",
    save_raw: bool = True,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Fetch and optionally persist price history for multiple tickers."""
    start, end = validate_date_range(start, end)
    output_path = Path(output_dir)
    frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        frame = fetch_price_history(
            ticker=ticker,
            start=start,
            end=end,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        frames[ticker] = frame
        if save_raw:
            save_price_frame(frame, output_path / f"{ticker}.csv")

    return frames
