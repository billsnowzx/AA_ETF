"""Reusable cleaning utilities for ETF price data."""

from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def ensure_datetime_index(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Return a copy of the frame indexed by a normalized datetime index."""
    cleaned = frame.copy()

    if date_column in cleaned.columns:
        cleaned[date_column] = pd.to_datetime(cleaned[date_column])
        cleaned = cleaned.set_index(date_column)
    else:
        cleaned.index = pd.to_datetime(cleaned.index)

    cleaned.index.name = "date"
    return cleaned


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """Raise an error if the frame does not include all required columns."""
    required = required_columns or REQUIRED_COLUMNS
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def compute_dollar_volume(
    frame: pd.DataFrame,
    price_column: str = "adj_close",
    volume_column: str = "volume",
) -> pd.Series:
    """Compute daily dollar volume without fabricating missing volume data."""
    return frame[price_column] * frame[volume_column]


def clean_price_frame(
    frame: pd.DataFrame,
    ticker: str | None = None,
    drop_missing_adj_close: bool = True,
) -> pd.DataFrame:
    """Clean and validate a single ETF price frame."""
    cleaned = ensure_datetime_index(frame)
    validate_required_columns(cleaned)

    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].copy()

    if "ticker" not in cleaned.columns and ticker is not None:
        cleaned["ticker"] = ticker

    if drop_missing_adj_close:
        before_rows = len(cleaned)
        cleaned = cleaned.dropna(subset=["adj_close"]).copy()
        dropped_rows = before_rows - len(cleaned)
        if dropped_rows > 0:
            LOGGER.info(
                "Dropped %s rows with missing adjusted close for %s.",
                dropped_rows,
                ticker or "unknown ticker",
            )

    cleaned["volume"] = pd.to_numeric(cleaned["volume"], errors="coerce")
    cleaned["dollar_volume"] = compute_dollar_volume(cleaned)

    return cleaned


def batch_clean_price_frames(
    frames: Mapping[str, pd.DataFrame],
    drop_missing_adj_close: bool = True,
) -> dict[str, pd.DataFrame]:
    """Clean a mapping of ticker-to-price-frame objects."""
    return {
        ticker: clean_price_frame(
            frame=frame,
            ticker=ticker,
            drop_missing_adj_close=drop_missing_adj_close,
        )
        for ticker, frame in frames.items()
    }


def combine_price_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple cleaned price frames into a single long DataFrame."""
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames.values(), axis=0)
    combined = combined.sort_index()
    if "ticker" in combined.columns:
        combined = combined.sort_values(by=["ticker"], kind="stable")
    return combined
