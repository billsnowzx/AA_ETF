"""ETF metadata download utilities for auditable Phase 1 snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)

DEFAULT_INFO_FIELDS: dict[str, str] = {
    "long_name": "longName",
    "quote_type": "quoteType",
    "category": "category",
    "expense_ratio": "expenseRatio",
    "total_assets": "totalAssets",
    "market_cap": "marketCap",
    "beta": "beta",
}


def _coerce_jsonable(value: object) -> object:
    """Convert values from yfinance metadata into JSON-serializable objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def fetch_etf_metadata(
    tickers: Iterable[str],
    info_fields: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch ETF metadata from yfinance and return an auditable summary table."""
    field_map = info_fields.copy() if info_fields is not None else DEFAULT_INFO_FIELDS.copy()
    rows: list[dict[str, object]] = []
    as_of_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for ticker in tickers:
        row: dict[str, object] = {"ticker": ticker, "as_of_utc": as_of_utc, "metadata_available": False}
        for column in field_map.keys():
            row[column] = None

        try:
            info = yf.Ticker(ticker).info
            if not isinstance(info, dict):
                raise ValueError(f"Ticker '{ticker}' metadata response is not a mapping.")
            for column, source_field in field_map.items():
                row[column] = _coerce_jsonable(info.get(source_field))
            row["metadata_available"] = True
        except Exception as exc:  # pragma: no cover - network behavior, tested via monkeypatch
            LOGGER.warning("ETF metadata unavailable for %s: %s", ticker, exc)

        rows.append(row)

    summary = pd.DataFrame(rows).set_index("ticker")
    return summary


def save_etf_metadata_snapshots(
    metadata_summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist one JSON snapshot per ticker for manual auditability."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for ticker, row in metadata_summary.iterrows():
        path = output_path / f"{ticker}.json"
        payload = {"ticker": ticker, **{key: _coerce_jsonable(value) for key, value in row.to_dict().items()}}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        saved[str(ticker)] = path
    return saved


def save_etf_metadata_summary(
    metadata_summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Persist ETF metadata summary as a CSV table for dashboard/report consumption."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_summary.to_csv(path, index=True)
    return path

