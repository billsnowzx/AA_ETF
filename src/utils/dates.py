"""Date parsing and validation helpers."""

from __future__ import annotations

import pandas as pd


def parse_date(date_value: str | None) -> pd.Timestamp | None:
    """Parse an optional date string into a normalized timestamp."""
    if date_value is None:
        return None
    return pd.Timestamp(date_value)


def validate_date_range(
    start: str | None,
    end: str | None,
) -> tuple[str | None, str | None]:
    """Validate a start/end date pair and return ISO date strings."""
    start_ts = parse_date(start)
    end_ts = parse_date(end)

    if start_ts is not None and end_ts is not None and start_ts >= end_ts:
        raise ValueError(
            f"Start date must be earlier than end date, received start={start_ts.date()} and end={end_ts.date()}."
        )

    normalized_start = start_ts.strftime("%Y-%m-%d") if start_ts is not None else None
    normalized_end = end_ts.strftime("%Y-%m-%d") if end_ts is not None else None
    return normalized_start, normalized_end
