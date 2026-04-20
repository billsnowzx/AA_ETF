import pandas as pd
import pytest

from src.data.clean_data import (
    batch_clean_price_frames,
    build_data_quality_summary,
    clean_price_frame,
    compute_dollar_volume,
)


def test_clean_price_frame_sorts_deduplicates_and_drops_missing_adj_close() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04"],
            "open": [102.0, 100.0, 101.0, 103.0],
            "high": [103.0, 101.0, 102.0, 104.0],
            "low": [101.0, 99.0, 100.0, 102.0],
            "close": [102.5, 100.5, 101.5, 103.5],
            "adj_close": [102.5, 100.5, 101.5, None],
            "volume": [1200, 1000, 1100, 1300],
        }
    )

    cleaned = clean_price_frame(frame, ticker="VTI")

    assert cleaned.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert cleaned.loc[pd.Timestamp("2024-01-02"), "open"] == 101.0
    assert cleaned["ticker"].tolist() == ["VTI", "VTI"]
    assert cleaned["dollar_volume"].tolist() == [111650.0, 123000.0]


def test_compute_dollar_volume_preserves_missing_volume() -> None:
    frame = pd.DataFrame(
        {
            "adj_close": [100.0, 101.0],
            "volume": [0.0, None],
        }
    )

    result = compute_dollar_volume(frame)

    assert result.iloc[0] == 0.0
    assert pd.isna(result.iloc[1])


def test_batch_clean_price_frames_returns_mapping() -> None:
    raw_frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "adj_close": [100.5],
            "volume": [1000],
        }
    )

    cleaned_frames = batch_clean_price_frames({"VTI": raw_frame})

    assert list(cleaned_frames) == ["VTI"]
    assert cleaned_frames["VTI"].index.name == "date"


def test_clean_price_frame_requires_standard_columns() -> None:
    frame = pd.DataFrame({"date": ["2024-01-02"], "adj_close": [100.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        clean_price_frame(frame)


def test_build_data_quality_summary_reports_missing_and_zero_volume() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "adj_close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 0.0, None],
            "dollar_volume": [100000.0, 0.0, None],
        }
    )

    summary = build_data_quality_summary({"VTI": frame})

    assert summary.loc["VTI", "start_date"] == "2024-01-02"
    assert summary.loc["VTI", "end_date"] == "2024-01-04"
    assert summary.loc["VTI", "observations"] == 3
    assert summary.loc["VTI", "missing_adj_close"] == 0
    assert summary.loc["VTI", "missing_volume"] == 1
    assert summary.loc["VTI", "zero_volume"] == 1
    assert summary.loc["VTI", "missing_dollar_volume"] == 1
    assert bool(summary.loc["VTI", "has_duplicate_dates"]) is False
