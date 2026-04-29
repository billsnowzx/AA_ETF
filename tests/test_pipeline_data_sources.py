import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.data_sources import hash_ticker_csvs, load_price_frames_from_csv


def test_load_price_frames_from_csv_loads_per_ticker_files() -> None:
    root = Path("data/cache") / f"data_sources_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "adj_close": [100.5],
                "volume": [1000],
                "ticker": ["VTI"],
            }
        )
        frame.to_csv(root / "VTI.csv", index=False)

        frames = load_price_frames_from_csv(["VTI"], root)

        assert list(frames) == ["VTI"]
        assert frames["VTI"].loc[0, "adj_close"] == 100.5
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_price_frames_from_csv_fails_loudly_for_missing_files() -> None:
    root = Path("data/cache") / f"data_sources_missing_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(FileNotFoundError, match="Missing local raw price files"):
            load_price_frames_from_csv(["VTI"], root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_hash_ticker_csvs_is_deterministic() -> None:
    root = Path("data/cache") / f"data_sources_hash_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / "VTI.csv").write_text("ticker,adj_close\nVTI,100\n", encoding="utf-8")

        first = hash_ticker_csvs(["VTI"], root)
        second = hash_ticker_csvs(["VTI"], root)

        assert first == second
        assert str(root / "VTI.csv") in first
        assert len(first[str(root / "VTI.csv")]) == 64
    finally:
        shutil.rmtree(root, ignore_errors=True)
