"""Data-source helpers for reproducible pipeline runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_price_frames_from_csv(
    tickers: Iterable[str],
    input_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load per-ticker raw price CSV files from a local directory."""
    input_path = Path(input_dir)
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for ticker in tickers:
        csv_path = input_path / f"{ticker}.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        frames[ticker] = pd.read_csv(csv_path)

    if missing:
        raise FileNotFoundError(
            "Missing local raw price files for offline run: " + ", ".join(missing)
        )
    return frames


def hash_files(paths: Iterable[str | Path]) -> dict[str, str]:
    """Return SHA256 hashes for existing files keyed by path string."""
    hashes: dict[str, str] = {}
    for path_like in paths:
        path = Path(path_like)
        if not path.exists() or not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def hash_ticker_csvs(tickers: Iterable[str], input_dir: str | Path) -> dict[str, str]:
    """Hash expected per-ticker CSV files in a local data directory."""
    input_path = Path(input_dir)
    return hash_files(input_path / f"{ticker}.csv" for ticker in tickers)
