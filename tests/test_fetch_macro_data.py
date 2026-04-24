import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.data.fetch_macro_data import (
    fetch_macro_series,
    save_macro_series_per_symbol,
)


def _sample_download_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=2, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Adj Close": [100.5, 101.25],
            "Volume": [1_000_000, 1_200_000],
        },
        index=index,
    )


def test_fetch_macro_series_builds_wide_table_and_derived_fields(monkeypatch) -> None:
    monkeypatch.setattr("src.data.fetch_macro_data.yf.download", lambda **_: _sample_download_frame())

    macro = fetch_macro_series(
        start="2024-01-01",
        end="2024-01-31",
        series_map={
            "us10y_yield": "^TNX",
            "us2y_yield": "^IRX",
            "hyg": "HYG",
            "lqd": "LQD",
        },
    )

    assert macro.index.name == "date"
    assert "us10y_yield" in macro.columns
    assert "us2y_yield" in macro.columns
    assert "us_2s10s_slope" in macro.columns
    assert "hyg_lqd_ratio" in macro.columns
    assert float(macro.iloc[0]["us_2s10s_slope"]) == 0.0


def test_save_macro_series_per_symbol_writes_csvs() -> None:
    macro = pd.DataFrame(
        {
            "us10y_yield": [4.2, 4.3],
            "vix": [15.0, 16.0],
        },
        index=pd.date_range("2024-01-02", periods=2, freq="B"),
    )
    macro.index.name = "date"
    output_dir = Path("data/cache") / f"test_macro_series_{uuid.uuid4().hex}"

    try:
        saved = save_macro_series_per_symbol(macro, output_dir=output_dir)
        assert (output_dir / "us10y_yield.csv").exists()
        assert (output_dir / "vix.csv").exists()
        assert "us10y_yield" in saved
        assert "vix" in saved
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
