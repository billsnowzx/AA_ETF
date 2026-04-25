import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.data.fetch_etf_metadata import (
    fetch_etf_metadata,
    save_etf_metadata_snapshots,
    save_etf_metadata_summary,
)


class _FakeTicker:
    def __init__(self, ticker: str):
        self.ticker = ticker

    @property
    def info(self) -> dict[str, object]:
        return {
            "longName": f"{self.ticker} Name",
            "quoteType": "ETF",
            "category": "Large Blend",
            "expenseRatio": 0.0003,
            "totalAssets": 1_000_000_000,
            "marketCap": 900_000_000,
            "beta": 1.02,
        }


def test_fetch_etf_metadata_returns_summary_table(monkeypatch) -> None:
    monkeypatch.setattr("src.data.fetch_etf_metadata.yf.Ticker", lambda ticker: _FakeTicker(ticker))

    summary = fetch_etf_metadata(["VTI", "VEA"])

    assert summary.index.tolist() == ["VTI", "VEA"]
    assert "long_name" in summary.columns
    assert "expense_ratio" in summary.columns
    assert bool(summary.loc["VTI", "metadata_available"]) is True


def test_save_etf_metadata_outputs_write_json_and_csv() -> None:
    output_dir = Path("data/cache") / f"test_etf_metadata_{uuid.uuid4().hex}"
    summary = pd.DataFrame(
        {
            "as_of_utc": ["2026-04-25T00:00:00Z"],
            "metadata_available": [True],
            "long_name": ["VTI Name"],
            "quote_type": ["ETF"],
        },
        index=pd.Index(["VTI"], name="ticker"),
    )
    try:
        snapshots = save_etf_metadata_snapshots(summary, output_dir=output_dir)
        csv_path = save_etf_metadata_summary(summary, output_dir / "etf_metadata_summary.csv")

        assert csv_path.exists()
        assert (output_dir / "VTI.json").exists()
        assert "VTI" in snapshots
        payload = json.loads((output_dir / "VTI.json").read_text(encoding="utf-8"))
        assert payload["ticker"] == "VTI"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
