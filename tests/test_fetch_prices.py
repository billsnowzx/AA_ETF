import shutil
import uuid
from pathlib import Path

import pandas as pd

from src.data.fetch_prices import configure_yfinance_tz_cache, fetch_price_history, fetch_prices


def _sample_download_frame(multi_index: bool = False) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=2, freq="B")
    data = {
        "Open": [100.0, 101.0],
        "High": [101.0, 102.0],
        "Low": [99.0, 100.0],
        "Close": [100.5, 101.5],
        "Adj Close": [100.5, 101.25],
        "Volume": [1_000_000, 1_200_000],
    }
    frame = pd.DataFrame(data, index=index)

    if multi_index:
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["VTI"]])

    return frame


def test_fetch_price_history_standardizes_columns(monkeypatch) -> None:
    def fake_download(**kwargs) -> pd.DataFrame:
        assert kwargs["tickers"] == "VTI"
        return _sample_download_frame()

    monkeypatch.setattr("src.data.fetch_prices.yf.download", fake_download)

    frame = fetch_price_history("VTI", start="2024-01-01", end="2024-01-31")

    assert list(frame.columns) == ["open", "high", "low", "close", "adj_close", "volume", "ticker"]
    assert frame.index.name == "date"
    assert frame["ticker"].tolist() == ["VTI", "VTI"]


def test_fetch_price_history_handles_yfinance_multiindex(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data.fetch_prices.yf.download",
        lambda **_: _sample_download_frame(multi_index=True),
    )

    frame = fetch_price_history("VTI")

    assert frame["adj_close"].tolist() == [100.5, 101.25]


def test_fetch_prices_saves_per_ticker_csv(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data.fetch_prices.yf.download",
        lambda **_: _sample_download_frame(),
    )

    output_dir = Path("data/cache") / f"test_fetch_prices_{uuid.uuid4().hex}"

    try:
        frames = fetch_prices(["VTI", "VEA"], output_dir=output_dir, save_raw=True)

        assert set(frames) == {"VTI", "VEA"}
        assert (output_dir / "VTI.csv").exists()
        assert (output_dir / "VEA.csv").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_fetch_price_history_retries_after_empty_response(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_download(**kwargs) -> pd.DataFrame:
        calls["count"] += 1
        if calls["count"] == 1:
            return pd.DataFrame()
        return _sample_download_frame()

    monkeypatch.setattr("src.data.fetch_prices.yf.download", fake_download)

    frame = fetch_price_history("VTI", max_retries=2, retry_delay_seconds=0.0)

    assert calls["count"] == 2
    assert not frame.empty
    assert frame["ticker"].iloc[0] == "VTI"


def test_configure_yfinance_tz_cache_uses_workspace_path(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_set_tz_cache_location(path: str) -> None:
        captured["path"] = path

    monkeypatch.setattr("src.data.fetch_prices.yf.set_tz_cache_location", fake_set_tz_cache_location)

    output_path = Path("data/cache") / f"test_yf_tz_{uuid.uuid4().hex}"
    try:
        cache_path = configure_yfinance_tz_cache(output_path)
        assert cache_path.exists()
        assert captured["path"] == str(output_path)
    finally:
        shutil.rmtree(output_path, ignore_errors=True)
