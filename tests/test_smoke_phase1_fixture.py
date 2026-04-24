import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

from run_pipeline import main
from scripts.check_required_outputs import validate_required_outputs
from src.dashboard.app import write_dashboard_html


def _build_fixture_frame(index: pd.DatetimeIndex, start_price: float) -> pd.DataFrame:
    prices = pd.Series(start_price + pd.RangeIndex(len(index)) * 0.25, index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "adj_close": prices,
            "volume": 1_500_000,
        },
        index=index,
    )
    frame.index.name = "date"
    return frame


def test_phase1_smoke_pipeline_with_local_fixture_data(monkeypatch) -> None:
    run_root = Path("data/cache") / f"smoke_phase1_{uuid.uuid4().hex}"
    raw_dir = run_root / "raw"
    processed_dir = run_root / "processed"
    table_dir = run_root / "tables"
    figure_dir = run_root / "figures"
    report_dir = run_root / "reports"

    tickers = ["VTI", "VEA", "IEMG", "AGG", "BNDX", "GLD", "VNQ"]
    index = pd.date_range("2024-01-02", periods=160, freq="B")
    fixture_frames = {
        ticker: _build_fixture_frame(index=index, start_price=90.0 + (i * 7.5))
        for i, ticker in enumerate(tickers)
    }

    monkeypatch.setattr("run_pipeline.fetch_prices", lambda **_kwargs: fixture_frames)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(table_dir),
            "--figure-dir",
            str(figure_dir),
            "--report-dir",
            str(report_dir),
            "--fail-on-missing-outputs",
            "--fail-on-empty-outputs",
            "--seed",
            "7",
            "--as-of-date",
            "2024-12-31",
        ],
    )

    try:
        main()
        write_dashboard_html(
            output_path=report_dir / "dashboard.html",
            output_dir=table_dir,
            figure_dir=figure_dir,
            report_dir=report_dir,
        )
        validate_required_outputs(
            table_dir=table_dir,
            figure_dir=figure_dir,
            report_dir=report_dir,
        )
        assert (table_dir / "risk_limit_breach_summary.csv").exists()
    finally:
        shutil.rmtree(run_root, ignore_errors=True)
