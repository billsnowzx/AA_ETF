import shutil
import uuid
from pathlib import Path

import pandas as pd

from run_pipeline import (
    build_argument_parser,
    save_processed_frames,
    write_data_quality_outputs,
    write_liquidity_outputs,
)


def test_argument_parser_exposes_rolling_window_default_and_override() -> None:
    parser = build_argument_parser()

    default_args = parser.parse_args(["--start", "2024-01-01"])
    override_args = parser.parse_args(["--start", "2024-01-01", "--rolling-window", "21"])

    assert default_args.rolling_window == 63
    assert default_args.risk_limits_config == "config/risk_limits.yaml"
    assert override_args.rolling_window == 21


def test_save_processed_frames_writes_per_ticker_csv() -> None:
    output_dir = Path("data/cache") / f"test_processed_frames_{uuid.uuid4().hex}"
    frame = pd.DataFrame(
        {"adj_close": [100.0], "volume": [1000], "dollar_volume": [100000.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    frame.index.name = "date"

    try:
        save_processed_frames({"VTI": frame}, output_dir)
        assert (output_dir / "VTI.csv").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_write_data_quality_outputs_writes_summary_file() -> None:
    output_dir = Path("data/cache") / f"test_data_quality_outputs_{uuid.uuid4().hex}"
    frame = pd.DataFrame(
        {"adj_close": [100.0], "volume": [1000.0], "dollar_volume": [100000.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    frame.index.name = "date"

    try:
        summary = write_data_quality_outputs({"VTI": frame}, output_dir)
        assert (output_dir / "data_quality_summary.csv").exists()
        assert summary.loc["VTI", "observations"] == 1
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_write_liquidity_outputs_writes_expected_files() -> None:
    output_dir = Path("data/cache") / f"test_liquidity_outputs_{uuid.uuid4().hex}"
    summary = pd.DataFrame(
        {"passes_liquidity_filter": [True]},
        index=pd.Index(["VTI"], name="ticker"),
    )
    etf_summary = pd.DataFrame(
        {"phase1_total_score": [40.0]},
        index=pd.Index(["VTI"], name="ticker"),
    )

    try:
        write_liquidity_outputs(["VTI"], summary, etf_summary, output_dir)
        assert (output_dir / "liquidity_summary.csv").exists()
        assert (output_dir / "etf_summary.csv").exists()
        assert (output_dir / "investable_universe.csv").read_text(encoding="utf-8") == "ticker\nVTI\n"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
