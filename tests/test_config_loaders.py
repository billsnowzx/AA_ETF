import math

import pytest

from run_pipeline import load_enabled_tickers
from src.portfolio.benchmarks import load_benchmark, load_benchmarks
from src.universe.universe_builder import load_asset_mapping, load_enabled_universe
from src.utils.config import load_yaml_file


def test_load_yaml_file_reads_mapping() -> None:
    config = load_yaml_file("config/etf_universe.yaml")

    assert config["universe_name"] == "phase_1_core"
    assert "tickers" in config


def test_load_enabled_tickers_matches_config() -> None:
    tickers = load_enabled_tickers("config/etf_universe.yaml")

    assert tickers == ["VTI", "VEA", "IEMG", "AGG", "BNDX", "GLD", "VNQ"]


def test_load_enabled_universe_returns_metadata_frame() -> None:
    universe = load_enabled_universe("config/etf_universe.yaml")

    assert list(universe.columns) == ["asset_class", "description"]
    assert universe.loc["VTI", "asset_class"] == "us_equity"
    assert "VNQ" in universe.index


def test_load_asset_mapping_reads_phase_one_assets() -> None:
    mapping = load_asset_mapping("config/asset_mapping.yaml")

    assert mapping["gold"]["tickers"] == ["GLD"]
    assert mapping["us_reit"]["label"] == "US REIT"
    assert mapping["international_aggregate_bond"]["tickers"] == ["BNDX"]


def test_load_benchmarks_reads_defined_benchmarks() -> None:
    benchmarks = load_benchmarks("config/benchmark_config.yaml")

    assert set(benchmarks) == {"benchmark_a", "benchmark_b"}


def test_load_benchmark_normalizes_weights() -> None:
    weights = load_benchmark("config/benchmark_config.yaml", "benchmark_a")

    assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-9)
    assert math.isclose(float(weights["VTI"]), 0.6, rel_tol=1e-9)
    assert math.isclose(float(weights["AGG"]), 0.4, rel_tol=1e-9)


def test_load_benchmark_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("config/benchmark_config.yaml", "missing_benchmark")
