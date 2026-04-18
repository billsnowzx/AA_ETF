"""Benchmark configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.portfolio.weights import normalize_weights
from src.utils.config import load_yaml_file


def load_benchmarks(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load benchmark definitions from YAML."""
    config = load_yaml_file(config_path)
    benchmarks = config.get("benchmarks")
    if not benchmarks:
        raise ValueError(f"No benchmarks found in '{config_path}'.")
    return benchmarks


def load_benchmark(
    config_path: str | Path,
    benchmark_name: str,
) -> pd.Series:
    """Load and normalize one benchmark weight vector."""
    benchmarks = load_benchmarks(config_path)
    if benchmark_name not in benchmarks:
        raise ValueError(f"Unknown benchmark '{benchmark_name}'.")

    weights = benchmarks[benchmark_name].get("weights", {})
    return normalize_weights(weights)
