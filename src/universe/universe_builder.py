"""Config-driven helpers for building the Phase 1 ETF universe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import load_yaml_file


def load_universe_config(config_path: str | Path) -> dict:
    """Load the ETF universe YAML configuration."""
    config = load_yaml_file(config_path)
    if "tickers" not in config:
        raise ValueError(f"Universe config '{config_path}' must define a 'tickers' mapping.")
    return config


def load_enabled_universe(config_path: str | Path) -> pd.DataFrame:
    """Return enabled ETF metadata as a DataFrame indexed by ticker."""
    config = load_universe_config(config_path)
    tickers = config["tickers"]

    rows = []
    for ticker, metadata in tickers.items():
        if metadata.get("enabled", False):
            rows.append(
                {
                    "ticker": ticker,
                    "asset_class": metadata.get("asset_class"),
                    "description": metadata.get("description"),
                }
            )

    if not rows:
        raise ValueError(f"No enabled tickers found in '{config_path}'.")

    return pd.DataFrame(rows).set_index("ticker")


def load_asset_mapping(config_path: str | Path) -> dict[str, dict]:
    """Load the asset mapping YAML configuration."""
    config = load_yaml_file(config_path)
    mapping = config.get("asset_mapping")
    if not mapping:
        raise ValueError(f"Asset mapping config '{config_path}' must define 'asset_mapping'.")
    return mapping


def build_universe_summary(
    universe_config_path: str | Path,
    liquidity_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a universe summary table by combining config metadata and liquidity metrics."""
    universe = load_enabled_universe(universe_config_path)

    if liquidity_summary is None:
        return universe

    summary = universe.join(liquidity_summary, how="left")
    return summary
