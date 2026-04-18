"""YAML configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"YAML file '{config_path}' must contain a mapping at the top level.")

    return config
