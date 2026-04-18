import shutil
import uuid
from pathlib import Path

import math
import pytest

from src.portfolio.weights import load_portfolio_template, normalize_weights


def test_normalize_weights_rescales_to_one() -> None:
    normalized = normalize_weights({"VTI": 35.0, "AGG": 15.0})

    assert math.isclose(float(normalized.sum()), 1.0, rel_tol=1e-9)
    assert math.isclose(float(normalized["VTI"]), 0.7, rel_tol=1e-9)
    assert math.isclose(float(normalized["AGG"]), 0.3, rel_tol=1e-9)


def test_normalize_weights_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="Negative weights"):
        normalize_weights({"VTI": 0.8, "AGG": -0.2})


def test_load_portfolio_template_uses_default_template() -> None:
    weights = load_portfolio_template("config/portfolio_templates.yaml")

    assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-9)
    assert list(weights.index) == ["VTI", "VEA", "IEMG", "AGG", "BNDX", "GLD", "VNQ"]


def test_load_portfolio_template_from_custom_file() -> None:
    test_dir = Path("data/cache") / f"test_weights_{uuid.uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    config_path = test_dir / "portfolio_templates.yaml"
    config_path.write_text(
        "\n".join(
            [
                "default_template: sample",
                "templates:",
                "  sample:",
                "    weights:",
                "      AAA: 2.0",
                "      BBB: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    try:
        weights = load_portfolio_template(config_path)
        assert math.isclose(float(weights["AAA"]), 2.0 / 3.0, rel_tol=1e-9)
        assert math.isclose(float(weights["BBB"]), 1.0 / 3.0, rel_tol=1e-9)
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
