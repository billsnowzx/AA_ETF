import shutil
import uuid
from pathlib import Path

import pytest

from scripts.check_required_outputs import (
    REQUIRED_FIGURES,
    REQUIRED_REPORTS,
    REQUIRED_TABLES,
    validate_required_outputs,
)


def _write_non_empty_files(base_dir: Path, filenames: list[str]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (base_dir / name).write_bytes(b"x")


def test_validate_required_outputs_accepts_complete_non_empty_outputs() -> None:
    root = Path("data/cache") / f"required_outputs_ok_{uuid.uuid4().hex}"
    table_dir = root / "tables"
    figure_dir = root / "figures"
    report_dir = root / "reports"
    try:
        _write_non_empty_files(table_dir, REQUIRED_TABLES)
        _write_non_empty_files(figure_dir, REQUIRED_FIGURES)
        _write_non_empty_files(report_dir, REQUIRED_REPORTS)
        validate_required_outputs(table_dir=table_dir, figure_dir=figure_dir, report_dir=report_dir)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validate_required_outputs_rejects_missing_or_empty_outputs() -> None:
    root = Path("data/cache") / f"required_outputs_bad_{uuid.uuid4().hex}"
    table_dir = root / "tables"
    figure_dir = root / "figures"
    report_dir = root / "reports"
    try:
        _write_non_empty_files(table_dir, REQUIRED_TABLES)
        _write_non_empty_files(figure_dir, REQUIRED_FIGURES)
        _write_non_empty_files(report_dir, REQUIRED_REPORTS)
        (table_dir / REQUIRED_TABLES[0]).write_bytes(b"")
        (figure_dir / REQUIRED_FIGURES[0]).unlink(missing_ok=True)
        with pytest.raises(ValueError, match="Required output validation failed"):
            validate_required_outputs(table_dir=table_dir, figure_dir=figure_dir, report_dir=report_dir)
    finally:
        shutil.rmtree(root, ignore_errors=True)
