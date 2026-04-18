import pytest

from src.utils.dates import parse_date, validate_date_range


def test_parse_date_returns_timestamp() -> None:
    parsed = parse_date("2024-01-15")

    assert parsed.strftime("%Y-%m-%d") == "2024-01-15"


def test_validate_date_range_normalizes_strings() -> None:
    start, end = validate_date_range("2024-01-01", "2024-02-01")

    assert start == "2024-01-01"
    assert end == "2024-02-01"


def test_validate_date_range_rejects_equal_or_reversed_dates() -> None:
    with pytest.raises(ValueError, match="Start date must be earlier"):
        validate_date_range("2024-02-01", "2024-02-01")
