import math

import pytest

from src.portfolio.transaction_cost import (
    load_one_way_transaction_cost_bps,
    transaction_cost_drag,
    turnover_traded_weight,
)


def test_turnover_traded_weight_for_initial_investment_is_full_weight() -> None:
    turnover = turnover_traded_weight({"VTI": 0.6, "AGG": 0.4})

    assert math.isclose(turnover, 1.0, rel_tol=1e-9)


def test_turnover_traded_weight_for_rebalance_uses_gross_weight_change() -> None:
    turnover = turnover_traded_weight(
        {"VTI": 0.6, "AGG": 0.4},
        current_weights={"VTI": 0.5, "AGG": 0.5},
    )

    assert math.isclose(turnover, 0.2, rel_tol=1e-9)


def test_transaction_cost_drag_uses_one_way_bps() -> None:
    cost = transaction_cost_drag(
        {"VTI": 0.6, "AGG": 0.4},
        current_weights={"VTI": 0.5, "AGG": 0.5},
        one_way_bps=5.0,
    )

    assert math.isclose(cost, 0.0001, rel_tol=1e-9)


def test_load_one_way_transaction_cost_bps_reads_config() -> None:
    bps = load_one_way_transaction_cost_bps("config/rebalance_rules.yaml")

    assert math.isclose(bps, 5.0, rel_tol=1e-9)


def test_turnover_rejects_negative_current_weight_sum() -> None:
    with pytest.raises(ValueError, match="must not sum to a negative value"):
        turnover_traded_weight({"VTI": 1.0}, current_weights={"VTI": -1.0})
