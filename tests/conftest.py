"""
Shared fixtures.

Every test here runs on CPU with no network and no API key: the rule engine
and the scorer are pure functions over plain data, which is the main reason
the pipeline was built with the rules separated from the model.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def healthy_deal():
    """A deal with nothing wrong: inside floor, pays back inside the term."""
    from levers import DealState
    return DealState(
        drp_vs_threshold=1.15,
        worst_line_margin=0.44,
        margin_floor=0.30,
        hardware_outlay_k=60.0,
        annual_reagent_k=180.0,
        lease_months=60,
        breakeven_months=22,
        competitor_present=False,
    )


@pytest.fixture
def squeezed_deal():
    """Margin below floor, customer pushing for a cut, competitor in play."""
    from levers import DealState
    return DealState(
        drp_vs_threshold=0.82,
        worst_line_margin=0.18,
        margin_floor=0.30,
        hardware_outlay_k=180.0,
        annual_reagent_k=117.0,
        lease_months=36,
        breakeven_months=44,
        competitor_present=True,
        customer_demands_reduction=True,
        alt_config_available=True,
        installed_base=True,
    )
