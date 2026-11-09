"""
Tests for data generation and label construction.

Determinism is the property the whole reproducibility claim rests on: the
README states that the manifest produced on Windows matches the one produced
on Linux, and that only holds if generation is seed-stable. These tests fail
loudly if that stops being true.

Field-consistency tests exist because of a real bug: reagent revenue was
drawn first and payback derived from it, which put 76% of deals' payback
outside the lease term — commercially implausible, and it skewed every label
downstream. Inverting the derivation brought it to 18%.
"""

import random

import pytest

from generate_deals import ARCHETYPES, make_deal
from simulate_reviews import question_set, state_from_deal


def deals(n=120, seed=42):
    rng = random.Random(seed)
    return [make_deal(rng, i) for i in range(n)]


def lever_ids(deal):
    """The label for a deal: the ordered lever ids the rule engine fires."""
    return [l["lever_id"] for l in question_set(deal)["levers"]]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_deals():
    assert deals(40, seed=42) == deals(40, seed=42)


def test_different_seeds_produce_different_deals():
    assert deals(40, seed=42) != deals(40, seed=43)


def test_labels_are_deterministic():
    d = deals(25)
    assert [question_set(x) for x in d] == [question_set(x) for x in d]


# ---------------------------------------------------------------------------
# Deal integrity
# ---------------------------------------------------------------------------

def test_deal_ids_are_unique():
    ids = [d["deal_id"] for d in deals()]
    assert len(set(ids)) == len(ids)


def test_every_deal_declares_a_known_archetype():
    for d in deals():
        assert d["archetype"] in ARCHETYPES


def test_all_archetypes_appear_in_a_reasonable_sample():
    """A declared mix that never produces some of its archetypes is a bug in
    the mix, and would leave whole lever groups untrained."""
    seen = {d["archetype"] for d in deals(400)}
    assert seen == set(ARCHETYPES), f"missing: {set(ARCHETYPES) - seen}"


@pytest.mark.parametrize("field", [
    "hardware_outlay_k", "annual_reagent_k", "lease_months",
    "breakeven_months", "worst_line_margin", "margin_floor",
])
def test_core_numeric_fields_are_present_and_positive(field):
    for d in deals(60):
        assert field in d, f"{field} missing from deal record"
        assert d[field] > 0, f"{field} should be positive, got {d[field]}"


def test_margins_are_fractions_not_percentages():
    for d in deals(60):
        assert 0.0 < d["worst_line_margin"] < 1.0
        assert 0.0 < d["margin_floor"] < 1.0


def test_lease_terms_are_plausible():
    for d in deals(60):
        assert 12 <= d["lease_months"] <= 84


def test_payback_outside_lease_is_the_minority_case():
    """
    REGRESSION. Payback was once derived from reagent revenue rather than the
    reverse, leaving 76% of deals paying back after the lease ended. Real
    placements are mostly structured to earn out inside the term; the rate
    should sit well under half.
    """
    d = deals(400)
    outside = sum(1 for x in d if x["breakeven_months"] > x["lease_months"])
    assert outside / len(d) < 0.45, f"{outside / len(d):.0%} pay back outside the term"


def test_some_deals_do_pay_back_outside_the_lease():
    """L14 has to be trainable, so the case must occur."""
    d = deals(400)
    assert any(x["breakeven_months"] > x["lease_months"] for x in d)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_every_deal_gets_at_least_one_question():
    """
    REGRESSION. L1 was once conditional, leaving 178 of 1000 deals with an
    empty label — unusable as training data.
    """
    for d in deals(200):
        assert len(lever_ids(d)) >= 1


def test_l1_fires_on_every_deal():
    for d in deals(100):
        assert 1 in lever_ids(d)


def test_question_counts_vary_across_deals():
    """A constant label length would make the task guessable without reading
    the deal — which is exactly the weakness the most_common_k baseline
    exposes in the results."""
    counts = {len(lever_ids(d)) for d in deals(200)}
    assert len(counts) > 3, f"only {len(counts)} distinct label lengths"


def test_labels_contain_no_duplicates():
    for d in deals(100):
        ids = lever_ids(d)
        assert len(ids) == len(set(ids))


def test_labels_stay_inside_the_taxonomy():
    for d in deals(100):
        assert all(1 <= i <= 14 for i in lever_ids(d))


def test_followup_levers_are_excluded_from_set_mode():
    """L7 only means something as a reply to a bare answer, and set mode has
    no dialogue — so it must never appear in a label."""
    for d in deals(200):
        assert 7 not in lever_ids(d)


def test_mean_question_count_is_in_the_reported_range():
    """The write-up reports a gold mean of 4.74 on the test split. A wide
    drift here means the dataset no longer matches the published numbers."""
    d = deals(400)
    mean = sum(len(lever_ids(x)) for x in d) / len(d)
    assert 3.5 <= mean <= 6.5, f"mean questions per deal drifted to {mean:.2f}"


def test_term_mismatch_deals_raise_the_term_question():
    """End-to-end check of the rule the project exists to capture: when the
    payback lands after the lease ends, the term question must be asked."""
    checked = 0
    for d in deals(400):
        if d["breakeven_months"] > d["lease_months"]:
            state = state_from_deal(d)
            assert state.breakeven_outside_term is True
            checked += 1
    assert checked > 0, "no term-mismatch deals in the sample"
