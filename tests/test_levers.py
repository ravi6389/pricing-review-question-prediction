"""
Tests for the rule engine.

These matter more than the model tests. The rules ARE the ground truth: if a
trigger is wrong, every training label downstream is wrong and no amount of
fine-tuning recovers it. Three of the bugs found during development were
trigger bugs, and each is pinned by a test below so it cannot return.
"""

import pytest

from levers import (
    LEVERS,
    LEVERS_BY_ID,
    MAX_UNEXPLAINED_WORDS,
    DealState,
    _is_bare,
    next_lever,
)


# ---------------------------------------------------------------------------
# Taxonomy integrity
# ---------------------------------------------------------------------------

def test_fourteen_levers_with_unique_ids():
    assert len(LEVERS) == 14
    ids = [l.id for l in LEVERS]
    assert ids == list(range(1, 15)), "lever ids must be 1..14 in order"
    assert len(set(ids)) == 14


def test_every_lever_has_name_intent_and_trigger():
    for l in LEVERS:
        assert l.name and not l.name[0].isupper(), f"{l.id}: name should be a slug"
        assert l.intent.endswith("."), f"{l.id}: intent should read as a sentence"
        assert callable(l.trigger)


def test_only_lever_7_is_a_followup():
    followups = [l.id for l in LEVERS if l.is_followup]
    assert followups == [7], "L7 drills into a bare answer; nothing else should"


# ---------------------------------------------------------------------------
# Individual triggers
# ---------------------------------------------------------------------------

def test_l1_always_fires(healthy_deal, squeezed_deal):
    """
    REGRESSION. L1 was once gated on price being below threshold, which left
    178 of 1000 reviews with zero questions. A pricing manager opens with this
    on every deal; on a healthy one the answer is simply short.
    """
    assert LEVERS_BY_ID[1].trigger(healthy_deal) is True
    assert LEVERS_BY_ID[1].trigger(squeezed_deal) is True


def test_l2_needs_l1_closed_and_margin_below_floor(squeezed_deal):
    assert LEVERS_BY_ID[2].trigger(squeezed_deal) is False, "L1 not closed yet"
    squeezed_deal.answered[1] = "no room, list is already thin"
    assert LEVERS_BY_ID[2].trigger(squeezed_deal) is True


def test_l2_does_not_fire_when_margin_is_healthy(healthy_deal):
    healthy_deal.answered[1] = "some room on the consumables"
    assert LEVERS_BY_ID[2].trigger(healthy_deal) is False


def test_l14_fires_only_when_payback_outlives_the_lease(squeezed_deal):
    squeezed_deal.answered.update({1: "no", 2: "no", 3: "no", 4: "no"})
    assert squeezed_deal.breakeven_outside_term is True  # 44 > 36
    assert LEVERS_BY_ID[14].trigger(squeezed_deal) is True

    squeezed_deal.lease_months = 60
    assert squeezed_deal.breakeven_outside_term is False
    assert LEVERS_BY_ID[14].trigger(squeezed_deal) is False


def test_l12_fires_only_while_bundling_is_unknown(squeezed_deal):
    assert squeezed_deal.competitor_bundling is None
    assert LEVERS_BY_ID[12].trigger(squeezed_deal) is True

    squeezed_deal.competitor_bundling = False
    assert LEVERS_BY_ID[12].trigger(squeezed_deal) is False


def test_l13_requires_confirmed_bundling(squeezed_deal):
    assert LEVERS_BY_ID[13].trigger(squeezed_deal) is False
    squeezed_deal.competitor_bundling = True
    assert LEVERS_BY_ID[13].trigger(squeezed_deal) is True


def test_competitor_levers_silent_without_a_competitor(healthy_deal):
    healthy_deal.answered.update({1: "a", 2: "b", 3: "c", 4: "d"})
    assert LEVERS_BY_ID[5].trigger(healthy_deal) is False
    assert LEVERS_BY_ID[12].trigger(healthy_deal) is False


# ---------------------------------------------------------------------------
# The `closed` bug
# ---------------------------------------------------------------------------

def test_answered_levers_count_as_closed(squeezed_deal):
    """
    REGRESSION, the costliest bug found. Triggers once tested `exhausted`
    only, so a lever the rep actually ANSWERED never unlocked the next rung
    and reviews died after ~2 questions. Mean length 2.1 -> 4.8 once fixed.
    """
    squeezed_deal.answered[1] = "no headroom, the list price is already keen"
    assert 1 in squeezed_deal.closed
    assert LEVERS_BY_ID[2].trigger(squeezed_deal) is True


def test_refused_levers_also_count_as_closed(squeezed_deal):
    squeezed_deal.exhausted.add(1)
    assert 1 in squeezed_deal.closed
    assert LEVERS_BY_ID[2].trigger(squeezed_deal) is True


def test_margin_levers_exhausted_needs_all_four(squeezed_deal):
    squeezed_deal.answered.update({1: "a", 2: "b", 3: "c"})
    assert squeezed_deal.margin_levers_exhausted is False
    squeezed_deal.answered[4] = "d"
    assert squeezed_deal.margin_levers_exhausted is True


# ---------------------------------------------------------------------------
# Bare-answer detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer", ["no", "No", " none ", "move on", "n/a", "no room"])
def test_flat_refusals_are_bare(answer):
    assert _is_bare(answer) is True


@pytest.mark.parametrize("answer", ["pricing pressure", "low base price"])
def test_short_unexplained_assertions_are_bare(answer):
    """
    REGRESSION. An earlier draft treated only flat refusals as bare. The real
    transcript shows the manager also drilling into short assertions that name
    a cause without giving a reason.
    """
    assert _is_bare(answer) is True


def test_an_explained_answer_is_not_bare():
    answer = ("There is no headroom because the tender capped unit price at "
              "the regional average two quarters ago.")
    assert _is_bare(answer) is False


def test_none_is_not_bare():
    assert _is_bare(None) is False


def test_bare_threshold_is_honoured():
    assert _is_bare(" ".join(["word"] * MAX_UNEXPLAINED_WORDS)) is True
    assert _is_bare(" ".join(["word"] * (MAX_UNEXPLAINED_WORDS + 1))) is False


# ---------------------------------------------------------------------------
# Ladder traversal
# ---------------------------------------------------------------------------

def test_review_opens_on_l1(squeezed_deal):
    assert next_lever(squeezed_deal).id == 1


def test_followups_jump_the_queue(squeezed_deal):
    """A bare answer to L6 must be drilled into before the ladder advances."""
    squeezed_deal.answered.update({1: "a", 2: "b", 3: "c", 4: "d", 5: "e"})
    squeezed_deal.answered[6] = "pricing pressure"
    nxt = next_lever(squeezed_deal)
    assert nxt.id == 7 and nxt.is_followup


def test_a_closed_lever_is_never_re_asked(squeezed_deal):
    seen = set()
    for _ in range(30):
        nxt = next_lever(squeezed_deal)
        if nxt is None:
            break
        assert nxt.id not in seen, f"L{nxt.id} asked twice"
        seen.add(nxt.id)
        squeezed_deal.answered[nxt.id] = "no"


def test_refusals_advance_rather_than_terminate(squeezed_deal):
    """
    REGRESSION. An earlier draft ended the review after 3 consecutive
    refusals. The source transcript shows ~10 straight refusals; a refusal
    moves down the ladder, it does not end the conversation.
    """
    asked = 0
    for _ in range(30):
        nxt = next_lever(squeezed_deal)
        if nxt is None:
            break
        squeezed_deal.exhausted.add(nxt.id)
        squeezed_deal.consecutive_refusals += 1
        asked += 1
    assert asked > 3, f"review died after {asked} refusals"


def test_ladder_terminates(squeezed_deal):
    for _ in range(50):
        nxt = next_lever(squeezed_deal)
        if nxt is None:
            return
        squeezed_deal.answered[nxt.id] = "no"
    pytest.fail("ladder did not terminate within 50 turns")


def test_healthy_deal_asks_fewer_questions_than_a_squeezed_one(
    healthy_deal, squeezed_deal
):
    """The count must vary with the deal — a fixed-length output is the
    failure mode the untrained models show."""
    def walk(state):
        n = 0
        while (nxt := next_lever(state)) is not None and n < 30:
            state.answered[nxt.id] = "noted"
            n += 1
        return n

    assert walk(healthy_deal) < walk(squeezed_deal)
