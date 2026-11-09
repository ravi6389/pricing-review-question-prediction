"""
Tests for the evaluation code.

A scorer that is quietly wrong is worse than no scorer, because it produces
numbers nobody questions. One real bug here — a leakage check that compared
whole question lists instead of individual questions — reported zero leakage
while 70% of test questions were present in training. These tests pin the
arithmetic against cases whose answers can be worked out by hand.
"""

import pytest

from evaluate import (
    SIM_FLOOR,
    UNMAPPABLE,
    LeverMapper,
    score_sets,
    split_question_list,
    trivial_baselines,
)


def row(gold, pred):
    return {"gold_lever_ids": list(gold), "pred_lever_ids": list(pred)}


# ---------------------------------------------------------------------------
# Splitting a generated answer into questions
# ---------------------------------------------------------------------------

def test_numbered_list_splits_into_items():
    text = ("1. What volume commitment sits behind this?\n"
            "2. Why are we below the regional floor here?\n"
            "3. Can we extend the lease to five years?")
    assert len(split_question_list(text)) == 3


@pytest.mark.parametrize("bullet", ["-", "*", "•"])
def test_bulleted_lists_split_too(bullet):
    text = f"{bullet} What is the contracted reagent volume here?\n{bullet} Why is the discount past the floor?"
    assert len(split_question_list(text)) == 2


def test_list_markers_are_stripped():
    items = split_question_list("1. What volume commitment sits behind this?")
    assert not items[0].startswith("1.")


def test_fragments_are_dropped():
    """
    REGRESSION. The leakage check once compared whole lists rather than
    individual questions, so it reported zero leakage while 70% of test
    questions appeared in training. Splitting correctly is what made the
    real number visible.
    """
    text = "Ok\n1. What volume commitment sits behind this deal?\nSure\n"
    assert len(split_question_list(text)) == 1


def test_unformatted_single_question_is_kept():
    assert len(split_question_list("What is the contracted reagent volume?")) == 1


def test_empty_input_yields_nothing():
    assert split_question_list("") == []
    assert split_question_list("   \n  ") == []


# ---------------------------------------------------------------------------
# Set scoring arithmetic
# ---------------------------------------------------------------------------

def test_perfect_prediction_scores_one():
    m = score_sets([row([1, 2, 3], [1, 2, 3])])
    assert m["micro_precision"] == 1.0
    assert m["micro_recall"] == 1.0
    assert m["micro_f1"] == 1.0
    assert m["exact_set_match"] == 1.0


def test_disjoint_prediction_scores_zero():
    m = score_sets([row([1, 2], [5, 6])])
    assert m["micro_f1"] == 0.0
    assert m["exact_set_match"] == 0.0


def test_precision_and_recall_by_hand():
    """gold {1,2,3}, pred {1,2,9}: tp=2, fp=1, fn=1 -> p=r=f1=2/3."""
    m = score_sets([row([1, 2, 3], [1, 2, 9])])
    assert m["micro_precision"] == pytest.approx(0.6667, abs=1e-4)
    assert m["micro_recall"] == pytest.approx(0.6667, abs=1e-4)
    assert m["micro_f1"] == pytest.approx(0.6667, abs=1e-4)


def test_over_asking_costs_precision_not_recall():
    m = score_sets([row([1, 2], [1, 2, 3, 4, 5])])
    assert m["micro_recall"] == 1.0
    assert m["micro_precision"] == pytest.approx(0.4, abs=1e-4)


def test_under_asking_costs_recall_not_precision():
    m = score_sets([row([1, 2, 3, 4], [1])])
    assert m["micro_precision"] == 1.0
    assert m["micro_recall"] == pytest.approx(0.25, abs=1e-4)


def test_duplicate_predictions_do_not_inflate_the_score():
    once = score_sets([row([1, 2], [1, 2])])
    twice = score_sets([row([1, 2], [1, 1, 2, 2])])
    assert twice["micro_precision"] == once["micro_precision"]


def test_exact_set_match_ignores_order():
    assert score_sets([row([1, 2, 3], [3, 1, 2])])["exact_set_match"] == 1.0


def test_unmappable_predictions_are_counted_not_scored():
    """An off-list question must not be credited, and must be reported."""
    m = score_sets([row([1, 2], [1, 2, UNMAPPABLE, UNMAPPABLE])])
    assert m["micro_precision"] == 1.0, "off-list items must not count as wrong levers"
    assert m["unmappable_questions"] == 2
    assert m["unmappable_rate"] == pytest.approx(0.5, abs=1e-4)


def test_count_error_signs_the_direction():
    assert score_sets([row([1, 2], [1, 2, 3, 4])])["mean_count_error"] == 2.0
    assert score_sets([row([1, 2, 3, 4], [1])])["mean_count_error"] == -3.0


def test_per_lever_recall_reports_gold_frequency():
    rows = [row([1, 2], [1]), row([1], [1]), row([1, 2], [2])]
    pl = score_sets(rows)["per_lever_recall"]
    l1 = next(v for k, v in pl.items() if k.startswith("L1_"))
    l2 = next(v for k, v in pl.items() if k.startswith("L2_"))
    assert l1["gold_deals"] == 3 and l1["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert l2["gold_deals"] == 2 and l2["recall"] == pytest.approx(0.5, abs=1e-4)


def test_macro_weights_every_deal_equally():
    """One long deal scored perfectly, one short deal missed entirely:
    micro is dragged up by the long deal, macro is not."""
    rows = [row([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]), row([7], [8])]
    m = score_sets(rows)
    assert m["macro_f1"] == pytest.approx(0.5, abs=1e-4)
    assert m["micro_f1"] > m["macro_f1"]


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------

def test_ask_all_scores_perfect_recall_and_poor_precision():
    """This is why recall alone cannot be the headline metric."""
    rows = [row([1, 2], [1]), row([1, 3], [1])]
    b = trivial_baselines(rows)
    assert b["ask_all_14"]["micro_recall"] == 1.0
    assert b["ask_all_14"]["micro_precision"] < 0.3


def test_most_common_baseline_exists_and_is_sized_to_the_gold_mean():
    rows = [row([1, 2, 3], [1]) for _ in range(10)]
    b = trivial_baselines(rows)
    key = next(k for k in b if k.startswith("most_common_"))
    assert len(b[key]["levers"]) == 3, "k should match the mean gold list length"
    assert b[key]["micro_f1"] == 1.0, "a constant gold set is perfectly guessable"


def test_baselines_are_always_reported():
    """They are computed automatically so the floor is in the results file
    rather than left for the reader to wonder about."""
    b = trivial_baselines([row([1, 2], [1])])
    assert "ask_all_14" in b
    assert any(k.startswith("most_common_") for k in b)


# ---------------------------------------------------------------------------
# Lever mapper and the similarity floor
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper():
    return LeverMapper({
        1: ["Is there any headroom to raise price on this deal?",
            "Can we increase price at all here?"],
        2: ["Can we raise price on the weakest margin lines?",
            "What about a targeted increase on low margin lines?"],
        14: ["The payback lands after the lease ends, can we extend the term?",
             "Can we move to a five year lease given the breakeven?"],
    })


def test_a_clear_question_maps_to_its_lever(mapper):
    lever, sim = mapper.map("Is there headroom to raise the price here?")
    assert lever == 1
    assert sim >= SIM_FLOOR


def test_an_off_topic_question_is_unmappable(mapper):
    """
    The floor is what keeps the hallucination rate honest. Without it every
    invented question is snapped to its nearest lever and the off-list rate
    reads as zero.
    """
    lever, sim = mapper.map("What is the catering budget for the launch event?")
    assert lever == UNMAPPABLE
    assert sim < SIM_FLOOR


def test_empty_text_is_unmappable(mapper):
    assert mapper.map("   ")[0] == UNMAPPABLE


def test_mapping_is_deterministic(mapper):
    q = "Can we extend the lease term given the payback?"
    assert mapper.map(q) == mapper.map(q)


def test_similarity_to_a_named_lever_is_bounded(mapper):
    s = mapper.similarity_to("Is there headroom to raise price?", 1)
    assert 0.0 <= s <= 1.0
