"""
Stage 4: run the lever rules over each deal to produce a review conversation.

WHAT THIS PRODUCES
    For each deal, a sequence of turns:
        manager asks lever L  ->  rep answers  ->  manager asks lever L'  -> ...
    plus a terminal verdict. The LEVER IDS are the ground truth. Stage 5 turns
    each lever into natural manager phrasing; the model is later scored on
    whether it picks the right lever, not on whether it matched a wording.

WHY THE REP IS SIMULATED BY RULES AND NOT A MODEL
    The rep's answers steer the ladder, so if a model improvised them the
    conversation trajectory would vary between runs and the dataset would stop
    being reproducible. The simulator instead draws answers from declared
    per-lever response distributions: mostly terse refusals, sometimes a short
    unexplained assertion (which triggers a follow-up), occasionally a
    substantive answer that closes the lever.

    Alternative considered and rejected: a model playing the rep. More natural
    phrasing, but non-reproducible, slower, and it would let the rep's model
    decide the conversation shape -- which is exactly the institutional
    knowledge the rules are supposed to own.

HONEST LIMITATION (goes in the write-up)
    Real reps negotiate, change their story, and volunteer information nobody
    asked for. This simulator does none of that. The conversations are
    therefore cleaner and shorter than real reviews, and a model trained on
    them will be under-prepared for an evasive or rambling counterparty.

Usage:
    python src/simulate_reviews.py --seed 42
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from levers import (  # noqa: E402
    LEVERS_BY_ID, DealState, next_lever, terminal_verdict,
)

# ---------------------------------------------------------------------------
# Declared rep-response behaviour
# ---------------------------------------------------------------------------

# How a rep responds, as declared probabilities. "refuse" closes the lever;
# "bare_assert" names a cause without explaining it and triggers a follow-up;
# "substantive" answers the lever properly and closes it.
RESPONSE_MIX = {"refuse": 0.56, "bare_assert": 0.26, "substantive": 0.18}

REFUSALS = [
    "no", "no room", "not possible", "no chance", "move on",
    "already tried that", "customer won't accept it", "no flexibility",
]

# Per-lever plausible short answers. Kept deliberately terse -- this mirrors
# how reps actually answer in a review, and it is what makes the follow-up
# behaviour fire.
BARE_ASSERTS = {
    1: ["price is at floor", "tender locked"],
    2: ["those lines are strategic", "volume driver"],
    3: ["customer wants it upfront", "rebates not allowed"],
    4: ["config is fixed", "menu is customer specified"],
    5: ["direct", "through a dealer"],
    6: ["pricing pressure", "price sensitivity"],
    7: ["low base price", "budget constrained", "public tender"],
    8: ["not competitive", "slightly above"],
    9: ["structurally negative", "cost base too high"],
    10: ["the reagent lines", "the service element"],
    11: ["we'd lose the account", "pull-through at risk"],
    12: ["bundled pricing", "free placement"],
    13: ["nothing contractual", "renewal is open"],
    14: ["term was customer driven", "they wanted it short"],
}

SUBSTANTIVE = {
    5: ["they sell direct with their own service team",
        "through a regional dealer who discounts hard"],
    6: ["yes, three competing bids on the same menu, all below our floor",
        "the customer published a benchmark price in the tender"],
    7: ["their reimbursement per test dropped this year so every input is squeezed",
        "they run a low base price model and volume is their only lever"],
    8: ["we're about 8 percent above the nearest bid on the same configuration",
        "competitive on instrument, well above on reagents"],
    11: ["we'd lose about 60 percent of the reagent pull-through at this site",
         "the installed base contract renews next year and would be exposed"],
    12: ["they're placing the analyser free against a five year reagent commitment",
         "bundling instrument, service and reagents into a single price"],
    13: ["nothing stops them re-bidding at renewal, we have no price protection",
         "we'd hold it for the term but renewal is wide open"],
}


def rep_answer(lever_id: int, rng: random.Random) -> tuple:
    """Return (answer_text, kind)."""
    kinds = list(RESPONSE_MIX)
    weights = [RESPONSE_MIX[k] for k in kinds]
    kind = rng.choices(kinds, weights=weights, k=1)[0]

    # Fall back DOWNWARDS in informativeness, not straight to a refusal.
    # An earlier version fell back to "refuse" whenever a lever had no entry in
    # the chosen pool, which pushed the realised refusal rate to 0.71 against a
    # declared 0.56 and starved the substantive class.
    if kind == "substantive" and lever_id in SUBSTANTIVE:
        return rng.choice(SUBSTANTIVE[lever_id]), "substantive"
    if kind in ("substantive", "bare_assert") and lever_id in BARE_ASSERTS:
        return rng.choice(BARE_ASSERTS[lever_id]), "bare_assert"
    return rng.choice(REFUSALS), "refuse"


def state_from_deal(deal: dict) -> DealState:
    return DealState(
        drp_vs_threshold=deal["drp_vs_threshold"],
        worst_line_margin=deal["worst_line_margin"],
        margin_floor=deal["margin_floor"],
        hardware_outlay_k=deal["hardware_outlay_k"],
        annual_reagent_k=deal["annual_reagent_k"],
        lease_months=deal["lease_months"],
        breakeven_months=deal["breakeven_months"],
        competitor_present=deal["competitor_present"],
        # The manager does NOT know these at the start of the review -- finding
        # them out is what levers 5 and 12 are for. Seeding them from the deal
        # record made lever 12 unfireable (its trigger is "bundling unknown")
        # while lever 13 fired immediately, skipping the ladder entirely.
        competitor_gtm=None,
        competitor_bundling=None,
        installed_base=deal["installed_base"],
        alt_config_available=deal["alt_config_available"],
        higher_margin_assay_available=deal["higher_margin_assay_available"],
        customer_demands_reduction=deal["customer_demands_reduction"],
    )


MAX_TURNS = 20  # safety valve; a real review never runs this long


def question_set(deal: dict) -> dict:
    """
    SET MODE: the questions a manager should raise on this deal, as a list.

    No conversation. Each lever is walked once, treated as answered truthfully
    from the deal record, and the ordered list of levers that fire is the
    label. Follow-up levers are excluded -- a follow-up only exists in reply to
    a bare answer, which has no meaning without a dialogue.

    WHY THIS AND NOT "ask everything": which levers fire depends on the deal's
    own numbers -- break-even against lease term, margin floor, whether a
    competitor is present, whether an installed base is at risk. So the set
    size varies by deal, and learning WHICH subset applies is the task. A fixed
    list for every deal would be memorisable in one example and would teach
    nothing.
    """
    state = state_from_deal(deal)
    truth_gtm = deal["competitor_gtm"]
    truth_bundling = deal["competitor_bundling"]
    levers = []

    for _ in range(MAX_TURNS):
        lever = next_lever(state)
        if lever is None:
            break
        if lever.is_followup:
            # cannot arise without a dialogue; close it and move on
            state.exhausted.add(lever.id)
            continue
        levers.append({
            "position": len(levers) + 1,
            "lever_id": lever.id,
            "lever_name": lever.name,
            "lever_intent": lever.intent,
        })
        # Treat as asked and truthfully answered, so the ladder advances and
        # facts the manager would learn become known.
        state.answered[lever.id] = "disclosed"
        if lever.id == 12:
            state.competitor_bundling = truth_bundling
        if lever.id == 5:
            state.competitor_gtm = truth_gtm or "unknown"

    return {
        "deal_id": deal["deal_id"],
        "n_questions": len(levers),
        "levers": levers,
        "verdict": terminal_verdict(state),
    }


def simulate(deal: dict, rng: random.Random) -> dict:
    state = state_from_deal(deal)
    turns = []
    # The true values, revealed only when the manager asks for them.
    truth_gtm = deal["competitor_gtm"]
    truth_bundling = deal["competitor_bundling"]

    for _ in range(MAX_TURNS):
        lever = next_lever(state)
        if lever is None:
            break

        answer, kind = rep_answer(lever.id, rng)
        turns.append({
            "turn": len(turns) + 1,
            "lever_id": lever.id,
            "lever_name": lever.name,
            "lever_intent": lever.intent,
            "is_followup": lever.is_followup,
            # levers ruled out BEFORE this question was asked -- the model sees
            # this, and re-asking any of them is what 'repeat rate' counts
            "exhausted_before": sorted(state.exhausted),
            "rep_answer": answer,
            "rep_answer_kind": kind,
        })

        if kind == "refuse":
            state.exhausted.add(lever.id)
            # A refused question still settles the matter for the review: the
            # manager stops treating it as unknown, so downstream triggers that
            # test "still unknown" do not loop.
            if lever.id == 12:
                state.competitor_bundling = False
            if lever.id == 5:
                state.competitor_gtm = "unknown"
        else:
            state.answered[lever.id] = answer
            if lever.id == 12:
                state.competitor_bundling = truth_bundling
            if lever.id == 5:
                state.competitor_gtm = truth_gtm or "unknown"

    return {
        "deal_id": deal["deal_id"],
        "n_turns": len(turns),
        "turns": turns,
        "verdict": terminal_verdict(state),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--mode", choices=["set", "dialogue"], default="set",
                    help="set: one question LIST per deal (default). "
                         "dialogue: adaptive turn-by-turn conversation.")
    args = ap.parse_args()

    deals = [json.loads(l) for l in (args.data / "deals.jsonl").open(encoding="utf-8")]
    rng = random.Random(args.seed)

    if args.mode == "set":
        sets = [question_set(d) for d in deals]
        with (args.data / "question_sets.jsonl").open("w", encoding="utf-8") as f:
            for r in sets:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        sizes = Counter(r["n_questions"] for r in sets)
        freq = Counter(l["lever_id"] for r in sets for l in r["levers"])
        total_q = sum(r["n_questions"] for r in sets)
        stats = {
            "stage": "4_question_sets",
            "mode": "set",
            "seed": args.seed,
            "n_deals": len(sets),
            "n_questions_total": total_q,
            "mean_questions_per_deal": round(total_q / len(sets), 2),
            "question_count_distribution": dict(sorted(sizes.items())),
            "lever_frequency": {
                f"L{i}_{LEVERS_BY_ID[i].name}": {
                    "deals": freq.get(i, 0),
                    "share_of_deals": round(freq.get(i, 0) / len(sets), 4)}
                for i in sorted(LEVERS_BY_ID)
            },
            "verdict_distribution": {
                k: round(v / len(sets), 3)
                for k, v in sorted(Counter(r["verdict"] for r in sets).items())},
            "note": (
                "Set size varies by deal because lever triggers depend on the "
                "deal's own economics. That variation IS the task: a fixed list "
                "for every deal would be memorisable from one example."
            ),
        }
        (args.data / "question_sets_card.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
        Path("results").mkdir(exist_ok=True)
        (Path("results") / "lever_distribution.json").write_text(
            json.dumps(stats["lever_frequency"], indent=2), encoding="utf-8")

        print(f"{len(sets)} deals, {total_q} questions, "
              f"mean {stats['mean_questions_per_deal']} per deal")
        print("questions per deal:", stats["question_count_distribution"])
        print("\nlever frequency (share of deals asking it):")
        for k, v in stats["lever_frequency"].items():
            bar = "#" * int(v["share_of_deals"] * 50)
            print(f"  {k:<42} {v['deals']:>5}  {bar}")
        print("\nverdicts:", stats["verdict_distribution"])
        return

    reviews = [simulate(d, rng) for d in deals]

    with (args.data / "reviews.jsonl").open("w", encoding="utf-8") as f:
        for r in reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lever_counts = Counter(t["lever_id"] for r in reviews for t in r["turns"])
    turn_lengths = Counter(r["n_turns"] for r in reviews)
    total_turns = sum(r["n_turns"] for r in reviews)

    stats = {
        "stage": "4_review_simulation",
        "seed": args.seed,
        "n_reviews": len(reviews),
        "n_turns_total": total_turns,
        "mean_turns_per_review": round(total_turns / len(reviews), 2),
        "turn_length_distribution": dict(sorted(turn_lengths.items())),
        "declared_response_mix": RESPONSE_MIX,
        "realised_response_mix": {
            k: round(v / total_turns, 3) for k, v in sorted(
                Counter(t["rep_answer_kind"] for r in reviews for t in r["turns"]).items())
        },
        "lever_frequency": {
            f"L{i}_{LEVERS_BY_ID[i].name}": {
                "turns": lever_counts.get(i, 0),
                "share_of_turns": round(lever_counts.get(i, 0) / total_turns, 4),
            }
            for i in sorted(LEVERS_BY_ID)
        },
        "verdict_distribution": {
            k: round(v / len(reviews), 3) for k, v in sorted(
                Counter(r["verdict"] for r in reviews).items())
        },
        "class_imbalance_note": (
            "Lever frequency is the label distribution for the next-lever task. "
            "It is heavily imbalanced by construction, because early ladder "
            "levers fire in almost every review while late ones need specific "
            "conditions. Report macro-F1 alongside accuracy: accuracy alone is "
            "dominated by the common early levers."
        ),
        "limitations": [
            "Rep answers are drawn from fixed per-lever pools, so the model sees "
            "far less linguistic variety on the rep side than in reality.",
            "The simulated rep never contradicts itself, stalls, or volunteers "
            "information, so the model is not trained for evasive counterparties.",
            "Answer kind is independent of the deal's economics; a real rep is "
            "more likely to refuse when genuinely boxed in.",
        ],
    }
    (args.data / "reviews_card.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "lever_distribution.json").write_text(
        json.dumps(stats["lever_frequency"], indent=2), encoding="utf-8")

    print(f"{len(reviews)} reviews, {total_turns} turns, "
          f"mean {stats['mean_turns_per_review']} turns/review")
    print("\nlever frequency (this is your label distribution):")
    for k, v in stats["lever_frequency"].items():
        bar = "#" * int(v["share_of_turns"] * 120)
        print(f"  {k:<42} {v['turns']:>5}  {bar}")
    print("\nverdicts:", stats["verdict_distribution"])
    print("response mix:", stats["realised_response_mix"])


if __name__ == "__main__":
    main()
