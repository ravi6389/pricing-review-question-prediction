"""
Stage 5: turn each lever into a manager's question, and assemble the training set.

COST DECISION -- VARIANT POOLS, NOT ONE CALL PER TURN
    There are 4,786 manager turns. Generating each one with a large model would
    be ~4,800 calls. Instead we generate ~40 PHRASINGS PER LEVER (560 calls) plus
    a pool of acknowledgement openers, then compose each turn as
        acknowledgement(previous answer) + question(current lever)
    This is roughly a tenth of the cost, it keeps full control over which lever
    each turn expresses, and -- the part that matters -- it lets us HOLD OUT
    phrasings for the test set.

SURFACE HOLDOUT ON BOTH SIDES OF THE CONVERSATION
    Phrasings 0-29 of each lever are used for train and validation; phrasings
    30-39 only for test. Combined with the held-out author voices in stage 2,
    the test set shares no surface wording with training on either the document
    side or the manager side. A model that memorised phrasings cannot score on
    it; a model that learned the ladder can.

WHAT THE MODEL SEES, AND WHAT IT DOES NOT
    The training prompt contains the deal note and the conversation so far. It
    does NOT contain the lever taxonomy -- learning the ladder is the task.
    Evaluation of BASE models uses an augmented prompt that does list the
    levers, because a base model has never seen the taxonomy and scoring it
    without one would be a rigged comparison. That augmented prompt is built in
    stage 7, not here.

Usage:
    python src/phrase_questions.py --seed 42
    python src/phrase_questions.py --regenerate     # call the model
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from levers import LEVERS, LEVERS_BY_ID  # noqa: E402
from llm import CachedLLM  # noqa: E402

import os as _os, sys as _sys  # noqa: E402
_ensure_src_on_path = _os.path.dirname(_os.path.abspath(__file__))
if _ensure_src_on_path not in _sys.path:
    _sys.path.insert(0, _ensure_src_on_path)
from logging_config import get_logger

log = get_logger(__name__)

N_VARIANTS = 40
N_TRAIN_VARIANTS = 30   # variants [0:30] -> train/val, [30:40] -> test

SYSTEM = (
    "You write the questions a commercial pricing review manager asks a sales "
    "representative about a deal. The manager is probing, specific and "
    "businesslike -- never chatty, never apologetic. Questions are one or two "
    "sentences. Reply with a JSON array of strings and nothing else."
)


def variants_prompt(lever) -> list:
    user = (
        f"The manager wants to: {lever.intent}\n\n"
        f"Write {N_VARIANTS} different ways of asking this, as a JSON array of "
        f"{N_VARIANTS} strings.\n"
        "Vary the wording substantially -- different verbs, different sentence "
        "shapes, some direct and some framed as a suggestion. Do not number "
        "them. Do not reference any previous answer; that is added separately. "
        "This is medical diagnostics: instrument placements, reagent "
        "pull-through, lease terms, tender pricing."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


ACK_PROMPT_SYSTEM = (
    "You write the short acknowledgement a pricing review manager puts in front "
    "of their next question, referring to what the representative just said. "
    "Reply with a JSON array of strings and nothing else."
)


def ack_prompt() -> list:
    user = (
        "Write 24 ways to open a follow-up question by referring back to what "
        "the representative just said, as a JSON array of 24 strings.\n"
        "Each must contain the placeholder {prior} exactly once, where their "
        "previous point will be inserted, and end with a connector into the "
        "next question.\n"
        'Examples of the shape: "You mentioned {prior} -- ", '
        '"You said {prior}, and I want to understand that better. ", '
        '"Given {prior}, "\n'
        "Vary length and formality. No numbering."
    )
    return [{"role": "system", "content": ACK_PROMPT_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Offline fallbacks (deterministic; used when LLM_PROVIDER=offline)
# ---------------------------------------------------------------------------

OFFLINE_STEMS = [
    "Can we look at {t}?",
    "What is the position on {t}?",
    "I want to test {t} before we move on.",
    "Is there anything available on {t}?",
    "Talk me through {t}.",
    "Where do we stand on {t}?",
    "Has {t} been explored?",
    "Could {t} be revisited?",
]

LEVER_TOPIC = {
    1: "room to raise the price on this deal",
    2: "a targeted increase on the lowest-margin lines",
    3: "a rebate structure instead of a direct reduction",
    4: "margin through instrument configuration or assay mix",
    5: "how the competitor goes to market, direct or through a dealer",
    6: "whether the competitive pressure and price sensitivity are justified",
    7: "what is actually driving the customer's price sensitivity",
    8: "whether our price is competitive in this tender",
    9: "whether the realised price can be improved",
    10: "which specific lines are driving the negative realised price",
    11: "the risk if we lose this business and the exposure on the installed base",
    12: "whether the competitor is bundling or placing instruments free",
    13: "what protects this business from price matching after the win and at renewal",
    14: "a lease term that ends before the deal pays back",
}

OFFLINE_ACKS = [
    "You mentioned {prior} -- ",
    "You said {prior}, and I want to understand that better. ",
    "Given {prior}, ",
    "Noting {prior}, ",
    "Since {prior}, ",
    "Understood on {prior}. ",
]


def offline_variants(lever) -> str:
    topic = LEVER_TOPIC[lever.id]
    out = []
    for i in range(N_VARIANTS):
        # Offline mode cycles a small stem list, so variants repeat. That is
        # fine for a plumbing test; real variety needs the model.
        out.append(OFFLINE_STEMS[i % len(OFFLINE_STEMS)].format(t=topic))
    return json.dumps(out)


def parse_array(raw: str, expected: int, label: str) -> list:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1].lstrip("json").strip()
    try:
        arr = json.loads(txt)
    except json.JSONDecodeError:
        raise SystemExit(f"\n{label}: model did not return valid JSON.\n{txt[:400]}\n")
    if not isinstance(arr, list) or not arr:
        raise SystemExit(f"\n{label}: expected a JSON array, got {type(arr).__name__}\n")
    # Pad by cycling rather than failing: a model returning 37 of 40 is fine.
    while len(arr) < expected:
        arr.append(arr[len(arr) % max(len(arr), 1)])
    return [str(x).strip() for x in arr[:expected]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

SET_SYSTEM = (
    "You are a commercial pricing review manager at a medical diagnostics "
    "company. Read the deal note below and list the questions you would put to "
    "the sales representative before approving it. Ask only what this "
    "particular deal warrants -- a clean deal needs few questions, a "
    "problematic one needs many. One question per line, numbered.\n\n"
    "Deal note:\n{document}"
)

TRAIN_SYSTEM = (
    "You are a commercial pricing review manager at a medical diagnostics "
    "company. You are reviewing the deal below with the sales representative "
    "who owns it. Ask the single most useful next question. Never repeat a "
    "point that has already been settled.\n\nDeal note:\n{document}"
)


def short_prior(answer: str) -> str:
    """Compress the rep's last answer into something an acknowledgement can carry."""
    a = answer.strip().rstrip(".")
    return a[0].lower() + a[1:] if a and a[0].isupper() else a


def build_set_example(qset, doc, variants, rng, split):
    """
    SET MODE: one example per deal. The target is the whole numbered list of
    questions, so the model learns WHICH questions this deal warrants and how
    many -- not just the next one.

    No acknowledgement prefixes: there is no previous answer to acknowledge.
    """
    lo, hi = (0, N_TRAIN_VARIANTS) if split in ("train", "val") else (N_TRAIN_VARIANTS, N_VARIANTS)
    lines, lever_ids = [], []
    for item in qset["levers"]:
        q = rng.choice(variants[item["lever_id"]][lo:hi])
        lines.append(f"{len(lines) + 1}. {q}")
        lever_ids.append(item["lever_id"])

    return {
        "deal_id": qset["deal_id"],
        "split": split,
        "n_questions": len(lines),
        "gold_lever_ids": lever_ids,
        "gold_lever_names": [i["lever_name"] for i in qset["levers"]],
        "verdict": qset["verdict"],
        "messages": [
            {"role": "system", "content": SET_SYSTEM.format(document=doc)},
            {"role": "assistant", "content": "\n".join(lines)},
        ],
    }


def build_examples(review, doc, variants, acks, rng, split):
    """One training example per manager turn."""
    lo, hi = (0, N_TRAIN_VARIANTS) if split in ("train", "val") else (N_TRAIN_VARIANTS, N_VARIANTS)
    out, history = [], []

    for t in review["turns"]:
        pool = variants[t["lever_id"]][lo:hi]
        question = rng.choice(pool)
        if history:
            ack = rng.choice(acks).replace("{prior}", short_prior(history[-1]["answer"]))
            # parse_array strips each string, which eats the trailing space that
            # separates the acknowledgement from the question.
            if not ack.endswith(" "):
                ack += " "
            # Lower-case the question's first letter so it reads as one
            # sentence -- except "I", which must stay capitalised.
            head = question.split(" ", 1)[0]
            question = ack + (question if head == "I"
                              else question[0].lower() + question[1:])

        messages = [{"role": "system", "content": TRAIN_SYSTEM.format(document=doc)}]
        for h in history:
            messages.append({"role": "assistant", "content": h["question"]})
            messages.append({"role": "user", "content": h["answer"]})

        out.append({
            "deal_id": review["deal_id"],
            "split": split,
            "turn": t["turn"],
            "gold_lever_id": t["lever_id"],
            "gold_lever_name": t["lever_name"],
            "is_followup": t["is_followup"],
            "exhausted_before": t["exhausted_before"],
            "messages": messages + [{"role": "assistant", "content": question}],
        })
        history.append({"question": question, "answer": t["rep_answer"]})

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--mode", choices=["set", "dialogue"], default="set")
    args = ap.parse_args()

    docs = {json.loads(l)["deal_id"]: json.loads(l)
            for l in (args.data / "documents.jsonl").open(encoding="utf-8")}
    source = "question_sets.jsonl" if args.mode == "set" else "reviews.jsonl"
    reviews = [json.loads(l) for l in (args.data / source).open(encoding="utf-8")]

    llm = CachedLLM("phrasings", regenerate=args.regenerate)
    n_calls = len(LEVERS) + 1
    log.info(f"provider={llm.provider} model={llm.model}")
    log.info(f"{n_calls} generation calls ({len(LEVERS)} lever pools + 1 acknowledgement pool)\n")

    variants = {}
    for lever in LEVERS:
        raw = llm.chat(
            variants_prompt(lever), temperature=0.9, max_tokens=2000,
            offline_fallback=lambda l=lever: offline_variants(l),
        )
        variants[lever.id] = parse_array(raw, N_VARIANTS, f"lever {lever.id}")
        log.info(f"  L{lever.id:<2} {lever.name:<40} {len(set(variants[lever.id]))} unique")

    raw = llm.chat(
        ack_prompt(), temperature=0.9, max_tokens=1200,
        offline_fallback=lambda: json.dumps(OFFLINE_ACKS),
    )
    acks = [a for a in parse_array(raw, 24, "acknowledgements") if "{prior}" in a] or OFFLINE_ACKS
    llm.save()
    log.info(f"  acknowledgement openers: {len(acks)}\n")

    rng = random.Random(args.seed)
    buckets = {"train": [], "val": [], "test": []}
    for review in reviews:
        doc = docs[review["deal_id"]]
        split = doc["split"]
        if args.mode == "set":
            buckets[split].append(
                build_set_example(review, doc["document"], variants, rng, split))
        else:
            buckets[split].extend(
                build_examples(review, doc["document"], variants, acks, rng, split))

    for split, name in (("train", "train"), ("val", "val"), ("test", "test_ood")):
        with (args.data / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in buckets[split]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # LEAKAGE CHECK -- run every time, reported in the card.
    # "No test leakage" is an explicit evaluation requirement, so it is
    # measured rather than asserted. Shared questions should be 0: train/val
    # draw phrasings [0:30] and test draws [30:40], which are disjoint IF the
    # generator actually returned 40 distinct phrasings. With the offline
    # template fallback it returns only ~8 distinct stems cycled, so the
    # slices collide and this number is large -- a true signal that the
    # surface holdout is not real until stage 5 is run against a real model.
    # In SET mode the final message is a whole numbered list, so comparing
    # messages wholesale would compare LISTS -- which almost never collide even
    # when every individual question is shared. Split to individual questions.
    def _questions(example):
        text = example["messages"][-1]["content"]
        if args.mode != "set":
            return [text]
        out = []
        for line in text.splitlines():
            line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            if line:
                out.append(line)
        return out

    q_train = {q for e in buckets["train"] + buckets["val"] for q in _questions(e)}
    q_test = {q for e in buckets["test"] for q in _questions(e)}
    d_train = {e["deal_id"] for e in buckets["train"] + buckets["val"]}
    d_test = {e["deal_id"] for e in buckets["test"]}
    shared_q = len(q_train & q_test)
    leakage = {
        "unique_questions_train_val": len(q_train),
        "unique_questions_test": len(q_test),
        "shared_questions": shared_q,
        "shared_question_rate": round(shared_q / max(len(q_test), 1), 4),
        "shared_deals": len(d_train & d_test),
        "surface_holdout_clean": shared_q == 0 and not (d_train & d_test),
    }
    if shared_q:
        log.info(f"\n  WARNING: {shared_q} manager questions appear in BOTH train and "
              f"test ({leakage['shared_question_rate']:.0%} of test).")
        log.info("  Expected when LLM_PROVIDER=offline (the template yields ~8 distinct")
        log.info("  stems per lever, so the 0-29 / 30-39 slices collide).")
        log.info("  Re-run with a real model to make the surface holdout genuine.\n")

    card = {
        "stage": "5_question_phrasing",
        "leakage_check": leakage,
        "seed": args.seed,
        "llm": llm.summary(),
        "variants_per_lever": N_VARIANTS,
        "train_variant_slice": [0, N_TRAIN_VARIANTS],
        "test_variant_slice": [N_TRAIN_VARIANTS, N_VARIANTS],
        "n_examples": {k: len(v) for k, v in buckets.items()},
        "mode": args.mode,
        "label_distribution": {
            split: {f"L{k}": v for k, v in sorted(Counter(
                lid for e in ex
                for lid in (e["gold_lever_ids"] if args.mode == "set"
                            else [e["gold_lever_id"]])).items())}
            for split, ex in buckets.items()
        },
        "mean_questions_per_example": {
            split: round(sum(e.get("n_questions", 1) for e in ex) / max(len(ex), 1), 2)
            for split, ex in buckets.items()
        },
        "holdout_note": (
            "Test examples use author voice pool B (stage 2) AND manager "
            "phrasing variants 30-39, neither of which appear in training. The "
            "test set therefore shares no surface wording with training on "
            "either side of the conversation."
        ),
        "limitations": [
            "Manager questions are composed from a fixed variant pool plus an "
            "acknowledgement opener, so they are more formulaic than a real "
            "manager, who would weave the previous answer into the question.",
            "Acknowledgements quote the rep's answer verbatim; a real manager "
            "paraphrases.",
        ],
    }
    (args.data / "training_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    log.info("examples: %s", card["n_examples"])
    log.info("mean questions per example: %s", card["mean_questions_per_example"])
    log.info("\nsample (test split, held-out surface):")
    ex = max(buckets["test"], key=lambda e: e.get("n_questions", 1))
    log.info("  [system] ...deal note...")
    for line in ex["messages"][-1]["content"].split("\n"):
        log.info(f"           {line}")
    log.info(f"  gold levers: {ex.get('gold_lever_ids', ex.get('gold_lever_id'))}")


if __name__ == "__main__":
    main()
