"""
Stage 2: turn each deal's true feature values into a natural deal write-up.

This is the step that makes the task an UNSTRUCTURED-DATA problem. The numbers
fixed in step 1 are never shown to the downstream models -- only this prose is.

TWO DESIGN DECISIONS THAT MATTER FOR THE WRITE-UP

1. AUTHOR VOICES, SPLIT BETWEEN TRAIN AND TEST.
   Each document is written in one of several author voices. Train and
   validation draw from voice pool A; the held-out test set draws from pool B,
   which shares no phrasing conventions with A. This is a HELD-OUT SURFACE, not
   merely held-out rows: a model that memorised training phrasings collapses on
   the test set, while a model that learned the task holds up. It is the direct
   evidence for "did it learn or memorise".

   Alternative considered and rejected: a single voice with a random train/test
   split. Cheaper, but then test and train differ only by sampling, and strong
   test numbers would prove nothing about generalisation.

2. DELIBERATE OMISSIONS.
   Fields listed in the deal's `omitted_in_document` are withheld from the
   prompt, so the document genuinely does not state them. The extractor in step
   3 must answer "not stated" rather than inventing a value -- which is how the
   hallucination-detection story gets a measurable basis.

Usage:
    python src/write_documents.py                 # replay cache
    python src/write_documents.py --regenerate    # call the model again
    LLM_PROVIDER=offline python src/write_documents.py   # template fallback
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Scripts are run with `-I` (isolated mode) by run.py so that the working
# directory never leaks onto sys.path -- a planted module next to the data
# could otherwise be imported ahead of a real one. That also removes this
# file's own directory, so add it back explicitly.
sys.path.insert(0, str(Path(__file__).parent))

from llm import CachedLLM  # noqa: E402

# ---------------------------------------------------------------------------
# Author voices. Pool A -> train/val. Pool B -> test only.
# ---------------------------------------------------------------------------

VOICES_A = {
    "terse_regional_manager":
        "a regional sales manager in a hurry. Short clipped sentences, drops "
        "articles, uses abbreviations like 'reag', 'instr', 'cust'. Lower case "
        "in places. No headings.",
    "verbose_account_manager":
        "an account manager who over-explains. Long run-on sentences, lots of "
        "hedging ('I think', 'roughly', 'give or take'), numbers written "
        "inconsistently (85k in one place, 85,000 in another).",
    "bullet_heavy_kam":
        "a key account manager who writes everything as rough bullets with "
        "dashes, fragments rather than sentences, occasional ALL CAPS for "
        "emphasis.",
    "narrative_sales_director":
        "a sales director telling the story of the account. Flowing prose, "
        "context about the relationship first and numbers buried mid-paragraph.",
}

VOICES_B = {
    "formal_tender_desk":
        "a tender desk analyst writing a formal file note. Full sentences, "
        "passive voice, numbers spelled out in words where natural ('one "
        "hundred and forty thousand'), section-like structure without headings.",
    "clipped_export_manager":
        "an export manager writing for an internal committee. Measured, "
        "impersonal, uses 'the customer' and 'the opportunity', gives figures "
        "as approximations ('in the region of', 'circa').",
}

# Human-readable labels for the fields we may put in the prompt.
FIELD_LABELS = {
    "segment": "customer segment",
    "region": "region",
    "channel": "route to market",
    "instrument_count": "number of instruments placed",
    "assay_count": "number of assays on the menu",
    "hardware_outlay_k": "hardware investment (thousands)",
    "consumable_investment_k": "consumable investment (thousands)",
    "annual_reagent_k": "annual reagent revenue (thousands)",
    "reagent_margin": "reagent gross margin (fraction)",
    "lease_months": "lease period in months",
    "list_price_k": "instrument list price (thousands)",
    "discount_pct": "discount given (fraction)",
    "drp_vs_threshold": "deal realised price as a multiple of threshold (below 1.0 is under threshold, negative is loss-making)",
    "worst_line_margin": "lowest line-item margin (fraction)",
    "customer_demands_reduction": "customer is pressing for a price reduction",
    "competitor_present": "a competitor is in the deal",
    "competitor_gtm": "competitor route to market",
    "competitor_bundling": "competitor is bundling or placing instruments free",
    "installed_base": "we already have an installed base at this account",
    "alt_config_available": "an alternative instrument configuration is available",
    "higher_margin_assay_available": "a higher-margin assay mix is available",
    "breakeven_months": "break-even point in months",
    "roi_pct": "return on investment (percent)",
    "deal_margin_pct": "deal margin (percent)",
}

# Never put these in the document: they are bookkeeping or would give the game away.
NEVER_IN_DOCUMENT = {
    "deal_id", "archetype", "omitted_in_document", "margin_floor",
    "target_breakeven_months", "annual_reagent_profit_k",
    "contract_reagent_revenue_k",
}

SYSTEM = (
    "You write internal sales deal notes for a medical diagnostics company that "
    "places analysers with customers and earns revenue on reagents. You write "
    "exactly as the described person would write, including their untidiness. "
    "Never use headings, never use a table, never add a summary or "
    "recommendation, and never mention any figure you were not given."
)


def build_prompt(deal: dict, voice_name: str, voice_desc: str) -> list:
    omitted = set(deal.get("omitted_in_document", []))
    facts = []
    for k, v in deal.items():
        if k in NEVER_IN_DOCUMENT or k in omitted or v is None:
            continue
        label = FIELD_LABELS.get(k)
        if label:
            facts.append(f"- {label}: {v}")

    user = (
        f"Write the deal note as {voice_desc}\n\n"
        f"Facts you may use (and only these):\n" + "\n".join(facts) + "\n\n"
        "Write 90 to 160 words. Weave the figures into the text naturally rather "
        "than listing them in the order given. Do not state anything that is not "
        "in the list above."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def offline_document(deal: dict, voice_name: str, rng: random.Random) -> str:
    """
    Deterministic template fallback, so the pipeline can be exercised without
    API credits. Clearly inferior to real generation -- the final dataset should
    be produced with a model, and the run metadata records which mode was used.
    """
    omitted = set(deal.get("omitted_in_document", []))

    def have(k):
        return k not in omitted and deal.get(k) is not None

    bits = [
        f"{deal['segment'].replace('_', ' ')} opportunity in {deal['region']}, "
        f"{deal['channel']} route."
    ]
    bits.append(
        f"{deal['instrument_count']} instrument(s), {deal['assay_count']} assay menu, "
        f"hardware {deal['hardware_outlay_k']}k against list {deal['list_price_k']}k "
        f"at {round(deal['discount_pct'] * 100)}% off, "
        f"consumable investment {deal['consumable_investment_k']}k."
    )
    bits.append(
        f"Reagent pull-through {deal['annual_reagent_k']}k a year at "
        f"{round(deal['reagent_margin'] * 100)}% margin over a "
        f"{deal['lease_months']} month lease; pays back in "
        f"{deal['breakeven_months']} months."
    )
    bits.append(f"ROI {deal['roi_pct']}%, deal margin {deal['deal_margin_pct']}%.")
    if have("drp_vs_threshold"):
        bits.append(f"Realised price sits at {deal['drp_vs_threshold']} of threshold.")
    if have("worst_line_margin"):
        bits.append(f"Weakest line is on {round(deal['worst_line_margin'] * 100)}% margin.")
    if deal.get("customer_demands_reduction"):
        bits.append("Customer is pushing for a reduction.")
    if deal.get("competitor_present"):
        s = "Competitor in play"
        if have("competitor_gtm"):
            s += f", selling {deal['competitor_gtm']}"
        if have("competitor_bundling") and deal["competitor_bundling"]:
            s += ", and bundling instruments in"
        bits.append(s + ".")
    if deal.get("installed_base"):
        bits.append("We have an installed base here already.")
    if have("alt_config_available") and deal["alt_config_available"]:
        bits.append("An alternative configuration is available.")
    if deal.get("higher_margin_assay_available"):
        bits.append("Could shift the mix to higher-margin assays.")
    return " ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None,
                    help="write only the first N documents (for a smoke test)")
    args = ap.parse_args()

    deals = [json.loads(l) for l in (args.data / "deals.jsonl").open(encoding="utf-8")]
    if args.limit:
        deals = deals[: args.limit]

    rng = random.Random(args.seed)
    llm = CachedLLM("documents", regenerate=args.regenerate)

    # Deterministic split BEFORE any generation, so voice pools are fixed.
    # 70/10/20 train/val/test.
    idx = list(range(len(deals)))
    rng.shuffle(idx)
    n_train, n_val = int(0.70 * len(idx)), int(0.10 * len(idx))
    split_of = {}
    for rank, i in enumerate(idx):
        split_of[i] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")

    # Build every prompt first, so we can tell the user up front how many real
    # API calls this run will make. Silently firing off a thousand sequential
    # calls with no output looks exactly like a hang.
    planned = []
    for i, deal in enumerate(deals):
        split = split_of[i]
        pool = VOICES_A if split in ("train", "val") else VOICES_B
        voice_name = rng.choice(sorted(pool))
        planned.append((deal, split, voice_name, pool[voice_name],
                        build_prompt(deal, voice_name, pool[voice_name])))

    n_cached = sum(1 for *_, m in planned if llm.is_cached(m, 0.8))
    n_calls = len(planned) - n_cached if not args.regenerate else len(planned)
    print(f"provider={llm.provider} model={llm.model}")
    print(f"{len(planned)} documents: {n_cached} already cached, "
          f"{n_calls} need a model call")
    if n_calls and llm.provider != "offline":
        print(f"  roughly {n_calls * 2 // 60 + 1} min at ~2s per call. "
              f"Ctrl+C is safe -- progress is saved every 25 documents.")
    print()

    out = []
    for n, (deal, split, voice_name, voice_desc, messages) in enumerate(planned, 1):
        text = llm.chat(
            messages,
            temperature=0.8,
            offline_fallback=lambda d=deal, v=voice_name: offline_document(d, v, rng),
        )
        if n % 25 == 0 or n == len(planned):
            print(f"  {n}/{len(planned)} documents", flush=True)
            llm.save()   # checkpoint, so Ctrl+C never loses completed work
        out.append({
            "deal_id": deal["deal_id"],
            "split": split,
            "voice": voice_name,
            "voice_pool": "A" if split in ("train", "val") else "B",
            "document": text,
            "omitted_in_document": deal["omitted_in_document"],
        })

    llm.save()
    with (args.data / "documents.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "stage": "2_document_writing",
        "seed": args.seed,
        "llm": llm.summary(),
        "n_documents": len(out),
        "split_counts": {s: sum(1 for r in out if r["split"] == s) for s in ("train", "val", "test")},
        "voice_pool_A": sorted(VOICES_A),
        "voice_pool_B_test_only": sorted(VOICES_B),
        "fields_never_in_document": sorted(NEVER_IN_DOCUMENT),
        "note": (
            "Test documents use voice pool B, which shares no phrasing conventions "
            "with pool A. This is a held-out SURFACE, not just held-out rows."
        ),
    }
    (args.data / "documents_card.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta["llm"], indent=2))
    print("splits:", meta["split_counts"])
    print("\nsample (pool A):\n ", out[0]["document"][:400])
    t = next(r for r in out if r["voice_pool"] == "B")
    print("\nsample (pool B, test):\n ", t["document"][:400])


if __name__ == "__main__":
    main()
