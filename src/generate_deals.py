"""
Stage 1 of the data pipeline: fix the TRUE feature values for each synthetic deal.

WHY A SCRIPT AND NOT A MODEL DOES THIS (for the write-up):

  1. It is the answer key. Later stages render these numbers into prose and an
     extractor reads them back out. Because the true values were fixed here,
     extraction can be SCORED. If a model had invented the deal and a second
     model had read the numbers back, the "ground truth" would itself be a
     guess and extraction accuracy would be unmeasurable.

  2. It controls coverage. Left to chance, rare situations (competitor bundling,
     break-even falling outside the lease term) would appear too few times to
     train or test on. Declared archetype probabilities guarantee every
     questioning lever gets enough examples.

DESIGN DECISION -- ARCHETYPE-CONDITIONED SAMPLING:
  Drawing every field independently gives incoherent deals (a huge instrument
  placement with a trivial reagent commitment) and leaves lever coverage to
  luck. Instead each deal first draws an ARCHETYPE from a declared mix, and the
  field distributions are conditioned on it. This buys coherence and guaranteed
  coverage at the cost of a tidier world than production -- recorded as a
  limitation in the dataset card.

  Alternative considered and rejected: rejection sampling against coverage
  targets. It hits the same targets but the resulting distribution is implicit
  in the acceptance rule rather than declared up front, which makes it harder to
  document honestly.

Usage:
    python src/generate_deals.py --n 1000 --seed 42
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# DECLARED DISTRIBUTIONS
# Everything a reviewer needs to judge the data's realism lives in this block.
# ---------------------------------------------------------------------------

# Archetypes shape the commercial posture of the deal. Probabilities declared
# so that every lever in the ladder gets a workable number of examples.
ARCHETYPES = {
    # healthy economics, little pressure -- short reviews, early levers only
    "clean": 0.18,
    # price sits below threshold, margin levers are the whole conversation
    "margin_squeeze": 0.24,
    # realised price is negative; the deep levers fire
    "underwater": 0.16,
    # competitor bundling or free placement drives the conversation
    "competitive_bundle": 0.18,
    # break-even falls outside the lease term
    "term_mismatch": 0.14,
    # incumbent displacement, installed base at risk
    "displacement": 0.10,
}

SEGMENTS = ["public_hospital", "private_hospital", "reference_lab", "clinic_chain"]
SEGMENT_W = [0.30, 0.26, 0.28, 0.16]

REGIONS = ["north", "south", "east", "west", "export"]
REGION_W = [0.24, 0.30, 0.16, 0.22, 0.08]

CHANNELS = ["tender", "direct", "dealer"]
CHANNEL_W = [0.42, 0.40, 0.18]

COMPETITOR_GTM = ["direct", "dealer"]
COMPETITOR_GTM_W = [0.62, 0.38]

MARGIN_FLOOR = 0.18  # policy floor for a single line item, declared constant

# Per-archetype field ranges. (low, high) inclusive, drawn uniformly unless noted.
# DESIGN DECISION -- PAYBACK IS DRAWN, REAGENT REVENUE IS DERIVED.
#
# A first version drew the annual reagent commitment directly and derived
# break-even from it. That produced break-even falling outside the lease term in
# 76% of deals, which is commercially absurd: a placement whose payback never
# lands inside the contract would not be signed. The fix mirrors how these deals
# are actually structured -- a deal desk SIZES the reagent commitment to hit a
# target payback -- so target_breakeven_months is drawn and annual_reagent_k is
# derived from it. Keep this note: it is a concrete example of a modelling
# assumption corrected by a sanity check, which the write-up cites.
ARCHETYPE_PARAMS = {
    "clean": dict(
        hardware_outlay_k=(40, 140), target_breakeven_months=(14, 26), reagent_margin=(0.34, 0.48),
        lease_months=[36, 60], drp_vs_threshold=(1.02, 1.18), worst_line_margin=(0.19, 0.34),
        competitor_present_p=0.30, bundling_p=0.10, installed_base_p=0.25,
        demands_reduction_p=0.15,
    ),
    "margin_squeeze": dict(
        hardware_outlay_k=(60, 200), target_breakeven_months=(22, 34), reagent_margin=(0.24, 0.36),
        lease_months=[36, 60], drp_vs_threshold=(0.86, 0.99), worst_line_margin=(0.06, 0.17),
        competitor_present_p=0.72, bundling_p=0.22, installed_base_p=0.45,
        demands_reduction_p=0.80,
    ),
    "underwater": dict(
        hardware_outlay_k=(90, 280), target_breakeven_months=(30, 46), reagent_margin=(0.18, 0.30),
        lease_months=[36, 60], drp_vs_threshold=(-0.12, -0.01), worst_line_margin=(0.01, 0.12),
        competitor_present_p=0.85, bundling_p=0.40, installed_base_p=0.55,
        demands_reduction_p=0.88,
    ),
    "competitive_bundle": dict(
        hardware_outlay_k=(70, 220), target_breakeven_months=(20, 34), reagent_margin=(0.22, 0.38),
        lease_months=[36, 60], drp_vs_threshold=(0.82, 1.02), worst_line_margin=(0.08, 0.22),
        competitor_present_p=1.00, bundling_p=1.00, installed_base_p=0.40,
        demands_reduction_p=0.70,
    ),
    "term_mismatch": dict(
        # a large placement on a short lease: payback lands beyond the term
        hardware_outlay_k=(110, 300), target_breakeven_months=(34, 56), reagent_margin=(0.20, 0.34),
        lease_months=[24, 36], drp_vs_threshold=(0.88, 1.08), worst_line_margin=(0.10, 0.26),
        competitor_present_p=0.55, bundling_p=0.18, installed_base_p=0.30,
        demands_reduction_p=0.45,
    ),
    "displacement": dict(
        hardware_outlay_k=(80, 240), target_breakeven_months=(20, 32), reagent_margin=(0.26, 0.40),
        lease_months=[36, 60], drp_vs_threshold=(0.84, 1.04), worst_line_margin=(0.09, 0.24),
        competitor_present_p=0.95, bundling_p=0.45, installed_base_p=1.00,
        demands_reduction_p=0.65,
    ),
}

# Fields that are sometimes genuinely absent from a deal write-up, so the
# extractor must return "not stated" rather than inventing a value.
OMISSION_PROB = {
    "competitor_gtm": 0.22,
    "competitor_bundling": 0.26,
    "worst_line_margin": 0.14,
    "alt_config_available": 0.18,
}


def _u(rng, lo, hi, nd=0):
    v = rng.uniform(lo, hi)
    return round(v, nd) if nd else int(round(v))


def _uf(rng, lo, hi, nd=3):
    return round(rng.uniform(lo, hi), nd)


def make_deal(rng: random.Random, deal_id: int) -> dict:
    arch = rng.choices(list(ARCHETYPES), weights=list(ARCHETYPES.values()), k=1)[0]
    p = ARCHETYPE_PARAMS[arch]

    hardware_outlay_k = _u(rng, *p["hardware_outlay_k"])
    reagent_margin = _uf(rng, *p["reagent_margin"])
    lease_months = rng.choice(p["lease_months"])

    # Size the reagent commitment to hit the target payback, as a deal desk would.
    target_breakeven = _u(rng, *p["target_breakeven_months"])
    annual_reagent_k = int(round(
        hardware_outlay_k / target_breakeven * 12 / reagent_margin
    ))

    competitor_present = rng.random() < p["competitor_present_p"]
    competitor_bundling = (rng.random() < p["bundling_p"]) if competitor_present else False
    installed_base = rng.random() < p["installed_base_p"]

    # ---- DERIVED, not drawn: these follow from the numbers above, as in reality
    annual_reagent_profit_k = round(annual_reagent_k * reagent_margin, 2)
    breakeven_months = (
        math.ceil(hardware_outlay_k / annual_reagent_profit_k * 12)
        if annual_reagent_profit_k > 0 else 999
    )
    contract_reagent_revenue_k = round(annual_reagent_k * lease_months / 12, 1)
    contract_reagent_profit_k = round(annual_reagent_profit_k * lease_months / 12, 1)
    roi_pct = round(
        (contract_reagent_profit_k - hardware_outlay_k) / hardware_outlay_k * 100, 1
    )
    deal_margin_pct = round(
        (contract_reagent_profit_k - hardware_outlay_k)
        / max(contract_reagent_revenue_k, 1e-6) * 100, 1
    )

    deal = {
        "deal_id": f"D{deal_id:05d}",
        "archetype": arch,                       # kept for analysis, NOT shown to the model
        # --- account shape
        "segment": rng.choices(SEGMENTS, weights=SEGMENT_W, k=1)[0],
        "region": rng.choices(REGIONS, weights=REGION_W, k=1)[0],
        "channel": rng.choices(CHANNELS, weights=CHANNEL_W, k=1)[0],
        # --- equipment and consumables
        "instrument_count": rng.choices([1, 2, 3, 4], weights=[0.52, 0.28, 0.13, 0.07], k=1)[0],
        "assay_count": _u(rng, 3, 18),
        "hardware_outlay_k": hardware_outlay_k,
        "consumable_investment_k": _u(rng, 4, 30),
        "target_breakeven_months": target_breakeven,
        "annual_reagent_k": annual_reagent_k,
        "reagent_margin": reagent_margin,
        # --- commercial
        "lease_months": lease_months,
        "list_price_k": round(hardware_outlay_k * rng.uniform(1.15, 1.55), 1),
        "discount_pct": _uf(rng, 0.04, 0.38, 3),
        "drp_vs_threshold": _uf(rng, *p["drp_vs_threshold"]),
        "worst_line_margin": _uf(rng, *p["worst_line_margin"]),
        "margin_floor": MARGIN_FLOOR,
        "customer_demands_reduction": rng.random() < p["demands_reduction_p"],
        # --- competitive
        "competitor_present": competitor_present,
        "competitor_gtm": (
            rng.choices(COMPETITOR_GTM, weights=COMPETITOR_GTM_W, k=1)[0]
            if competitor_present else None
        ),
        "competitor_bundling": competitor_bundling,
        "installed_base": installed_base,
        "alt_config_available": rng.random() < 0.46,
        "higher_margin_assay_available": rng.random() < 0.52,
        # --- derived
        "annual_reagent_profit_k": annual_reagent_profit_k,
        "contract_reagent_revenue_k": contract_reagent_revenue_k,
        "breakeven_months": breakeven_months,
        "roi_pct": roi_pct,
        "deal_margin_pct": deal_margin_pct,
    }

    # Which fields the written-up document will omit, so the extractor must abstain
    omitted = [f for f, prob in OMISSION_PROB.items() if rng.random() < prob]
    deal["omitted_in_document"] = omitted
    return deal


def coverage_report(deals):
    """
    Prove every lever has enough examples BEFORE any model work starts.
    A lever with nine examples is untrainable and untestable; better to find
    that out here than at evaluation time.
    """
    n = len(deals)
    conds = {
        "L1 drp below threshold": lambda d: d["drp_vs_threshold"] < 1.0,
        "L2 line below margin floor": lambda d: d["worst_line_margin"] < d["margin_floor"],
        "L3 customer demands reduction": lambda d: d["customer_demands_reduction"],
        "L4 config or assay alternative": lambda d: d["alt_config_available"] or d["higher_margin_assay_available"],
        "L5/L12 competitor present": lambda d: d["competitor_present"],
        "L9 drp negative": lambda d: d["drp_vs_threshold"] < 0,
        "L11 installed base": lambda d: d["installed_base"],
        "L13 competitor bundling": lambda d: d["competitor_bundling"],
        "L14 breakeven outside term": lambda d: d["breakeven_months"] > d["lease_months"],
    }
    return {k: {"count": sum(1 for d in deals if f(d)),
                "share": round(sum(1 for d in deals if f(d)) / n, 3)}
            for k, f in conds.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    deals = [make_deal(rng, i + 1) for i in range(args.n)]

    with (args.out / "deals.jsonl").open("w", encoding="utf-8") as f:
        for d in deals:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    card = {
        "stage": "1_deal_generation",
        "seed": args.seed,
        "n_deals": len(deals),
        "n_features_per_deal": len(deals[0]) - 2,  # excl. deal_id and archetype bookkeeping
        "declared_archetype_mix": ARCHETYPES,
        "realised_archetype_mix": {
            k: round(v / len(deals), 3)
            for k, v in sorted(Counter(d["archetype"] for d in deals).items())
        },
        "declared_field_ranges_by_archetype": ARCHETYPE_PARAMS,
        "declared_omission_probability": OMISSION_PROB,
        "margin_floor_policy": MARGIN_FLOOR,
        "derived_fields": [
            "annual_reagent_profit_k = annual_reagent_k * reagent_margin",
            "breakeven_months = ceil(hardware_outlay_k / annual_reagent_profit_k * 12)",
            "contract_reagent_revenue_k = annual_reagent_k * lease_months / 12",
            "roi_pct = (contract_reagent_profit_k - hardware_outlay_k) / hardware_outlay_k * 100",
        ],
        "lever_coverage": coverage_report(deals),
        "limitations": [
            "Archetype-conditioned sampling produces a more balanced and more "
            "internally coherent population than a real pipeline, where most "
            "deals are unremarkable and hard cases are rare.",
            "Feature values are noise-free. Real deal write-ups contain stale "
            "figures, disputed margins, and numbers that disagree between pages.",
            "Policy constants (margin floor, threshold) are illustrative, not "
            "any company's actual thresholds.",
            "A single margin floor applies to all segments; in reality floors "
            "differ by segment, channel and region.",
        ],
    }
    (args.out / "deals_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(f"wrote {len(deals)} deals, {card['n_features_per_deal']} features each")
    print("\nrealised archetype mix:")
    for k, v in card["realised_archetype_mix"].items():
        print(f"  {k:<20} {v}")
    print("\nlever coverage:")
    for k, v in card["lever_coverage"].items():
        print(f"  {k:<32} {v['count']:>5}  ({v['share']})")


if __name__ == "__main__":
    main()
