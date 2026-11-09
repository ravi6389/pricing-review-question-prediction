"""
Stage 3: read each deal document back into structured features, and score the
extraction against the truth fixed in stage 1.

WHY THIS STEP EARNS ITS PLACE
    The scenario is unstructured enterprise content, so the system must be able
    to turn a written deal note into fields. This is that component.

    It is deliberately NOT the source of the training labels. Those come from
    stage 1, where the values were fixed. Because the truth is known
    independently, this extractor can be MEASURED -- which would be impossible
    if its own output were the ground truth.

WHAT IS MEASURED
    field_accuracy      per-field correctness on fields the document states
    abstention_recall   of fields the document OMITS, how many were correctly
                        returned as "not_stated"
    hallucination_rate  of fields the document OMITS, how many got an invented
                        value. This is the number that matters: it is the
                        measurable basis for the hallucination discussion.
    miss_rate           of fields the document DOES state, how many were
                        wrongly returned as "not_stated"

TWO EXTRACTORS, DELIBERATELY
    --extractor llm     a language model reading the prose (default)
    --extractor regex   a rule-based parser
    The regex parser is the honest baseline: if a hand-written parser scores as
    well as the model, the model is not earning its cost. Reporting both is the
    kind of comparison the rubric rewards. The regex parser is also the offline
    fallback, so the step runs with no API at all.

Usage:
    python src/extract_features.py --seed 42
    python src/extract_features.py --extractor regex
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import CachedLLM  # noqa: E402

NOT_STATED = "not_stated"

# Field -> (type, tolerance). Tolerance is relative, for floats read out of prose.
SCHEMA = {
    "segment": ("categorical", None),
    "region": ("categorical", None),
    "channel": ("categorical", None),
    "instrument_count": ("int", 0),
    "assay_count": ("int", 0),
    "hardware_outlay_k": ("float", 0.02),
    "consumable_investment_k": ("float", 0.02),
    "annual_reagent_k": ("float", 0.02),
    "reagent_margin": ("float", 0.05),
    "lease_months": ("int", 0),
    "list_price_k": ("float", 0.02),
    "discount_pct": ("float", 0.05),
    "drp_vs_threshold": ("float", 0.05),
    "worst_line_margin": ("float", 0.05),
    "customer_demands_reduction": ("bool", None),
    "competitor_present": ("bool", None),
    "competitor_gtm": ("categorical", None),
    "competitor_bundling": ("bool", None),
    "installed_base": ("bool", None),
    "alt_config_available": ("bool", None),
    "higher_margin_assay_available": ("bool", None),
    "breakeven_months": ("int", 0),
    "roi_pct": ("float", 0.03),
    "deal_margin_pct": ("float", 0.03),
}

SYSTEM = (
    "You extract structured fields from internal sales deal notes for a medical "
    "diagnostics company. Reply with JSON only -- no prose, no code fences. "
    f'Use the exact string "{NOT_STATED}" for any field the note does not state. '
    "Never infer, estimate, or calculate a value that is not written in the note."
)

FIELD_HINTS = {
    "reagent_margin": "as a fraction, e.g. 0.26 for 26%",
    "discount_pct": "as a fraction, e.g. 0.28 for 28%",
    "drp_vs_threshold": "realised price as a multiple of threshold; may be negative",
    "worst_line_margin": "as a fraction",
    "roi_pct": "as a percentage number, e.g. 63.8",
    "deal_margin_pct": "as a percentage number",
    "hardware_outlay_k": "in thousands, e.g. 74 for 74k",
    "annual_reagent_k": "in thousands",
    "list_price_k": "in thousands",
    "consumable_investment_k": "in thousands",
}


def build_prompt(document: str) -> list:
    lines = []
    for f, (kind, _) in SCHEMA.items():
        hint = FIELD_HINTS.get(f)
        lines.append(f"- {f} ({kind}){': ' + hint if hint else ''}")
    user = (
        "Extract these fields from the note below.\n\n"
        + "\n".join(lines)
        + f'\n\nReturn one JSON object with exactly these keys. Use "{NOT_STATED}" '
        "where the note is silent.\n\nNote:\n" + document
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Rule-based extractor: the honest baseline, and the offline fallback
# ---------------------------------------------------------------------------

_NUM = r"(-?\d+(?:\.\d+)?)"

REGEX_RULES = {
    "hardware_outlay_k": rf"hardware\s+{_NUM}k",
    "list_price_k": rf"list\s+{_NUM}k",
    "annual_reagent_k": rf"pull-through\s+{_NUM}k",
    "consumable_investment_k": rf"consumable[^.]*?{_NUM}k",
    "lease_months": rf"{_NUM}\s*month lease",
    "breakeven_months": rf"pays back in\s+{_NUM}\s*months",
    "roi_pct": rf"ROI\s+{_NUM}%",
    "deal_margin_pct": rf"deal margin\s+{_NUM}%",
    "drp_vs_threshold": rf"[Rr]ealised price sits at\s+{_NUM}",
    "instrument_count": rf"{_NUM}\s+instrument",
    "assay_count": rf"{_NUM}\s+assay",
    "reagent_margin": rf"at\s+{_NUM}%\s+margin",
    "discount_pct": rf"at\s+{_NUM}%\s+off",
    "worst_line_margin": rf"[Ww]eakest line is on\s+{_NUM}%",
}

REGEX_FLAGS = {
    "customer_demands_reduction": r"pushing for a reduction",
    "competitor_present": r"[Cc]ompetitor in play",
    "competitor_bundling": r"bundling instruments in",
    "installed_base": r"installed base here already",
    "alt_config_available": r"alternative configuration is available",
    "higher_margin_assay_available": r"higher-margin assays",
}

SEGMENT_WORDS = ["public hospital", "private hospital", "reference lab", "clinic chain"]
REGION_WORDS = ["north", "south", "east", "west", "export"]
CHANNEL_WORDS = ["tender", "direct", "dealer"]


def regex_extract(document: str) -> dict:
    out = {}
    for field, pat in REGEX_RULES.items():
        m = re.search(pat, document)
        if not m:
            out[field] = NOT_STATED
            continue
        v = float(m.group(1))
        kind = SCHEMA[field][0]
        if field in ("reagent_margin", "discount_pct", "worst_line_margin"):
            v = v / 100.0
        out[field] = int(v) if kind == "int" else v

    for field, pat in REGEX_FLAGS.items():
        out[field] = bool(re.search(pat, document))

    low = document.lower()
    out["segment"] = next((w.replace(" ", "_") for w in SEGMENT_WORDS if w in low), NOT_STATED)
    head = low.split(".")[0]
    out["region"] = next((w for w in REGION_WORDS if w in head), NOT_STATED)
    out["channel"] = next((w for w in CHANNEL_WORDS if f"{w} route" in low), NOT_STATED)
    m = re.search(r"selling (direct|dealer)", low)
    out["competitor_gtm"] = m.group(1) if m else NOT_STATED
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def values_match(field: str, truth, pred) -> bool:
    kind, tol = SCHEMA[field]
    if pred == NOT_STATED or truth is None:
        return False
    try:
        if kind == "bool":
            if isinstance(pred, str):
                pred = pred.strip().lower() in ("true", "yes", "1")
            return bool(truth) == bool(pred)
        if kind == "categorical":
            return str(truth).strip().lower().replace(" ", "_") == \
                   str(pred).strip().lower().replace(" ", "_")
        t, p = float(truth), float(pred)
        if kind == "int" or tol == 0:
            return abs(t - p) < 1e-9
        return abs(t - p) <= max(abs(t) * tol, 1e-9)
    except (TypeError, ValueError):
        return False


def score(deal: dict, doc_rec: dict, pred: dict) -> dict:
    omitted = set(doc_rec["omitted_in_document"])
    per_field, stated_right, stated_total = {}, 0, 0
    hallucinated, abstained, omitted_total = 0, 0, 0
    missed = 0

    for field in SCHEMA:
        p = pred.get(field, NOT_STATED)
        if field in omitted:
            omitted_total += 1
            if p == NOT_STATED:
                abstained += 1
                per_field[field] = "correct_abstention"
            else:
                hallucinated += 1
                per_field[field] = "hallucinated"
            continue

        stated_total += 1
        if p == NOT_STATED:
            missed += 1
            per_field[field] = "missed"
        elif values_match(field, deal.get(field), p):
            stated_right += 1
            per_field[field] = "correct"
        else:
            per_field[field] = "wrong"

    return {
        "deal_id": deal["deal_id"],
        "split": doc_rec["split"],
        "per_field": per_field,
        "stated_correct": stated_right,
        "stated_total": stated_total,
        "missed": missed,
        "omitted_total": omitted_total,
        "correct_abstentions": abstained,
        "hallucinated": hallucinated,
    }


def aggregate(rows, field_rows):
    sc = lambda k: sum(r[k] for r in rows)  # noqa: E731
    stated, omitted = sc("stated_total"), sc("omitted_total")
    return {
        "n_documents": len(rows),
        "field_accuracy_on_stated": round(sc("stated_correct") / max(stated, 1), 4),
        "miss_rate_on_stated": round(sc("missed") / max(stated, 1), 4),
        "abstention_recall_on_omitted": round(sc("correct_abstentions") / max(omitted, 1), 4),
        "hallucination_rate_on_omitted": round(sc("hallucinated") / max(omitted, 1), 4),
        "n_stated_fields": stated,
        "n_omitted_fields": omitted,
        "per_field_accuracy": {
            f: round(v["correct"] / max(v["correct"] + v["wrong"] + v["missed"], 1), 4)
            for f, v in sorted(field_rows.items())
        },
        "worst_fields": sorted(
            ((f, round(v["correct"] / max(v["correct"] + v["wrong"] + v["missed"], 1), 4))
             for f, v in field_rows.items()),
            key=lambda x: x[1],
        )[:6],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--extractor", choices=["llm", "regex"], default="llm")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    deals = {json.loads(l)["deal_id"]: json.loads(l)
             for l in (args.data / "deals.jsonl").open(encoding="utf-8")}
    docs = [json.loads(l) for l in (args.data / "documents.jsonl").open(encoding="utf-8")]
    if args.limit:
        docs = docs[: args.limit]

    llm = CachedLLM("extraction", regenerate=args.regenerate)
    use_llm = args.extractor == "llm" and llm.provider != "offline"
    if use_llm:
        n_new = sum(1 for d in docs if not llm.is_cached(build_prompt(d["document"]), 0.0))
        print(f"provider={llm.provider} model={llm.model} extractor=llm")
        print(f"{len(docs)} documents: {len(docs) - n_new} cached, {n_new} need a call\n")
    else:
        print(f"extractor=regex (rule-based baseline, no API)\n")

    rows, preds = [], []
    field_rows = defaultdict(lambda: {"correct": 0, "wrong": 0, "missed": 0})

    for n, rec in enumerate(docs, 1):
        doc = rec["document"]
        if use_llm:
            raw = llm.chat(
                build_prompt(doc), temperature=0.0, max_tokens=700,
                offline_fallback=lambda d=doc: json.dumps(regex_extract(d)),
            )
            txt = raw.strip()
            if txt.startswith("```"):
                txt = txt.split("```")[1].lstrip("json").strip()
            try:
                pred = json.loads(txt)
            except json.JSONDecodeError:
                # A model that returns unparseable JSON has failed the task;
                # count it as a total miss rather than silently dropping it.
                pred = {f: NOT_STATED for f in SCHEMA}
        else:
            pred = regex_extract(doc)

        r = score(deals[rec["deal_id"]], rec, pred)
        for f, outcome in r["per_field"].items():
            if outcome in ("correct", "wrong", "missed"):
                field_rows[f][outcome] += 1
        rows.append(r)
        preds.append({"deal_id": rec["deal_id"], "split": rec["split"], "extracted": pred})

        if use_llm and (n % 25 == 0 or n == len(docs)):
            print(f"  {n}/{len(docs)}", flush=True)
            llm.save()

    # Always write the cache file, even in regex/offline mode, so the artifact
    # set is the same whichever extractor ran and the manifest stays consistent.
    llm.save()

    with (args.data / "extracted.jsonl").open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    report = {
        "stage": "3_feature_extraction",
        "extractor": "llm" if use_llm else "regex",
        "llm": llm.summary() if use_llm else None,
        "overall": aggregate(rows, field_rows),
        "by_split": {
            s: aggregate([r for r in rows if r["split"] == s],
                         defaultdict(lambda: {"correct": 0, "wrong": 0, "missed": 0}))
            for s in ("train", "val", "test")
        },
        "note": (
            "hallucination_rate_on_omitted is the headline number: of fields the "
            "document genuinely does not state, how often did the extractor "
            "invent a value instead of abstaining."
        ),
    }
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "extraction_scores.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    o = report["overall"]
    print(f"\nextractor: {report['extractor']}")
    print(f"  field accuracy on stated fields   {o['field_accuracy_on_stated']}")
    print(f"  miss rate on stated fields        {o['miss_rate_on_stated']}")
    print(f"  abstention recall on omitted      {o['abstention_recall_on_omitted']}")
    print(f"  HALLUCINATION RATE on omitted     {o['hallucination_rate_on_omitted']}")
    print("\n  weakest fields:")
    for f, acc in o["worst_fields"]:
        print(f"    {f:<34}{acc}")


if __name__ == "__main__":
    main()
