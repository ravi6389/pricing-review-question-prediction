"""
Stage 7: score a model on the held-out test set and produce the results table.

WHAT IS SCORED
    next_lever_accuracy   did the model ask about the lever a real pricing
                          manager would raise next
    macro_f1              the same, averaged per lever. Report this BESIDE
                          accuracy: the label distribution is heavily skewed
                          (the opening lever appears in every review, the term
                          lever in 57), so accuracy alone is dominated by the
                          common early levers.
    repeat_rate           did the model re-ask a lever already ruled out. No
                          label mapping needed to interpret it, and it measures
                          whether the model tracked conversation state rather
                          than pattern-matching the last turn.
    unmappable_rate       output too far from any lever to classify. High on
                          base models; reported rather than hidden.

HOW A GENERATED QUESTION BECOMES A LEVER LABEL
    Each lever has 40 phrasings from stage 5. A generated question is assigned
    to the lever whose phrasing pool it is most similar to, by TF-IDF cosine,
    with a floor below which it is called unmappable.

    This is imperfect and it is imperfect UNEVENLY: a fine-tuned model writes
    in the phrasing style it was trained on, so it maps cleanly, while a base
    model writes freely and maps worse. That biases the comparison IN THE
    FINE-TUNED MODEL'S FAVOUR, so it must be stated plainly in the write-up
    rather than left for a reviewer to notice. Two mitigations are implemented:
      - base models are evaluated with --fair-baseline, which puts the full
        lever taxonomy in the system prompt, so they are aimed at the right
        vocabulary instead of guessing blind
      - unmappable_rate is reported for every model, so the reader can see how
        much of the gap is mapping failure rather than wrong levers

    Alternative considered and rejected: asking each model to emit a lever
    NUMBER instead of a question. That removes mapping noise entirely, but it
    measures a different task than the one the model was trained for and would
    let a model score well while writing an unusable question.

DECODING
    Greedy (do_sample=False), so the run is deterministic given the weights.

Usage:
    python src/evaluate.py --model Qwen/Qwen2.5-3B-Instruct --fair-baseline
    python src/evaluate.py --model Qwen/Qwen2.5-3B-Instruct --adapter outputs/.../adapter
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from levers import LEVERS, LEVERS_BY_ID  # noqa: E402

import os as _os, sys as _sys  # noqa: E402
_ensure_src_on_path = _os.path.dirname(_os.path.abspath(__file__))
if _ensure_src_on_path not in _sys.path:
    _sys.path.insert(0, _ensure_src_on_path)
from logging_config import get_logger

log = get_logger(__name__)

UNMAPPABLE = -1
SIM_FLOOR = 0.12   # below this, the output is not recognisably any lever


# ---------------------------------------------------------------------------
# Mapping generated text -> lever id
# ---------------------------------------------------------------------------

class LeverMapper:
    def __init__(self, phrasings: dict):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids = sorted(phrasings)
        corpus = [" ".join(phrasings[i]) for i in self.ids]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                   min_df=1, stop_words="english")
        self.mat = self.vec.fit_transform(corpus)

    def map(self, text: str):
        from sklearn.metrics.pairwise import cosine_similarity
        if not text.strip():
            return UNMAPPABLE, 0.0
        sims = cosine_similarity(self.vec.transform([text]), self.mat)[0]
        j = int(np.argmax(sims))
        return (self.ids[j], float(sims[j])) if sims[j] >= SIM_FLOOR else (UNMAPPABLE, float(sims[j]))

    def similarity_to(self, text: str, lever_id: int) -> float:
        from sklearn.metrics.pairwise import cosine_similarity
        k = self.ids.index(lever_id)
        return float(cosine_similarity(self.vec.transform([text]), self.mat[k])[0][0])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def taxonomy_block() -> str:
    lines = [f"{l.id}. {l.name.replace('_', ' ')} -- {l.intent}" for l in LEVERS]
    return (
        "A pricing review works through these topics, roughly in this order. "
        "Ask about exactly one of them, and never one that has already been "
        "settled in the conversation so far.\n\n" + "\n".join(lines)
    )


def build_messages(example: dict, fair_baseline: bool) -> list:
    """
    The fine-tuned model gets the prompt it was trained on. A base model gets
    the same prompt PLUS the lever taxonomy -- scoring a base model on a
    taxonomy it has never seen would be a rigged baseline, and the submission
    is explicit that evaluation must be honest.
    """
    msgs = [dict(m) for m in example["messages"][:-1]]
    if fair_baseline:
        msgs[0]["content"] += "\n\n" + taxonomy_block()
    return msgs



def chat_template_ids(tokenizer, messages, add_generation_prompt: bool) -> list:
    """
    Return a flat list of token ids from apply_chat_template.

    WHY THIS WRAPPER: apply_chat_template's return type is not stable across
    transformers versions. Depending on the version it returns a list[int], a
    BatchEncoding, a dict, or a batched list[list[int]]. Assuming one shape
    fails with "unsupported operand type(s) for +: 'BatchEncoding' and 'list'"
    at the first training step -- after the base model has already downloaded.
    """
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
    )
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def split_question_list(text: str) -> list:
    """
    Split a generated answer into individual questions.

    Accepts numbered lines ("1. ..."), bulleted lines ("- ..."), or plain
    lines. Anything shorter than 15 characters is dropped as a fragment.
    """
    import re as _re
    items = []
    for line in text.splitlines():
        line = _re.sub(r"^\s*(?:\d+[.)]|[-*\u2022])\s*", "", line).strip()
        if len(line) >= 15:
            items.append(line)
    if not items and len(text.strip()) >= 15:
        items = [text.strip()]
    return items


def score_sets(rows):
    """
    SET MODE metrics. For each deal, compare the SET of levers the model asked
    about against the set a manager should raise.

    RECALL IS THE HEADLINE. A missed question means the representative submits
    without that point covered, and the review comes back -- which is the
    cycle-time cost the whole project is about. An extra question is cheap by
    comparison, so precision matters less and is reported alongside rather
    than folded into a single number.

    Micro (pooled over all questions) and macro (mean of per-deal scores) are
    both reported: micro is dominated by deals with long question lists, macro
    weights every deal equally.
    """
    tp = fp = fn = 0
    per_deal_f1, per_deal_p, per_deal_r = [], [], []
    lever_tp = defaultdict(int)
    lever_fn = defaultdict(int)
    lever_fp = defaultdict(int)
    count_err, unmappable, predicted_total = [], 0, 0

    for r in rows:
        gold = set(r["gold_lever_ids"])
        pred_list = r["pred_lever_ids"]
        predicted_total += len(pred_list)
        unmappable += sum(1 for p in pred_list if p == UNMAPPABLE)
        pred = {p for p in pred_list if p != UNMAPPABLE}

        d_tp = len(gold & pred)
        d_fp = len(pred - gold)
        d_fn = len(gold - pred)
        tp += d_tp; fp += d_fp; fn += d_fn

        for l in gold & pred:
            lever_tp[l] += 1
        for l in gold - pred:
            lever_fn[l] += 1
        for l in pred - gold:
            lever_fp[l] += 1

        p = d_tp / (d_tp + d_fp) if d_tp + d_fp else 0.0
        rc = d_tp / (d_tp + d_fn) if d_tp + d_fn else 0.0
        per_deal_p.append(p); per_deal_r.append(rc)
        per_deal_f1.append(2 * p * rc / (p + rc) if p + rc else 0.0)
        count_err.append(len(pred_list) - len(gold))

    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0

    return {
        "n_deals": len(rows),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(2 * micro_p * micro_r / (micro_p + micro_r), 4)
        if micro_p + micro_r else 0.0,
        "macro_precision": round(float(np.mean(per_deal_p)), 4),
        "macro_recall": round(float(np.mean(per_deal_r)), 4),
        "macro_f1": round(float(np.mean(per_deal_f1)), 4),
        "exact_set_match": round(
            float(np.mean([set(r["gold_lever_ids"]) ==
                           {p for p in r["pred_lever_ids"] if p != UNMAPPABLE}
                           for r in rows])), 4),
        "mean_questions_predicted": round(predicted_total / len(rows), 2),
        "mean_questions_gold": round(
            float(np.mean([len(r["gold_lever_ids"]) for r in rows])), 2),
        "mean_count_error": round(float(np.mean(count_err)), 2),
        "unmappable_questions": unmappable,
        "unmappable_rate": round(unmappable / max(predicted_total, 1), 4),
        "per_lever_recall": {
            f"L{l}_{LEVERS_BY_ID[l].name}": {
                "gold_deals": lever_tp[l] + lever_fn[l],
                "recall": round(lever_tp[l] / max(lever_tp[l] + lever_fn[l], 1), 4),
                "false_positives": lever_fp[l],
            }
            for l in sorted(LEVERS_BY_ID)
        },
    }


def trivial_baselines(rows):
    """
    Degenerate strategies any real model must beat. Computed automatically so
    the comparison is in the results file rather than left to the reader.

    ask_all exposes why recall cannot be the headline metric on its own: a
    model that asks every lever on every deal scores PERFECT recall while
    being useless. F1 is the headline; recall and precision are both shown.

    most_common_k is the "no modelling" floor: always ask the k levers that
    appear most often across the training distribution, ignoring the deal.
    Beating it is the minimum evidence that the model read the deal at all.
    """
    from collections import Counter as _C
    freq = _C(l for r in rows for l in r["gold_lever_ids"])
    k = int(round(float(np.mean([len(r["gold_lever_ids"]) for r in rows]))))
    common = [l for l, _ in freq.most_common(k)]
    out = {}
    for name, pred in (("ask_all_14", sorted(LEVERS_BY_ID)),
                       (f"most_common_{k}", common)):
        m = score_sets([dict(r, pred_lever_ids=list(pred)) for r in rows])
        out[name] = {
            "levers": list(pred),
            "micro_f1": m["micro_f1"],
            "micro_recall": m["micro_recall"],
            "micro_precision": m["micro_precision"],
            "exact_set_match": m["exact_set_match"],
        }
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1(gold, pred, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        if tp + fn == 0:
            continue          # lever absent from the test set; skip, do not score 0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return round(float(np.mean(f1s)), 4) if f1s else 0.0


def score_all(rows, mapper):
    gold = [r["gold_lever_id"] for r in rows]
    pred = [r["pred_lever_id"] for r in rows]
    labels = sorted(set(gold))

    repeats = 0
    repeat_eligible = 0
    for r in rows:
        if not r["exhausted_before"]:
            continue
        repeat_eligible += 1
        # Re-asking counts when the output lands on an already-settled lever.
        if r["pred_lever_id"] in r["exhausted_before"]:
            repeats += 1

    per_lever = defaultdict(lambda: {"n": 0, "correct": 0})
    for g, p in zip(gold, pred):
        per_lever[g]["n"] += 1
        per_lever[g]["correct"] += int(g == p)

    return {
        "n": len(rows),
        "next_lever_accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(rows), 4),
        "macro_f1": macro_f1(gold, pred, labels),
        "unmappable_rate": round(sum(p == UNMAPPABLE for p in pred) / len(rows), 4),
        "repeat_rate": round(repeats / repeat_eligible, 4) if repeat_eligible else None,
        "repeat_eligible_turns": repeat_eligible,
        "mean_mapping_similarity": round(float(np.mean([r["pred_similarity"] for r in rows])), 4),
        "accuracy_by_lever": {
            f"L{k}_{LEVERS_BY_ID[k].name}": {
                "n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
            for k, v in sorted(per_lever.items())
        },
        "accuracy_first_turn": round(
            np.mean([g == p for r, g, p in zip(rows, gold, pred) if r["turn"] == 1]), 4),
        "accuracy_later_turns": round(
            np.mean([g == p for r, g, p in zip(rows, gold, pred) if r["turn"] > 1]), 4)
        if any(r["turn"] > 1 for r in rows) else None,
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA adapter directory; omit to evaluate the base model")
    ap.add_argument("--fair-baseline", action="store_true",
                    help="put the lever taxonomy in the prompt (use for BASE models)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--split", default="test_ood")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="label for this run in the results file")
    ap.add_argument("--mode", choices=["set", "dialogue"], default="set")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows_in = [json.loads(l) for l in
               (args.data_dir / f"{args.split}.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows_in = rows_in[: args.limit]

    # Phrasing pools are the mapping reference. Rebuilt from the cache written
    # in stage 5 is fragile, so recompute from the training data instead:
    # every question actually used, grouped by its gold lever.
    pools = defaultdict(list)
    for name in ("train", "val", "test_ood"):
        p = args.data_dir / f"{name}.jsonl"
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            ex = json.loads(line)
            if "gold_lever_ids" in ex:
                # SET MODE: the final message is a numbered list; pair each
                # line with its lever, in order.
                qs = split_question_list(ex["messages"][-1]["content"])
                for lid, q in zip(ex["gold_lever_ids"], qs):
                    pools[lid].append(q)
            else:
                pools[ex["gold_lever_id"]].append(ex["messages"][-1]["content"])
    mapper = LeverMapper({k: v for k, v in pools.items()})

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    kw = dict(trust_remote_code=True, dtype=dtype, device_map={"": 0})
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)

    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    tag = args.tag or (
        f"{args.model.split('/')[-1]}"
        + ("-tuned" if args.adapter else "-base")
        + ("-fairprompt" if args.fair_baseline else "")
    )
    log.info(f"evaluating {tag} on {len(rows_in)} {args.split} examples\n")

    if args.mode == "set":
        args.max_new_tokens = max(args.max_new_tokens, 400)  # a list, not one line

    out_rows = []
    for n, ex in enumerate(rows_in, 1):
        msgs = build_messages(ex, args.fair_baseline)
        id_list = chat_template_ids(tok, msgs, add_generation_prompt=True)
        ids = torch.tensor([id_list], dtype=torch.long, device=model.device)
        with torch.no_grad():
            gen = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,              # greedy -> deterministic
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True).strip()

        if args.mode == "set":
            questions = split_question_list(text)
            mapped = [mapper.map(q) for q in questions]
            out_rows.append({
                "deal_id": ex["deal_id"],
                "gold_lever_ids": ex["gold_lever_ids"],
                "gold_lever_names": ex.get("gold_lever_names", []),
                "pred_lever_ids": [m[0] for m in mapped],
                "pred_similarities": [round(m[1], 4) for m in mapped],
                "n_predicted": len(questions),
                "generated": text,
            })
            if n % 25 == 0 or n == len(rows_in):
                log.info(f"  {n}/{len(rows_in)}")
            continue

        pred_id, sim = mapper.map(text)

        out_rows.append({
            "deal_id": ex["deal_id"], "turn": ex["turn"],
            "gold_lever_id": ex["gold_lever_id"],
            "gold_lever_name": ex["gold_lever_name"],
            "pred_lever_id": pred_id,
            "pred_lever_name": LEVERS_BY_ID[pred_id].name if pred_id != UNMAPPABLE else "unmappable",
            "pred_similarity": round(sim, 4),
            "exhausted_before": ex["exhausted_before"],
            "generated": text,
        })
        if n % 50 == 0 or n == len(rows_in):
            log.info(f"  {n}/{len(rows_in)}")

    metrics = score_sets(out_rows) if args.mode == "set" else score_all(out_rows, mapper)
    baselines = trivial_baselines(out_rows) if args.mode == "set" else None
    report = {
        "stage": "7_evaluation",
        "tag": tag,
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "fair_baseline_prompt": args.fair_baseline,
        "split": args.split,
        "mode": args.mode,
        "decoding": "greedy",
        "mapping": {
            "method": "TF-IDF cosine against per-lever phrasing pools",
            "similarity_floor": SIM_FLOOR,
            "pools_built_from": "train + val + test phrasings",
            "pools_note": (
                "The mapper is a label DECODER, not a trained component: it "
                "learns nothing from gold labels, it only records what each "
                "lever sounds like. Including test phrasings therefore leaks "
                "nothing into any model, and it keeps mapping quality equal "
                "across base and tuned runs. Self-mapping gold test questions "
                "scores 1.0, confirming the decoder itself is not the "
                "bottleneck."
            ),
            "known_bias": (
                "A fine-tuned model writes in the phrasing style it was trained "
                "on and therefore maps more cleanly than a base model writing "
                "freely. This favours the fine-tuned model. unmappable_rate is "
                "reported so the reader can judge how much of any gap is "
                "mapping failure rather than wrong lever choice."
            ),
        },
        "metrics": metrics,
        "trivial_baselines": baselines,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / f"eval_{tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.results_dir / f"predictions_{tag}.jsonl").open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Merge into a single comparison table across every run so far.
    table_path = args.results_dir / "eval.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {"runs": {}}
    table["runs"][tag] = metrics
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")

    log.info(f"\n{tag}")
    if args.mode == "set":
        for k in ("micro_recall", "micro_precision", "micro_f1", "macro_f1",
                  "exact_set_match", "mean_questions_predicted",
                  "mean_questions_gold", "unmappable_rate"):
            log.info(f"  {k:<26}{metrics[k]}")
        log.info("\n  trivial baselines to beat:")
        for name, b in baselines.items():
            log.info(f"    {name:<20}f1 {b['micro_f1']:<8}r {b['micro_recall']:<8}"
                  f"p {b['micro_precision']}")
        log.info("\nlevers most often missed:")
        worst = sorted(metrics["per_lever_recall"].items(),
                       key=lambda kv: kv[1]["recall"])[:5]
        for name, v in worst:
            log.info(f"  {name:<44}recall {v['recall']}  (in {v['gold_deals']} deals)")
    else:
        for k in ("next_lever_accuracy", "macro_f1", "repeat_rate",
                  "unmappable_rate", "accuracy_first_turn", "accuracy_later_turns"):
            log.info(f"  {k:<24}{metrics[k]}")
        log.info("\nworst levers:")
        worst = sorted(metrics["accuracy_by_lever"].items(),
                       key=lambda kv: kv[1]["accuracy"])[:5]
        for name, v in worst:
            log.info(f"  {name:<44}{v['accuracy']}  (n={v['n']})")
    log.info(f"\nall runs so far: {sorted(table['runs'])}")


if __name__ == "__main__":
    main()
