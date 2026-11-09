"""
Measure validation loss for a BASE model, before any fine-tuning.

Answers "how surprised is this model by our house-style questions, untouched?"
and gives the before-number that the capability-lift claim is measured against.

WHAT LOSS HERE CAN AND CANNOT COMPARE

  Within one model family -- Qwen2.5 0.5B / 1.5B / 3B / 7B -- the tokenizer is
  identical, so loss per token measures the same quantity and the numbers form
  a fair scaling curve.

  ACROSS families -- Qwen vs Llama -- it does NOT. Different tokenizers split
  the same text into different numbers of tokens, so per-token loss is not a
  like-for-like quantity and neither is perplexity. Comparing them on loss and
  declaring a winner would be a methodological error.
  For cross-family comparison use the task metric from stage 7 (next-lever
  accuracy), which is tokenizer-independent.

  This script therefore reports mean_tokens_per_example alongside the loss, so
  the reader can see how different the tokenizations are rather than having to
  trust that they are similar.

Loss is computed with exactly the masking used in training: only the final
manager question contributes, not the deal note or the conversation history.

Usage:
    python src/baseline_loss.py --model Qwen/Qwen2.5-0.5B-Instruct
    python src/baseline_loss.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from train import chat_template_ids, set_seed, gpu_profile  # noqa: E402

import os as _os, sys as _sys  # noqa: E402
_ensure_src_on_path = _os.path.dirname(_os.path.abspath(__file__))
if _ensure_src_on_path not in _sys.path:
    _sys.path.insert(0, _ensure_src_on_path)
from logging_config import get_logger

log = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    gpu = gpu_profile()
    log.info(json.dumps(gpu, indent=2))
    if args.load_4bit and not gpu["can_4bit"]:
        raise SystemExit(f"4-bit needs compute capability 7.0+, this GPU is {gpu['capability']}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(l) for l in
            (args.data_dir / f"{args.split}.jsonl").open(encoding="utf-8")]
    # Deterministic stride, so the subsample spans all conversation depths
    # instead of only the earliest reviews in file order.
    if args.limit and len(rows) > args.limit:
        step = max(1, len(rows) // args.limit)
        rows = rows[::step][: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if gpu["bf16"] else torch.float16
    kw = dict(trust_remote_code=True, dtype=dtype, device_map={"": 0})
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)

    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()

    total_loss, total_tokens, total_len, n_trunc = 0.0, 0, 0, 0

    for i, row in enumerate(rows, 1):
        msgs = row["messages"]
        prompt_ids = chat_template_ids(tok, msgs[:-1], add_generation_prompt=True)
        target_ids = tok(msgs[-1]["content"], add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [tok.eos_token_id]

        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(ids) > args.max_len:
            n_trunc += 1
            cut = len(ids) - args.max_len
            ids, labels = ids[cut:], labels[cut:]

        total_len += len(ids)
        t_ids = torch.tensor([ids], device=model.device)
        t_lab = torch.tensor([labels], device=model.device)

        with torch.no_grad():
            out = model(input_ids=t_ids, labels=t_lab)

        # HF returns loss already averaged over the supervised tokens, so
        # weight by that count to get a true corpus mean rather than a mean
        # of per-example means (which would over-weight short questions).
        n_sup = int((t_lab != -100).sum().item()) - 1   # shift for next-token
        if n_sup <= 0:
            continue
        total_loss += float(out.loss.item()) * n_sup
        total_tokens += n_sup

        if i % 50 == 0 or i == len(rows):
            log.info(f"  {i}/{len(rows)}  running loss {total_loss / total_tokens:.4f}")

    mean_loss = total_loss / total_tokens
    report = {
        "model": args.model,
        "quantised_4bit": args.load_4bit,
        "split": args.split,
        "n_examples": len(rows),
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(math.exp(mean_loss), 3),
        "supervised_tokens": total_tokens,
        "mean_tokens_per_example": round(total_len / len(rows), 1),
        "mean_supervised_tokens_per_example": round(total_tokens / len(rows), 1),
        "truncated_examples": n_trunc,
        "gpu": gpu,
        "comparability_warning": (
            "Loss and perplexity are comparable only between models sharing a "
            "tokenizer. Compare across families on next-lever accuracy instead."
        ),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.results_dir / "baseline_loss.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {"models": {}}
    table["models"][args.model] = report
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")

    log.info(f"\n{args.model}")
    log.info(f"  mean loss                    {report['mean_loss']}")
    log.info(f"  perplexity                   {report['perplexity']}")
    log.info(f"  mean tokens / example        {report['mean_tokens_per_example']}")
    log.info(f"  supervised tokens / example  {report['mean_supervised_tokens_per_example']}")
    log.info(f"\nall measured so far:")
    for m, r in sorted(table["models"].items(), key=lambda kv: kv[1]["mean_loss"]):
        log.info(f"  {r['mean_loss']:>8}  ppl {r['perplexity']:>9}  "
              f"tok/ex {r['mean_tokens_per_example']:>7}  {m}")


if __name__ == "__main__":
    main()
