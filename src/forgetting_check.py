"""
Catastrophic forgetting check: did fine-tuning cost the model its general ability?

THE RISK
    The model was trained on one narrow task with a fixed output shape -- a
    numbered list of pricing-review questions. A model that has overfitted to
    that shape stops being a general assistant: ask it to write an email and it
    returns pricing questions anyway. That failure is invisible in task metrics,
    which is exactly why it has to be tested separately.

WHAT IS MEASURED
    task_bleed_rate   on prompts that have NOTHING to do with pricing, how often
                      does the model still emit a numbered list or pricing
                      vocabulary. This is the headline number: 0 means no
                      bleed, 1 means the model answers everything as a pricing
                      review.
    mean_length       collapse in output length is the other common symptom
    refusal_rate      empty or near-empty answers
    side-by-side      base and tuned answers to the same prompt, saved in full
                      so a reader can judge rather than trust the metric

    The prompts are embedded rather than downloaded so the check runs offline
    and is identical for anyone reproducing it.

Usage:
    python src/forgetting_check.py --model Qwen/Qwen2.5-3B-Instruct \
        --adapter outputs/Qwen2.5-3B-Instruct-lora/adapter
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import chat_template_ids, set_seed, gpu_profile  # noqa: E402

# General instructions, deliberately unrelated to pricing reviews. A healthy
# model answers these normally; a model that has forgotten returns questions.
PROMPTS = [
    "Explain photosynthesis to a ten year old.",
    "Write a two sentence apology to a customer whose order arrived late.",
    "What is the difference between a list and a tuple in Python?",
    "Give me three ideas for a team lunch.",
    "Summarise the plot of Romeo and Juliet in three sentences.",
    "Convert 45 degrees Celsius to Fahrenheit and show the working.",
    "What are the main causes of inflation?",
    "Write a haiku about rain.",
    "How do I politely decline a meeting invitation?",
    "Name three capital cities in South America.",
    "What does a product manager do day to day?",
    "Explain the difference between a virus and a bacterium.",
    "Draft a one line out-of-office message for a week's leave.",
    "What is compound interest and why does it matter?",
    "Suggest a simple vegetarian dinner for four people.",
    "How does a refrigerator keep food cold?",
    "Write a short thank-you note to a colleague who covered your shift.",
    "What is the capital of Japan and roughly how many people live there?",
    "Explain what an API is to someone non-technical.",
    "Give me two tips for sleeping better.",
]

# Vocabulary that should NOT appear when answering the prompts above.
TASK_WORDS = re.compile(
    r"\b(reagent|lease|margin|realised price|instrument placement|tender|"
    r"pull-through|rebate|break-even|assay|discount|competitor bundl|"
    r"installed base|pricing review|deal)\b", re.I)
NUMBERED_LIST = re.compile(r"^\s*1[.)]\s+\S", re.M)


def generate(model, tok, prompt, max_new_tokens=160):
    ids = chat_template_ids(tok, [{"role": "user", "content": prompt}],
                            add_generation_prompt=True)
    t = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][t.shape[-1]:], skip_special_tokens=True).strip()


def score(answers):
    n = len(answers)
    bled = [a for a in answers if NUMBERED_LIST.search(a) or TASK_WORDS.search(a)]
    return {
        "n_prompts": n,
        "task_bleed_rate": round(len(bled) / n, 4),
        "numbered_list_rate": round(
            sum(1 for a in answers if NUMBERED_LIST.search(a)) / n, 4),
        "pricing_vocab_rate": round(
            sum(1 for a in answers if TASK_WORDS.search(a)) / n, 4),
        "mean_chars": round(sum(len(a) for a in answers) / n, 1),
        "empty_rate": round(sum(1 for a in answers if len(a.strip()) < 20) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    gpu = gpu_profile()
    dtype = torch.bfloat16 if gpu["bf16"] else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("loading base...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype, device_map={"": 0})
    base.eval()
    base_answers = [generate(base, tok, p) for p in PROMPTS]
    print("base done", flush=True)

    print("attaching adapter...", flush=True)
    tuned = PeftModel.from_pretrained(base, str(args.adapter))
    tuned.eval()
    tuned_answers = [generate(tuned, tok, p) for p in PROMPTS]
    print("tuned done", flush=True)

    report = {
        "model": args.model,
        "adapter": str(args.adapter),
        "base": score(base_answers),
        "tuned": score(tuned_answers),
        "side_by_side": [
            {"prompt": p, "base": b, "tuned": t}
            for p, b, t in zip(PROMPTS, base_answers, tuned_answers)
        ],
        "how_to_read": (
            "task_bleed_rate is the headline: on prompts with nothing to do "
            "with pricing, how often the model still answers like a pricing "
            "review. A large rise from base to tuned means the adapter has "
            "overwritten general behaviour. A small rise with similar "
            "mean_chars means the adapter is task-specific without having "
            "damaged the base model."
        ),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    (args.results_dir / f"forgetting_{tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'metric':<22}{'base':>10}{'tuned':>10}")
    for k in ("task_bleed_rate", "numbered_list_rate", "pricing_vocab_rate",
              "mean_chars", "empty_rate"):
        print(f"{k:<22}{report['base'][k]:>10}{report['tuned'][k]:>10}")
    print("\n--- example: 'Write a haiku about rain.' ---")
    i = PROMPTS.index("Write a haiku about rain.")
    print("BASE :", base_answers[i][:300].replace("\n", " / "))
    print("TUNED:", tuned_answers[i][:300].replace("\n", " / "))


if __name__ == "__main__":
    main()
