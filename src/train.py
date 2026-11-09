"""
Stage 6: fine-tune a small language model to ask the next pricing-review question.

Handles 0.5B through 7B from one script. Quantisation and precision are chosen
from the GPU that is actually present, because getting this wrong is the most
common way a run dies forty minutes in.

    python src/train.py --model Qwen/Qwen2.5-3B-Instruct
    python src/train.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit

KEY DESIGN DECISIONS (each is a write-up line)

1. LOSS IS MASKED TO THE FINAL ASSISTANT TURN ONLY.
   Each example is a conversation ending with the manager's next question. If
   loss were computed over the whole sequence the model would also learn to
   generate the REP's replies and the deal note, which is not the task and
   wastes capacity. Everything before the final question is labelled -100.

2. ADAPTERS ON ALL SEVEN LINEAR PROJECTIONS, NOT JUST ATTENTION.
   target_modules covers q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
   down_proj. The QLoRA paper's finding is that WHICH modules carry adapters
   matters more than the rank: restricting them to attention and freezing the
   feed-forward block measurably underperforms, and the MLP projections hold
   most of the parameters. Attention-only is the common shortcut and it is
   worse at equal cost.

3. RANK 16, ALPHA 32.
   Alpha/rank = 2 is the usual effective scaling. Rank 16 on ~3.4k examples is
   deliberately modest: the task is a decision policy over 14 options plus a
   house phrasing style, not new world knowledge, so capacity is not the
   binding constraint and a larger rank mostly buys overfitting. Rank 8 and 32
   are cheap to sweep if time allows -- record what you find either way.

4. PRECISION CHOSEN FROM THE GPU.
   bf16 needs Ampere (compute capability 8.0+). A Kaggle T4 is Turing (7.5), so
   this script uses fp16 there. 4-bit NF4 is unreliable below 7.0, so the script
   refuses to quantise on a P100 rather than failing cryptically mid-run.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set in the environment by run.py; set here too for notebook use.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def gpu_profile() -> dict:
    if not torch.cuda.is_available():
        return {"name": None, "capability": None, "bf16": False, "can_4bit": False}
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": f"{major}.{minor}",
        "bf16": major >= 8,                     # Ampere or newer
        "can_4bit": major >= 7,                 # Turing or newer; P100 is 6.0
        "total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
    }



# Candidate linear-projection names across model families. Different families
# fuse projections differently:
#   Qwen2.5 / Llama / Gemma : q,k,v,o + gate,up,down   (seven separate)
#   Phi-3                   : qkv_proj, o_proj, gate_up_proj, down_proj (fused)
# Hard-coding one family's names makes PEFT raise "target modules not found"
# on another, so the names are DETECTED from the loaded model instead.
CANDIDATE_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # attention, separate
    "qkv_proj",                                       # attention, fused (Phi-3)
    "gate_proj", "up_proj", "down_proj",              # MLP, separate
    "gate_up_proj",                                   # MLP, fused (Phi-3)
    "Wqkv", "out_proj", "fc1", "fc2",                 # older Phi / GPT-NeoX style
]


def detect_target_modules(model) -> list:
    """
    Return the projection names present in this model, attention AND MLP.

    Whichever naming a family uses, the principle is the same: adapt both the
    attention and the feed-forward projections. The QLoRA finding is that WHICH
    modules carry adapters matters more than the rank, and the MLP block holds
    most of the parameters -- so an attention-only adapter set underperforms at
    equal cost.
    """
    import torch.nn as nn
    present = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or module.__class__.__name__ in (
                "Linear4bit", "Linear8bitLt", "QuantLinear"):
            leaf = name.split(".")[-1]
            if leaf in CANDIDATE_TARGETS:
                present.add(leaf)
    if not present:
        raise SystemExit(
            "\nCould not find any known projection names in this model.\n"
            "Inspect them with:\n"
            "  [n.split('.')[-1] for n,_ in model.named_modules()][:60]\n"
        )
    return sorted(present)


def load_base_model(model_name: str, load_kwargs: dict):
    """
    Load a causal LM, falling back for multimodal checkpoints.

    Gemma 3 at 4B and above ships as a multimodal model and does not load
    through AutoModelForCausalLM on every transformers version. The text tower
    is what we fine-tune, so fall back to the image-text class and use it
    directly -- the chat template still produces text-only batches.
    """
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs), "causal_lm"
    except (ValueError, KeyError, OSError) as first_error:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
            print(f"  note: loaded via AutoModelForImageTextToText "
                  f"(multimodal checkpoint, fine-tuning the text tower)")
            return model, "image_text_to_text"
        except Exception:
            raise SystemExit(
                f"\nCould not load {model_name}.\nFirst error: {first_error}\n")


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


# ---------------------------------------------------------------------------
# Dataset: mask everything except the final assistant turn
# ---------------------------------------------------------------------------

class NextQuestionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int):
        self.rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len
        self.n_truncated = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        context, target = msgs[:-1], msgs[-1]["content"]

        prompt_ids = chat_template_ids(self.tok, context, add_generation_prompt=True)
        target_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tok.eos_token_id]

        input_ids = prompt_ids + target_ids
        # -100 means "no loss here": the deal note, the history and the rep's
        # replies are context, not something the model should learn to produce.
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > self.max_len:
            # Trim from the LEFT so the final question is always kept intact;
            # truncating the right would cut the thing we are training on.
            self.n_truncated += 1
            overflow = len(input_ids) - self.max_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


# ---------------------------------------------------------------------------


def build_training_args(TrainingArguments, wanted: dict):
    """
    Construct TrainingArguments against whatever transformers version is
    installed, instead of assuming one.

    WHY: TrainingArguments' signature changes between major transformers
    releases -- `evaluation_strategy` became `eval_strategy`, and some
    arguments have been removed outright. A hosted environment such as Kaggle
    upgrades underneath you, so a hard-coded call fails with
    "unexpected keyword argument" after the base model has already downloaded.

    Known renames are remapped; anything still unsupported is dropped and
    REPORTED, never dropped silently -- a quietly ignored learning-rate
    schedule would change results without changing the log.

    Returns (args, dropped) so the dropped list goes into train_log.json.
    """
    import inspect
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    renames = {
        "eval_strategy": ["eval_strategy", "evaluation_strategy"],
        "evaluation_strategy": ["evaluation_strategy", "eval_strategy"],
    }

    final, dropped = {}, []
    for key, value in wanted.items():
        if key.startswith("_"):      # internal hint, never a real argument
            continue
        candidates = renames.get(key, [key])
        for name in candidates:
            if name in allowed:
                final[name] = value
                break
        else:
            dropped.append(key)

    # warmup_ratio is the one worth rescuing rather than dropping: without any
    # warmup, a 2e-4 learning rate on step one can destabilise the run.
    if "warmup_ratio" in dropped and "warmup_steps" in allowed:
        ratio = wanted.get("warmup_ratio", 0.0)
        steps_per_epoch = wanted.get("_steps_per_epoch", 0)
        total = int(steps_per_epoch * wanted.get("num_train_epochs", 1))
        if total:
            final["warmup_steps"] = max(1, int(total * ratio))
            dropped.remove("warmup_ratio")

    final.pop("_steps_per_epoch", None)
    if dropped:
        print(f"\n  NOTE: this transformers version does not accept "
              f"{dropped}; proceeding without them.\n")
    return TrainingArguments(**final), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-4bit", action="store_true",
                    help="QLoRA. Required above ~3B on a 16GB card.")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap training examples, for a quick smoke test")
    ap.add_argument("--max-val", type=int, default=150,
                    help="cap validation examples. Validation loss is read as a "
                         "trend, not a headline number -- the reported results "
                         "come from stage 7 on the held-out test set -- so "
                         "evaluating all 460 every epoch buys nothing and costs "
                         "~10 min per epoch on a 3B. Set 0 to use all of them.")
    args = ap.parse_args()

    set_seed(args.seed)
    gpu = gpu_profile()
    print(json.dumps(gpu, indent=2))

    if args.load_4bit and not gpu["can_4bit"]:
        raise SystemExit(
            f"\n4-bit quantisation needs compute capability 7.0+; this GPU is "
            f"{gpu['capability']} ({gpu['name']}).\n"
            "On Kaggle, switch the accelerator to T4 x2 instead of P100.\n"
        )

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    use_bf16 = gpu["bf16"]
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(trust_remote_code=True, dtype=compute_dtype, device_map={"": 0})
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,     # nested quantisation: ~0.4 bits/param saved
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model, load_path = load_base_model(args.model, load_kwargs)
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = detect_target_modules(model)
    print(f"  target modules detected: {target_modules}")

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # All seven linear projections -- attention AND the MLP block. See the
        # module docstring: which modules carry adapters matters more than rank.
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} / {total:,}  ({100 * trainable / total:.3f}%)")

    train_ds = NextQuestionDataset(args.data_dir / "train.jsonl", tok, args.max_len)
    val_ds = NextQuestionDataset(args.data_dir / "val.jsonl", tok, args.max_len)
    if args.max_train:
        train_ds.rows = train_ds.rows[: args.max_train]
    if args.max_val:
        # Deterministic subsample: take every k-th example rather than the
        # first N, so the slice spans all turn depths instead of only the
        # earliest reviews in file order.
        step = max(1, len(val_ds.rows) // args.max_val)
        val_ds.rows = val_ds.rows[::step][: args.max_val]
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.model.split("/")[-1] + ("-qlora" if args.load_4bit else "-lora")
    ckpt_dir = args.out_dir / run_name

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    wanted = dict(
        output_dir=str(ckpt_dir),
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="paged_adamw_8bit" if args.load_4bit else "adamw_torch",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
        dataloader_num_workers=0,   # determinism over speed
        _steps_per_epoch=steps_per_epoch,
    )
    targs, dropped_args = build_training_args(TrainingArguments, wanted)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )

    t0 = time.time()
    result = trainer.train()
    minutes = round((time.time() - t0) / 60, 1)

    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    log = {
        "stage": "6_training",
        "run_name": run_name,
        "base_model": args.model,
        "gpu": gpu,
        "minutes": minutes,
        "load_path": load_path,
        # Every hyperparameter, logged. The submission asks for a reproducible
        # training script with all hyperparameters recorded.
        "hyperparameters": {
            "seed": args.seed,
            "load_4bit": args.load_4bit,
            "quant": {"type": "nf4", "double_quant": True,
                      "compute_dtype": str(compute_dtype)} if args.load_4bit else None,
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "lora_dropout": args.dropout,
            "target_modules": sorted(lora.target_modules),
            "learning_rate": args.lr,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "max_seq_len": args.max_len,
            "optimizer": targs.optim,
            "precision": "bf16" if use_bf16 else "fp16",
        },
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "val_subsampled": bool(args.max_val),
        "training_args_dropped_by_version": dropped_args,
        "transformers_version": __import__("transformers").__version__,
        "train_examples_truncated": train_ds.n_truncated,
        "final_train_loss": round(result.training_loss, 4),
        "log_history": trainer.state.log_history,
        "adapter_path": str(adapter_dir),
    }
    out = args.results_dir / f"train_log_{run_name}.json"
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")
    # run.py looks for this fixed name
    (args.results_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\ndone in {minutes} min -- adapter at {adapter_dir}")
    print(f"final train loss {log['final_train_loss']}")
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    for e in evals:
        print(f"  epoch {e.get('epoch')}: eval_loss {round(e['eval_loss'], 4)}")
    if train_ds.n_truncated:
        print(f"note: {train_ds.n_truncated} training examples were left-truncated "
              f"at {args.max_len} tokens")


if __name__ == "__main__":
    main()
