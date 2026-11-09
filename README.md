# Predicting Pricing-Review Questions from Deal Notes

**Avathon AI Hiring Challenge — Track B (SLM Fine-Tuning), Scenario S2**
Ravi Shankar Prasad

**Video walkthrough (5 min):** https://drive.google.com/file/d/1bWHvZuBcsudgZY80ij5G5ynrm4aYWxdO/view
**Write-up:** [`writeup/Avathon_TrackB_RaviShankarPrasad.pdf`](writeup/Avathon_TrackB_RaviShankarPrasad.pdf)

---

## What this does

Given a free-text deal note written by a sales rep, predict **which questions a pricing manager will ask** — drawn from a 14-question taxonomy, typically 1 to 13 per deal. The rep answers them before submitting, so the review round that currently costs days happens up front.

Five small language models were fine-tuned with LoRA and compared against their own untrained baselines and two no-model baselines.

## Headline result

200 held-out deals, micro-F1 over the question set. **Every fine-tuned model beat every base model.**

| Model | Base F1 | Tuned F1 | Off-list base → tuned | Questions base → tuned |
|---|---|---|---|---|
| Llama 3.2 1B | 0.319 | **0.559** | 68% → 6% | 14.6 → 8.7 |
| Qwen2.5 3B | 0.361 | 0.508 | 61% → 14% | 12.1 → 4.1 |
| Llama 3.2 3B | 0.256 | 0.484 | 56% → 8% | 8.0 → 10.5 |
| Qwen2.5 0.5B | 0.314 | 0.471 | 82% → 19% | 19.2 → 4.6 |
| Qwen2.5 1.5B | 0.349 | 0.390 | 62% → 24% | 13.9 → 3.2 |

Gold mean: 4.74 questions per deal. "Off-list" is the share of generated questions matching none of the 14 levers — the hallucination proxy.

Serving the winner: **2.48 GB on a Tesla T4, median 2.9 s, p95 4.5 s.**

## Tests

```bash
pip install pytest
pytest -q          # 81 tests, ~1 second, no GPU and no API key
```

The rule engine and the scorer are pure functions over plain data, which is what makes them testable without a model. Coverage is concentrated where a silent error would be most expensive:

| Area | What is pinned |
|---|---|
| `test_levers.py` | All 14 trigger conditions, ladder traversal, bare-answer detection, and three regressions — the `closed` bug, the unconditional L1 fix, and refusals advancing rather than terminating |
| `test_scoring.py` | Precision/recall/F1 by hand-checkable cases, the similarity floor that keeps the hallucination rate honest, question splitting, and the trivial baselines |
| `test_pipeline.py` | Seed determinism, label integrity, archetype coverage, and the payback-derivation regression |

Several tests exist because the bug they describe actually happened. A scorer that is quietly wrong is worse than no scorer, because nobody questions its numbers.

## Logging

Every stage logs to the console and to `results/pipeline.log`. Verbosity is set with `--log-level` or the `LOG_LEVEL` environment variable; the file always keeps `DEBUG`.

```bash
python run.py step4 --log-level DEBUG
```

## Reproducing

```bash
pip install -r requirements.txt
python run.py all          # stages 1-5 (dataset); runs offline from cache, no API key needed
python run.py verify       # check artifact hashes against results/manifest.json
```

All LLM calls are cached on a SHA-256 of (model, messages, temperature), so the dataset regenerates byte-identically without credentials. Seeds, `PYTHONHASHSEED` and the cuBLAS workspace are pinned before any import. The manifest produced on Windows matches the one produced on Linux.

Training and evaluation (stages 6–7) need a GPU — see [`kaggle/KAGGLE.md`](kaggle/KAGGLE.md) for the notebook runbook.

```bash
python run.py 6 --model Qwen/Qwen2.5-0.5B-Instruct
python run.py 7 --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/Qwen2.5-0.5B-Instruct-lora/adapter
```

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1 Generate | `src/generate_deals.py` | 1,000 deals with known field values — the answer key |
| 2 Write | `src/write_documents.py` | Deal notes in sales voices; test uses two held-out voices |
| 3 Extract | `src/extract_features.py` | Measures field recovery (87.9% stated / 16.4% invented on omitted) |
| 4 Select | `src/simulate_reviews.py` | Rule engine fires levers → the training label |
| 5 Phrase | `src/phrase_questions.py` | 40 wordings per lever; 30 train / 10 test |
| 6 Train | `src/train.py` | LoRA adapters, five models |
| 7 Evaluate | `src/evaluate.py` | Ten runs + two no-model baselines |

Supporting: `src/levers.py` (the 14-lever taxonomy and trigger conditions), `src/baseline_loss.py`, `src/forgetting_check.py`.

## Design notes

- **A script fixes the truth; the language model only writes prose.** If a model invented the deal and another read it back, the second model's reading would *become* the answer key and extraction accuracy would be unmeasurable.
- **Rules select the questions; the model only phrases them.** The rules were drafted from one real 14-turn review and corrected by hand; replaying them against that transcript reproduces 13 of its 14 questions in order.
- **The test set is held out on three axes** — unseen deals, unseen author voices, unseen question wordings (0 shared questions, 0 shared deals, machine-checked every run).
- **Loss is masked to the question list**, so no gradient is spent learning to write deal notes.

Full reasoning, including rejected alternatives, is in the write-up. A working log of every decision is in [`writeup/decisions.md`](writeup/decisions.md).

## Known limitations

1. All data is synthetic; external validity is untested.
2. A no-model baseline (always ask the five most frequent questions) scores 0.620, above every model — the lever trigger frequencies are too skewed and need rebalancing.
3. Scoring uses a TF-IDF lever mapper with a known confusion between two adjacent levers.
4. The base-model prompt retains wording from an earlier conversational framing, so the base-vs-tuned gap is an upper bound.
5. The catastrophic-forgetting check was run on one model, not all five.

## Repository

```
run.py                  deterministic runner, 7 stages + verify
tests/                  pytest suite (81 tests, no GPU required)
src/                    pipeline stages
data/                   generated dataset + provenance cards
results/                scores, manifest, forgetting check
cache/                  LLM response cache (enables key-free reproduction)
writeup/                PDF write-up, decisions log, video script
kaggle/                 GPU runbook
```

Model weights are not committed — rerun stage 6 to regenerate adapters.
