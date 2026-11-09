# Full results — all models, base and fine-tuned

Test set: 200 held-out deals (deals and wordings unseen in training). Greedy decoding.
Gold mean question count: **4.74**. Base models were given the full 14-question taxonomy in
their prompt.

---

## Table 1 — Headline comparison, ranked by F1

| # | Model | Tuned | **F1** | Recall | Precision | Macro-F1 | Off-list | Qs asked |
|---|---|---|---|---|---|---|---|---|
| — | *most_common_5 (no model)* | — | *0.620* | *0.637* | *0.604* | — | — | *5.00* |
| 1 | Llama 3.2 1B | **yes** | **0.559** | 0.668 | 0.481 | 0.534 | 6% | 8.73 |
| 2 | Qwen2.5 3B | **yes** | 0.508 | 0.438 | 0.604 | 0.484 | 14% | 4.12 |
| — | *ask_all_14 (no model)* | — | *0.506* | *1.000* | *0.339* | — | — | *14.00* |
| 3 | Llama 3.2 3B | **yes** | 0.484 | 0.638 | 0.390 | 0.422 | 8% | 10.49 |
| 4 | Qwen2.5 0.5B | **yes** | 0.471 | 0.391 | 0.590 | 0.426 | 19% | 4.57 |
| 5 | Qwen2.5 1.5B | **yes** | 0.390 | 0.295 | 0.575 | 0.355 | 24% | 3.21 |
| 6 | Qwen2.5 3B | no | 0.361 | 0.358 | 0.365 | 0.275 | 61% | 12.07 |
| 7 | Qwen2.5 1.5B | no | 0.349 | 0.367 | 0.333 | 0.261 | 62% | 13.87 |
| 8 | Llama 3.2 1B | no | 0.319 | 0.309 | 0.330 | 0.243 | 68% | 14.57 |
| 9 | Qwen2.5 0.5B | no | 0.314 | 0.254 | 0.412 | 0.211 | 82% | 19.15 |
| 10 | Llama 3.2 3B | no | 0.256 | 0.222 | 0.304 | 0.169 | 56% | 8.03 |

**Every fine-tuned model outscores every base model** (worst tuned 0.390 > best base 0.361).

---

## Table 2 — Paired lift, base → fine-tuned

| Model | F1 base | F1 tuned | **Lift** | Off-list base → tuned | Qs base → tuned |
|---|---|---|---|---|---|
| Llama 3.2 1B | 0.319 | 0.559 | **+0.240** | 68% → 6% | 14.6 → 8.7 |
| Llama 3.2 3B | 0.256 | 0.484 | **+0.227** | 56% → 8% | 8.0 → 10.5 |
| Qwen2.5 0.5B | 0.314 | 0.471 | +0.156 | 82% → 19% | 19.2 → 4.6 |
| Qwen2.5 3B | 0.361 | 0.508 | +0.146 | 61% → 14% | 12.1 → 4.1 |
| Qwen2.5 1.5B | 0.349 | 0.390 | +0.041 | 62% → 24% | 13.9 → 3.2 |

Mean lift: **Llama +0.234, Qwen +0.115** — Llama gains roughly twice as much from the same
adapter configuration.

---

## Table 3 — Validation loss before fine-tuning

100 validation examples, loss computed only on the question list.

| Model | Val loss | Perplexity | Tokens/example |
|---|---|---|---|
| Qwen2.5 1.5B | **3.002** | 20.12 | 372.1 |
| Llama 3.2 3B | 3.159 | 23.55 | 379.0 |
| Qwen2.5 0.5B | 3.198 | 24.47 | 372.1 |
| Llama 3.2 1B | 3.211 | 24.80 | 379.0 |
| Qwen2.5 3B | 3.273 | 26.39 | 372.1 |

Within Qwen (same tokenizer) this is a fair comparison and it is **non-monotonic in size**:
1.5B starts best, 3B starts worst. Qwen-vs-Llama is *not* comparable on loss — different
tokenizers split the same text differently (372 vs 379 tokens per example).

**Note the inversion:** Qwen 1.5B has the best starting loss and the worst final task score.
Loss on the target text did not predict task performance.

---

## Table 4 — Trivial baselines (no model reads the deal)

| Strategy | F1 | Recall | Precision | Questions |
|---|---|---|---|---|
| most_common_5 — always ask levers 1, 12, 2, 3, 4 | **0.620** | 0.637 | 0.604 | 5 |
| ask_all_14 — always ask everything | 0.506 | 1.000 | 0.339 | 14 |

`ask_all_14` shows why recall cannot be the headline metric: a useless strategy achieves
perfect recall. `most_common_5` outscores every model, which is a flaw in the dataset
design — see the error analysis.

---

## Table 5 — Per-lever recall, best model vs best base model

Tuned Llama 3.2 1B against base Qwen2.5 3B. "Deals" is how many of the 200 test deals
should raise that question.

| Lever | Deals | Tuned Llama-1B | Base Qwen-3B |
|---|---|---|---|
| L1 price increase headroom | 200 | 1.00 | 0.19 |
| L12 competitor bundling | 132 | 0.81 | 0.67 |
| L2 targeted increase, low-margin lines | 121 | 1.00 | 0.26 |
| L3 rebate instead of reduction | 89 | 0.45 | 0.73 |
| L4 config or assay mix | 62 | 0.94 | 0.16 |
| L13 post-win matching and renewal | 61 | 0.33 | 0.30 |
| L5 competitor go-to-market | 50 | 0.48 | 0.30 |
| L6 pressure and sensitivity justified | 50 | 0.50 | 0.26 |
| L8 our price competitiveness | 50 | 0.42 | 0.10 |
| L9 realised price improvement | 46 | 0.20 | 0.30 |
| L10 negative realised-price line drivers | 46 | 0.04 | 0.46 |
| L11 loss risk and installed base | 28 | 0.21 | 0.46 |
| L14 term vs break-even mismatch | 13 | 0.00 | 0.69 |

**Recall broadly tracks how often a question appears in training.** The rare, late-ladder
questions — which lines are losing money, what protects us at renewal, term mismatch — are
the ones every model misses, and they are exactly the questions that matter most on a bad
deal.

The base model's apparently strong recall on rare levers (L14 at 0.69) is an artefact of
volume: it asks 12–15 questions on every deal, so it hits rare ones by accident. Its
precision on L14 is correspondingly poor — 130 false positives against 13 true cases.

---

## Known measurement issues

1. **L1 recall is understated.** The opening question applies to all 200 deals, yet tuned
   Qwen-3B recalls it only 48% while the adjacent L2 shows recall 1.00 with 71 false
   positives. The two are semantically close and the TF-IDF mapper is very likely
   assigning one to the other. The largest single failure in the table is probably a
   scoring artefact, not a model failure.
2. **The base-model prompt was flawed.** It carried a leftover instruction from the
   conversational variant ("ask about exactly one of them… in the conversation so far")
   that conflicts with set generation, and contained no user turn. Both plausibly depress
   base scores, so the base-vs-tuned gap is an **upper bound**.
3. **Mapping favours tuned models.** Generated questions are matched to levers by TF-IDF
   similarity; a model writing in its training style maps more cleanly. Off-list rate is
   reported per run so the reader can judge how much of each gap is mapping failure.
