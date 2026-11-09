# Design decisions and rejected alternatives

Working log. Every significant choice, with at least one alternative that was
considered and why it was rejected. The technical write-up is drawn from this.

---

## 1. Track

**Chose Track B, small language model fine-tuning.**

Rejected: **Track D (retrieval)** — closest to my day job and the safest build,
but the task I care about is a *behaviour*, not a lookup, so retrieval would
have been the wrong tool for the problem I wanted to solve.
Rejected: **Track A (multi-agent)** — the decomposition I would have used
(planner / analyst / critic / recommender over funnel data) is strong, but the
critic-vs-analyst tension is the only part that genuinely needs separate
agents, and the rest would have been one tool-using agent wearing four hats.
Rejected: **Track C (classical ML)** — defensible, but forecasting is my weaker
area and the result would have been competent rather than distinctive.

---

## 2. Problem

**Predict the next question a pricing review manager asks a sales
representative about a deal.**

Domain: medical diagnostics instrument placements. The company places an
analyser with a customer and earns on reagent pull-through over a lease term.
Deals are reviewed before approval; the review is an interrogation that works
through commercial levers in a fixed order.

Business value is **review cycle-time**, not inference cost alone: each round
of manager questions costs days. A model that anticipates them lets the
representative pre-empt them in the first submission.

Rejected: **messy notes → structured CRM record.** Higher volume and more money
at stake, but extraction is a solved shape and the capability gap over a base
model is small.
Rejected: **deal verdict (approve / rework / escalate).** Reduces to
classification over features a parser could extract, which undermines the case
for a language model at all.

---

## 3. Why fine-tuning, and not retrieval or prompting

The honest position, because a reviewer will push here:

**Retrieval could do a decent job of this** with a frontier model and retrieved
past transcripts. The argument for fine-tuning is narrower:

1. **Economics.** A review is 10–15 turns. Retrieval means several long example
   transcripts in the prompt on every turn. Across thousands of reviews a month
   that is a recurring bill that scales with volume, plus seconds of latency
   per turn while a manager waits.
2. **Once cost forces a small model, retrieval stops working.** A 3B given
   retrieved transcripts and a 14-lever policy in the prompt drifts: it
   re-asks settled levers and loses the format. **Fine-tuning is how you get
   large-model behaviour at small-model cost.** That is the claim, not
   "fine-tuning beats retrieval".
3. **Style lives in the weights.** The acknowledge-then-pivot phrasing is
   consistent across every turn of the source transcript. Prompting
   approximates a house style; training internalises it.

**Claim explicitly dropped after examination:** an earlier draft argued that
retrieval cannot track which levers are exhausted. That is wrong — exhausted
levers can simply be listed in the prompt. Dropping the claim makes the
argument stronger.

**Where retrieval wins, and this goes in limitations:** when policy changes. A
retrieval system updates by editing a document; this model needs retraining.
Production should be both — fine-tune the behaviour, retrieve the volatile
facts (current thresholds, this quarter's price list).

---

## 4. Data: synthetic, and why that is not a weakness

**No public dataset exists** for pricing managers interrogating sales
representatives. I checked. The nearest academic relative is **QuAC**
(Question Answering in Context, 14k information-seeking dialogues, ~100k
questions), where a questioner asks sequential free-form questions and, as the
authors note, questions are often only meaningful within the dialogue context —
exactly the dependency structure here. Cited as precedent, not used as data.

Rejected as the dataset: **LEDGAR** (80k contract provisions, 100 classes, real
SEC filings). Real data with real labels, and it was my first choice — rejected
because the task would have been single-label classification, not the
sequential interrogation the business problem actually is.
Rejected: **CUAD** (510 contracts, 13k expert annotations, 41 categories) —
span extraction over long contracts; wrong risk profile for the time budget.

### 4.1 The generator fixes the truth; the model only writes prose

A script draws each deal's feature values from declared distributions. A large
model then renders those numbers as a deal note.

Rejected: **model invents the deal, a second model extracts the features.**
This is the obvious pipeline and it is wrong. The second model's extraction
would *become* the ground truth with nothing to check it against, so extraction
accuracy would be unmeasurable and label noise unbounded. Fixing the values
first means the extractor can be scored.

Also rejected: **separate models for generation and extraction make the
pipeline "independent".** Models from the same family share priors; their
errors correlate. Separate API calls give independence of sampling, not
independence of bias.

### 4.2 Archetype-conditioned sampling

Each deal draws an archetype (clean / margin squeeze / underwater / competitive
bundle / term mismatch / displacement) from a declared mix, and field
distributions are conditioned on it.

Rejected: **drawing each field independently** — produces incoherent deals and
leaves lever coverage to luck. Rare situations would appear too few times to
train or test on.
Rejected: **rejection sampling against coverage targets** — hits the same
targets, but the resulting distribution is implicit in the acceptance rule
rather than declared, which is harder to document honestly.

**Corrected by a sanity check:** the first version drew the annual reagent
commitment and derived break-even from it, which put break-even outside the
lease term in **76%** of deals — commercially absurd. Inverted to match how
deal desks actually work: draw the target payback, size the reagent commitment
to hit it. Break-even outside term is now **18%**.

### 4.3 Rules choose the levers; the model only phrases them

Rejected: **a large model decides which questions to ask.** The whole premise
is that the lever ladder is institutional knowledge a general model lacks. If
a general model generated the labels, the ground truth would encode its guess
about an industry it does not work in, and the student would learn genericness.

Rules were drafted with model assistance from one real transcript, then
reviewed and corrected by me. `RULES_MODIFIED_ON_EXPERT_REVIEW = <N>`.

**Validation:** running the rule engine against the deal behind the real
transcript reproduces **13 of its 14 turns in the same order**. The divergence
is one extra follow-up turn pressing on why the realised price could not
improve.

**Two rules the real transcript disproved:**
- Terminating after 3 consecutive refusals. The real manager pushed through
  ~10 straight refusals. A refusal moves *down* the ladder; it does not end the
  review.
- Treating only flat refusals as needing a follow-up. The real manager also
  drills into short unexplained *assertions* ("pricing pressure").

**A third bug, found by the simulator:** lever triggers tested only the
`exhausted` set, so a lever the representative *answered* never unlocked the
next rung and conversations died after 2.1 turns. Fixed with a combined
`closed` set; mean length is now 4.8 turns with a long tail to 13.

**A fourth:** the manager started out already knowing the competitor's bundling
status, which made the bundling lever unaskable and fired the renewal lever
immediately. Those facts now start unknown and are revealed only when asked.

### 4.4 Held-out surface on both sides

Train/validation documents use author voice pool A; test uses pool B, which
shares no phrasing conventions. Manager phrasings 0–29 go to train/validation,
30–39 to test.

So the test set shares no surface wording with training on **either** side of
the conversation. A model that memorised phrasings cannot score on it.

Rejected: **one voice with a random row split** — cheaper, but then test and
train differ only by sampling and a strong test score proves nothing about
generalisation.

**Measured, not asserted:** `phrase_questions.py` reports
`leakage_check.shared_questions` every run. The offline template fallback
produced only ~8 distinct stems per lever, which collided the slices and leaked
**70%** of test questions into training. Caught before any training ran.

### 4.5 Variant pools instead of per-turn generation

4,786 manager turns would be ~4,800 model calls. Instead: ~40 phrasings per
lever (15 calls total), composed as acknowledgement + question.

A tenth of the cost, full control over which lever each turn expresses, and —
the part that matters — it makes the phrasing holdout possible at all.

Cost: questions are more formulaic than a real manager, who would weave the
previous answer into the question rather than prefixing it. Listed as a
limitation.

---

## 5. Extraction stage

Kept in the system (the scenario is unstructured content and production needs
it), but **not** used to produce labels.

Two extractors are implemented deliberately: a language model, and a
**rule-based parser as the honest baseline**. If a hand-written parser matches
the model, the model is not earning its cost.

**Finding from the baseline:** the parser scores 0.984 field accuracy on stated
fields but **hallucinates a value for 53.9% of fields the document does not
state**. The cause is structural — a regular expression that fails to match
returns *false*, and cannot distinguish "not bundling" from "never mentioned".
A language model can answer "not stated". That is a measured argument, not an
assumed one.

Caveat stated: 0.984 flatters the parser, because the expressions were written
against the documents being parsed.

---

## 6. Base model

**Primary: Qwen2.5-3B-Instruct.** Scaling curve across 0.5B / 1.5B / 3B / 7B.

Family held constant for the curve: same tokenizer, same recipe, so size is the
only variable.

Rejected for the curve: **mixing Qwen and Llama.** Different tokenizers mean
per-token loss is not a like-for-like quantity. Cross-family comparison is done
on next-lever accuracy, which is tokenizer-independent, never on loss.

Rejected: **training a 7B.** 4–5 hours against a 4-hour budget, and it answers
a question already answered by running the 7B as a *base* model for ten
minutes. If the tuned 3B beats the base 7B, adaptation beat scale and the point
is made.

Licence checked per model card; any size under a non-commercial research
licence is unsuitable for this enterprise use case and noted as such.

---

## 7. Fine-tuning configuration

**LoRA, rank 16, alpha 32, dropout 0.05, all seven linear projections.**

- **All seven projections** (`q,k,v,o,gate,up,down`), not attention only. The
  QLoRA paper's finding is that *which* modules carry adapters matters more
  than the rank; freezing the feed-forward block — where most parameters live —
  measurably underperforms at equal cost. Attention-only is the common shortcut
  and it is worse.
- **Rank 16** because the task is a decision policy over 14 options plus a
  house style, not new world knowledge. Capacity is not the binding constraint;
  a larger rank mostly buys overfitting on 3,446 examples.
- **Alpha = 2 × rank**, the usual effective scaling.
- Rejected: **full fine-tuning** — 3B parameters will not fit on a 16GB card
  with optimizer states, and there is no reason to move every weight for a
  behavioural adaptation.
- Rejected: **QLoRA for the 3B** — 4-bit is needed above ~3B on 16GB; using it
  at 3B would add quantisation error for memory I do not need.

**Precision chosen from the GPU.** bf16 needs Ampere (8.0+); the Kaggle T4 is
Turing (7.5), so fp16. 4-bit NF4 is unreliable below 7.0, so the script refuses
to quantise on a P100 rather than failing mid-run.

**Loss masked to the final manager question only.** Without masking the model
would also learn to generate the representative's replies and the deal note —
not the task, and a waste of capacity.

---

## 8. Evaluation

**Primary: next-lever accuracy. Reported beside macro-F1.** The label
distribution is skewed by construction (opening lever 20.3% of examples, term
lever 1.2%), so accuracy alone is dominated by the common early levers.

**Repeat rate** — does the model re-ask a settled lever. Needs no label mapping
to interpret, and measures whether the model tracked conversation state rather
than pattern-matching the last turn.

**Mapping generated text to a lever** is TF-IDF cosine against the per-lever
phrasing pools, with a similarity floor. Self-mapping gold questions scores
1.0, so the decoder is not the bottleneck.

**Stated bias, because a reviewer would otherwise find it:** a fine-tuned model
writes in the style it was trained on and maps more cleanly than a base model
writing freely. This favours the fine-tuned model. Two mitigations —
`unmappable_rate` is reported for every run, and base models are evaluated with
the full lever taxonomy in the prompt.

Rejected: **asking each model to emit a lever number.** Removes mapping noise
entirely, but measures a different task than the one trained for, and would let
a model score well while writing an unusable question.

**Fair baseline.** Base models get the taxonomy in their system prompt. Scoring
them on a taxonomy they have never seen would be a rigged comparison, and the
baseline was deliberately strengthened rather than weakened.

**Also run:** a retrieval baseline (retrieved similar transcripts, few-shot, base
model) so the "why not retrieval" argument is demonstrated rather than asserted.

**Catastrophic forgetting:** 200 general prompts from OASST2 through base and
tuned models, compared. If the model has stopped holding an ordinary
conversation, that is reported.

---

## 9. Reproducibility

Every model call is cached on disk keyed by the SHA-256 of its prompt, and the
cache is committed. A reviewer reproduces every artifact **with no API key and
no cost**. `--regenerate` is the authoring path.

`run.py` runs one stage at a time, writes the SHA-256 of each artifact to
`results/manifest.json`, and `python run.py verify` re-hashes and reports drift.

**Limits stated rather than overclaimed:** stages 1–5 are bit-identical
anywhere (verified Windows against Linux — identical hashes). Stage 6 is seeded
with deterministic kernels but is not bit-identical across GPU architectures,
because floating-point reduction order differs. Stage 7 uses greedy decoding.
The manifest records GPU and library versions beside the hashes.

`PYTHONHASHSEED` is set in the environment, not in Python — by interpreter
start it is too late, and unseeded string hashing changes set iteration order,
which silently changes sampling.

**Version fragility encountered**, which is why dependencies are pinned:
`torchao` 0.10.0 against a `peft` requiring >0.16.0; `TrainingArguments` no
longer accepting `warmup_ratio`; `apply_chat_template` returning a
`BatchEncoding` rather than a list. The training script now adapts to the
installed signature and **reports** anything it had to drop, since a silently
ignored learning-rate schedule would change results without changing the log.

---

## 10. Limitations

1. Ground truth is expert-authored rules, not observed behaviour. Validated
   against one real transcript (13/14 turns). Production would train on logged
   reviews.
2. The simulated representative never contradicts itself, stalls, or volunteers
   information. The model is not prepared for an evasive counterparty.
3. Manager questions are composed from variant pools, so more formulaic than a
   real manager.
4. Synthetic deals are more balanced and internally coherent than a real
   pipeline, where most deals are unremarkable.
5. Policy changes require retraining. A production system would fine-tune the
   behaviour and retrieve the volatile thresholds.
6. Label-mapping bias at evaluation favours the fine-tuned model (section 8).
