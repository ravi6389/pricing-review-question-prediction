# Running stages 6 and 7 on Kaggle

Stages 1–5 run on your laptop and produce the data. Only training and
evaluation need a GPU.

## 1. Notebook settings

- **Accelerator: GPU T4 x2.** Not P100 — 4-bit quantisation is unreliable below
  compute capability 7.0, and the P100 is 6.0. The T4 is 7.5.
- **Internet: On.** Needed to pull base weights from HuggingFace.
- Persistence: "Files only" so `/kaggle/working` survives between sessions.

The script checks the GPU itself and refuses to start a 4-bit run on a P100
rather than dying forty minutes in.

## 2. Get your data onto Kaggle

Easiest route — upload the generated data as a Kaggle Dataset:

1. Datasets → New Dataset
2. Upload `data/train.jsonl`, `data/val.jsonl`, `data/test_ood.jsonl`
3. Name it `pricing-review-data`
4. In the notebook: Add Input → your dataset

It lands at `/kaggle/input/pricing-review-data/`.

Re-upload as a new version whenever you regenerate the data.

## 3. Setup cell

```python
!pip -q install -U "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" \
                   "bitsandbytes>=0.43" "datasets>=2.20"

import torch
print(torch.__version__, torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))
# expect (7, 5) on a T4. (6, 0) means you are on a P100 -- change the accelerator.
```

## 4. Put the source on the machine

Either clone your repo:

```python
!git clone https://github.com/<you>/avathon-pricing-review.git /kaggle/working/repo
%cd /kaggle/working/repo
```

Or, before pushing, add `src/train.py` and `src/evaluate.py` to the same Kaggle
dataset and copy them across:

```python
!mkdir -p /kaggle/working/src
!cp /kaggle/input/pricing-review-data/*.py /kaggle/working/src/
%cd /kaggle/working
```

## 5. Smoke test before the real run

Always. Sixty examples, one epoch — proves the data loads, the chat template
applies, the adapter attaches and a step runs, in about two minutes.

```python
!python src/train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data-dir /kaggle/input/pricing-review-data \
  --out-dir /kaggle/working/outputs \
  --results-dir /kaggle/working/results \
  --max-train 60 --epochs 1
```

If that finishes, scale up. If it fails, you have lost two minutes rather than
an hour.

## 6. The real runs, cheapest first

```python
# ~40 min
!python src/train.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir /kaggle/input/pricing-review-data \
  --out-dir /kaggle/working/outputs --results-dir /kaggle/working/results

# ~1.5 h -- the primary model
!python src/train.py --model Qwen/Qwen2.5-3B-Instruct \
  --data-dir /kaggle/input/pricing-review-data \
  --out-dir /kaggle/working/outputs --results-dir /kaggle/working/results

# ~4-5 h -- only if Sunday evening still has runway
!python src/train.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --data-dir /kaggle/input/pricing-review-data \
  --out-dir /kaggle/working/outputs --results-dir /kaggle/working/results
```

Cheapest first so a session timeout never leaves you with nothing.

## 7. Get the results back

Adapters are small — tens of megabytes — so commit them with the notebook
output and download, then copy `results/train_log_*.json` into the repo. The
base weights are not committed; the adapter plus the pinned base model name is
enough to reproduce.

## Quota and session notes

- 30 GPU hours per week, which is ample: the full plan is under 8.
- A session ends if left idle. "Save Version → Run All" runs it detached, which
  is the safer way to launch the long 7B job.
- `/kaggle/working` is capped at 20 GB. Keep `save_total_limit=1` (the script
  already does) or checkpoints will fill it.

## Licence check before you commit to a base model

Verify the licence on each model card and note it in the write-up. Some Qwen2.5
sizes are Apache 2.0 while others sit under a more restrictive research
licence — and "I rejected this base model because its licence does not permit
commercial use" is exactly the kind of rejection reasoning the rubric rewards.
