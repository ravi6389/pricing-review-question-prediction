"""
Deterministic pipeline runner.

    python run.py step1          # run one step
    python run.py verify         # re-hash every artifact and compare to manifest
    python run.py list           # show steps and status

WHY THIS EXISTS
    Reviewers re-run the pipeline and must get identical output. Three of the
    seven steps call a large language model, which is not reproducible even at
    temperature 0. So every model call is cached on disk, keyed by the SHA-256
    of its exact prompt, and the cache is committed to the repository. By
    default the pipeline REPLAYS the cache and never touches the network; pass
    --regenerate to a model step to call the API again (which rewrites the
    cache, and is therefore an authoring action, not a reproduction action).

    Every step writes the SHA-256 of each artifact it produced into
    results/manifest.json. `python run.py verify` re-hashes the files on disk
    and reports any drift, so a reviewer can confirm reproduction in one command.

HONEST LIMITS OF DETERMINISM (stated in the README too)
    Steps 1-5 are bit-identical on any machine: pure Python, seeded, cached.
    Step 6 (training) is seeded and runs with deterministic kernels enabled, so
    it is reproducible on the SAME GPU model and library versions. It is NOT
    bit-identical across different GPU architectures -- floating-point
    reduction order differs. Step 7 (evaluation) uses greedy decoding, so it is
    deterministic given identical weights.
    Claiming more than this would be false, so the manifest records the GPU
    name and library versions alongside the hashes.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.json"

GLOBAL_SEED = 42

# ---------------------------------------------------------------------------
# Step registry: (id, description, command, artifacts it must produce)
# ---------------------------------------------------------------------------

STEPS = {
    "step1": dict(
        desc="Fix true deal feature values from declared distributions",
        cmd=[sys.executable, "-I", "src/generate_deals.py", "--n", "1000", "--seed", str(GLOBAL_SEED)],
        artifacts=["data/deals.jsonl", "data/deals_card.json"],
        uses_model=False,
    ),
    "step2": dict(
        desc="Write each deal up as natural prose (CACHED model calls)",
        cmd=[sys.executable, "-I", "src/write_documents.py", "--seed", str(GLOBAL_SEED)],
        artifacts=["data/documents.jsonl", "data/documents_card.json", "cache/documents.json"],
        uses_model=True,
    ),
    "step3": dict(
        desc="Extract features back from prose and score against step 1 truth",
        cmd=[sys.executable, "-I", "src/extract_features.py", "--seed", str(GLOBAL_SEED)],
        artifacts=["data/extracted.jsonl", "results/extraction_scores.json", "cache/extraction.json"],
        uses_model=True,
    ),
    "step4": dict(
        desc="Run the lever rules to produce review conversations",
        cmd=[sys.executable, "-I", "src/simulate_reviews.py", "--seed", str(GLOBAL_SEED)],
        artifacts=["data/question_sets.jsonl", "data/question_sets_card.json", "results/lever_distribution.json"],
        uses_model=False,
    ),
    "step5": dict(
        desc="Phrase each lever in the manager's voice (CACHED model calls)",
        cmd=[sys.executable, "-I", "src/phrase_questions.py", "--seed", str(GLOBAL_SEED)],
        artifacts=["data/train.jsonl", "data/val.jsonl", "data/test_ood.jsonl", "cache/phrasings.json"],
        uses_model=True,
    ),
    "step6": dict(
        desc="Fine-tune with LoRA (seeded, deterministic kernels)",
        cmd=[sys.executable, "-I", "src/train.py", "--seed", str(GLOBAL_SEED)],
        artifacts=["results/train_log.json"],
        uses_model=False,
    ),
    "step7": dict(
        desc="Evaluate base and tuned models with greedy decoding",
        cmd=[sys.executable, "-I", "src/evaluate.py", "--seed", str(GLOBAL_SEED)],
        artifacts=["results/eval.json"],
        uses_model=False,
    ),
}

STEP_ORDER = list(STEPS)


# ---------------------------------------------------------------------------
# Determinism scaffolding
# ---------------------------------------------------------------------------

def seeded_env() -> dict:
    """
    Environment every step inherits. PYTHONHASHSEED must be set in the ENV, not
    in Python -- by the time the interpreter is running it is too late, and
    unseeded string hashing changes set and dict iteration order, which silently
    changes sampling order.
    """
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    # Required for deterministic cuBLAS reductions in step 6.
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_fingerprint() -> dict:
    """Recorded with the hashes so a reviewer can tell apples from oranges."""
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": GLOBAL_SEED,
    }
    try:
        import torch  # noqa
        fp["torch"] = torch.__version__
        fp["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        fp["torch"] = None
    for mod in ("transformers", "peft", "datasets"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = None
    return fp


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"environment": {}, "steps": {}}


def save_manifest(m: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def run_step(step: str, regenerate: bool) -> int:
    spec = STEPS[step]
    cmd = list(spec["cmd"])
    if regenerate:
        if not spec["uses_model"]:
            print(f"{step} makes no model calls; --regenerate has no effect.")
        else:
            cmd.append("--regenerate")

    print(f"\n=== {step}: {spec['desc']} ===")
    print("  $", " ".join(cmd))
    if spec["uses_model"] and not regenerate:
        print("  (replaying cached model calls -- no network access)")

    proc = subprocess.run(cmd, cwd=ROOT, env=seeded_env())
    if proc.returncode != 0:
        print(f"\n{step} FAILED (exit {proc.returncode}). Manifest not updated.")
        return proc.returncode

    missing = [a for a in spec["artifacts"] if not (ROOT / a).exists()]
    if missing:
        print(f"\n{step} finished but did not produce: {missing}")
        return 1

    m = load_manifest()
    m["environment"] = environment_fingerprint()
    m["steps"][step] = {
        "description": spec["desc"],
        "artifacts": {a: sha256(ROOT / a) for a in spec["artifacts"]},
    }
    save_manifest(m)

    print(f"\n{step} complete. Artifact hashes:")
    for a, h in m["steps"][step]["artifacts"].items():
        print(f"  {h[:16]}...  {a}")

    nxt = STEP_ORDER.index(step) + 1
    if nxt < len(STEP_ORDER):
        print(f"\nPaused. Next: python run.py {STEP_ORDER[nxt]}")
    else:
        print("\nPipeline complete. Verify with: python run.py verify")
    return 0


def verify() -> int:
    m = load_manifest()
    if not m["steps"]:
        print("No manifest yet. Run the steps first.")
        return 1

    ok = True
    print("Verifying artifacts against results/manifest.json\n")
    for step in STEP_ORDER:
        if step not in m["steps"]:
            continue
        for a, expected in m["steps"][step]["artifacts"].items():
            p = ROOT / a
            if not p.exists():
                print(f"  MISSING   {a}")
                ok = False
                continue
            actual = sha256(p)
            if actual == expected:
                print(f"  match     {a}")
            else:
                print(f"  DRIFT     {a}\n              expected {expected}\n              actual   {actual}")
                ok = False

    rec, cur = m.get("environment", {}), environment_fingerprint()
    drifted = {k: (rec.get(k), cur.get(k)) for k in rec if rec.get(k) != cur.get(k)}
    if drifted:
        print("\nEnvironment differs from the recorded run:")
        for k, (was, now) in drifted.items():
            print(f"  {k}: recorded {was!r}, now {now!r}")
        print("  Steps 1-5 should still match exactly. Step 6 may differ on a"
              "\n  different GPU or library version -- this is expected and documented.")

    print("\nALL ARTIFACTS MATCH" if ok else "\nDRIFT DETECTED -- see above")
    return 0 if ok else 1


def show_list() -> int:
    m = load_manifest()
    print(f"{'step':<8}{'model':<8}{'ran':<6}description")
    for s in STEP_ORDER:
        spec = STEPS[s]
        print(f"{s:<8}{'yes' if spec['uses_model'] else 'no':<8}"
              f"{'yes' if s in m['steps'] else 'no':<6}{spec['desc']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=STEP_ORDER + ["verify", "list"])
    ap.add_argument("--regenerate", action="store_true",
                    help="re-call the model API for a cached step (authoring, not reproduction)")
    args = ap.parse_args()

    if args.target == "verify":
        return verify()
    if args.target == "list":
        return show_list()
    return run_step(args.target, args.regenerate)


if __name__ == "__main__":
    raise SystemExit(main())
