"""
prepare_demo_dataset.py
=======================
Downloads Intel/orca_dpo_pairs from HuggingFace and creates three ready-to-use
files for the demo project:

  storage/demo/orca_dpo_pairs/
    sft_train.jsonl   — 300 rows, {"instruction": ..., "output": ...}
    dpo_train.jsonl   — 300 rows, {"prompt": ..., "chosen": ..., "rejected": ...}
    eval.jsonl        — 20 rows,  {"id": n, "question": ..., "reference": ...}
    manifest.json     — metadata for the presets API

Usage (from backend/ directory with venv active):
    python scripts/prepare_demo_dataset.py
    python scripts/prepare_demo_dataset.py --train-rows 500 --eval-rows 30
"""

import argparse
import json
import sys
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--train-rows", type=int, default=300,
                    help="Number of rows for training files (default: 300)")
parser.add_argument("--eval-rows",  type=int, default=20,
                    help="Number of rows for eval file (default: 20)")
parser.add_argument("--offset",     type=int, default=0,
                    help="Skip first N rows before taking train slice (default: 0)")
args = parser.parse_args()

TRAIN_ROWS = args.train_rows
EVAL_ROWS  = args.eval_rows
OFFSET     = args.offset
TOTAL_NEED = OFFSET + TRAIN_ROWS + EVAL_ROWS

# ── Output directory ──────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BACKEND_DIR / "storage" / "demo" / "orca_dpo_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT_DIR}")
print(f"Train rows: {TRAIN_ROWS}  |  Eval rows: {EVAL_ROWS}  |  Offset: {OFFSET}")
print(f"Total rows needed from dataset: {TOTAL_NEED}")
print()

# ── Download dataset ──────────────────────────────────────────────────────────

print("Downloading Intel/orca_dpo_pairs from HuggingFace...")

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not found. Install it:")
    print("  pip install datasets")
    sys.exit(1)

try:
    ds = load_dataset("Intel/orca_dpo_pairs", split="train", streaming=False)
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    print("Check your internet connection or HuggingFace access.")
    sys.exit(1)

print(f"Dataset loaded: {len(ds)} rows total")

# Validate expected columns
required = {"question", "chosen", "rejected"}
actual = set(ds.column_names)
if not required.issubset(actual):
    # Try alternate column names
    col_map = {}
    for col in actual:
        lc = col.lower()
        if lc in ("question", "prompt", "input", "instruction"):
            col_map["question"] = col
        elif lc in ("chosen", "good_answer", "accepted", "positive"):
            col_map["chosen"] = col
        elif lc in ("rejected", "bad_answer", "negative", "refused"):
            col_map["rejected"] = col
    if len(col_map) < 3:
        print(f"ERROR: Cannot find required columns in dataset.")
        print(f"  Found columns: {sorted(actual)}")
        print(f"  Expected: question, chosen, rejected")
        sys.exit(1)
    print(f"Column remapping: {col_map}")
else:
    col_map = {"question": "question", "chosen": "chosen", "rejected": "rejected"}

if len(ds) < TOTAL_NEED:
    print(f"WARNING: Dataset only has {len(ds)} rows, need {TOTAL_NEED}.")
    print("Adjusting row counts...")
    available = len(ds) - OFFSET
    if available < 20:
        print("Not enough rows. Exiting.")
        sys.exit(1)
    EVAL_ROWS  = min(EVAL_ROWS, available // 6)
    TRAIN_ROWS = available - EVAL_ROWS
    print(f"Adjusted: train={TRAIN_ROWS}, eval={EVAL_ROWS}")

# ── Slice data ────────────────────────────────────────────────────────────────

train_slice = ds.select(range(OFFSET, OFFSET + TRAIN_ROWS))
eval_slice  = ds.select(range(OFFSET + TRAIN_ROWS, OFFSET + TRAIN_ROWS + EVAL_ROWS))

print(f"Train slice: rows {OFFSET} – {OFFSET + TRAIN_ROWS - 1}")
print(f"Eval slice:  rows {OFFSET + TRAIN_ROWS} – {OFFSET + TRAIN_ROWS + EVAL_ROWS - 1}")

q_col  = col_map["question"]
c_col  = col_map["chosen"]
r_col  = col_map["rejected"]

# ── Write sft_train.jsonl ─────────────────────────────────────────────────────

sft_path = OUT_DIR / "sft_train.jsonl"
with sft_path.open("w") as f:
    for row in train_slice:
        question = str(row[q_col]).strip()
        chosen   = str(row[c_col]).strip()
        if not question or not chosen:
            continue
        f.write(json.dumps({"instruction": question, "output": chosen}) + "\n")

sft_count = sum(1 for _ in sft_path.open())
print(f"Written: sft_train.jsonl  ({sft_count} rows)")

# ── Write dpo_train.jsonl ─────────────────────────────────────────────────────

dpo_path = OUT_DIR / "dpo_train.jsonl"
with dpo_path.open("w") as f:
    for row in train_slice:
        question = str(row[q_col]).strip()
        chosen   = str(row[c_col]).strip()
        rejected = str(row[r_col]).strip()
        if not question or not chosen or not rejected:
            continue
        f.write(json.dumps({
            "prompt":   question,
            "chosen":   chosen,
            "rejected": rejected,
        }) + "\n")

dpo_count = sum(1 for _ in dpo_path.open())
print(f"Written: dpo_train.jsonl  ({dpo_count} rows)")

# ── Write eval.jsonl ──────────────────────────────────────────────────────────

eval_path = OUT_DIR / "eval.jsonl"
with eval_path.open("w") as f:
    for i, row in enumerate(eval_slice, start=1):
        question  = str(row[q_col]).strip()
        reference = str(row[c_col]).strip()
        if not question or not reference:
            continue
        f.write(json.dumps({
            "id":        i,
            "question":  question,
            "reference": reference,
        }) + "\n")

eval_count = sum(1 for _ in eval_path.open())
print(f"Written: eval.jsonl       ({eval_count} rows)")

# ── Write manifest.json ───────────────────────────────────────────────────────

manifest = {
    "id":          "orca_dpo_pairs",
    "name":        "Orca DPO Pairs (Intel)",
    "description": (
        "High-quality reasoning and step-by-step problem solving tasks. "
        "Questions cover math, logic, coding, and explanations. "
        "Ideal for demonstrating SFT vs DPO vs ORPO fine-tuning differences. "
        "Source: Intel/orca_dpo_pairs on HuggingFace."
    ),
    "hf_source":   "Intel/orca_dpo_pairs",
    "variants": [
        {
            "id":          "sft",
            "label":       "SFT Training (Alpaca format)",
            "description": "300 rows — instruction/output pairs using the chosen (good) answers. "
                           "Use this for Supervised Fine-Tuning.",
            "file":        "sft_train.jsonl",
            "rows":        sft_count,
            "format_type": "alpaca",
            "status_after_import": "uploaded",
        },
        {
            "id":          "dpo",
            "label":       "DPO/ORPO Training (preference format)",
            "description": "300 rows — prompt/chosen/rejected triplets. "
                           "Use this for DPO or ORPO training. Skips the tokenize step.",
            "file":        "dpo_train.jsonl",
            "rows":        dpo_count,
            "format_type": "dpo",
            "status_after_import": "uploaded",
        },
        {
            "id":          "eval",
            "label":       "Evaluation Set (20 questions)",
            "description": "20 held-out questions with reference answers. "
                           "Use these to compare all 3 fine-tuned models.",
            "file":        "eval.jsonl",
            "rows":        eval_count,
            "format_type": None,
            "status_after_import": "uploaded",
        },
    ],
}

manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Written: manifest.json")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("Demo dataset prepared successfully.")
print()
print("Files created:")
print(f"  {sft_path}")
print(f"  {dpo_path}")
print(f"  {eval_path}")
print(f"  {manifest_path}")
print()
print("Next steps:")
print("  1. Start the platform (backend + celery + dashboard)")
print("  2. Create a project called 'demo-sft-dpo-orpo'")
print("  3. Go to Datasets → Create Dataset → Presets tab")
print("  4. Import 'SFT Training' variant and 'DPO/ORPO Training' variant")
print("  5. Process SFT dataset: Clean → Format (Alpaca) → Tokenize")
print("  6. Process DPO dataset: Clean → Format (DPO/Preference)")
print("  7. Create 3 training jobs: demo-sft (SFT), demo-dpo (DPO), demo-orpo (ORPO)")
print("  8. Merge + Export all 3 to Ollama")
print("  9. Use Side-by-Side Inference with the 20 eval questions")
print("=" * 60)
