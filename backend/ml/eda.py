"""
EDA — Exploratory Data Analysis for fine-tuning datasets.

Input:  file path (CSV / JSON / JSONL)
Output: stats dict matching DatasetStats schema

No FastAPI, no DB imports. Pure Python + pandas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ── Column normalisation ──────────────────────────────────────────────────────

INSTRUCTION_ALIASES = [
    "instruction", "prompt", "question", "input_text", "text",
    "query", "human", "user", "q",
]
INPUT_ALIASES = ["input", "context", "system"]
OUTPUT_ALIASES = [
    "output", "response", "completion", "answer", "target",
    "best answer", "correct answer", "correct answers", "a",
    "assistant", "bot", "gpt",
]


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    # Build a map of lowercased column name → actual column name
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def _detect_format(df: pd.DataFrame) -> str:
    """Detect dataset format: 'sft', 'dpo', or 'eval'."""
    cols = {c.lower() for c in df.columns}
    # DPO/ORPO: must have prompt + chosen + rejected
    if "prompt" in cols and "chosen" in cols and "rejected" in cols:
        return "dpo"
    # Eval: question + reference (with optional id)
    if "reference" in cols and ("question" in cols or "id" in cols):
        return "eval"
    return "sft"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard: instruction, input (opt), output.

    Handles three formats:
      SFT  — instruction/output (many alias variants)
      DPO  — prompt/chosen/rejected  → maps prompt→instruction, chosen→output
      eval — question/reference      → maps question→instruction, reference→output
    """
    fmt = _detect_format(df)
    lower_map = {c.lower(): c for c in df.columns}

    if fmt == "dpo":
        rename = {
            lower_map["prompt"]:   "instruction",
            lower_map["chosen"]:   "output",
            lower_map["rejected"]: "input",   # store rejected in input for stats
        }
        df = df.rename(columns=rename)
        return df[["instruction", "input", "output"]]

    if fmt == "eval":
        q_col = lower_map.get("question") or lower_map.get("id")
        r_col = lower_map.get("reference")
        rename = {q_col: "instruction", r_col: "output"}
        df = df.rename(columns=rename)
        df["input"] = ""
        return df[["instruction", "input", "output"]]

    # SFT — original alias-based logic
    instr_col  = _find_col(df, INSTRUCTION_ALIASES)
    input_col  = _find_col(df, INPUT_ALIASES)
    output_col = _find_col(df, OUTPUT_ALIASES)

    if instr_col is None or output_col is None:
        raise ValueError(
            f"Cannot find instruction/output columns. Found: {list(df.columns)}"
        )

    rename = {instr_col: "instruction", output_col: "output"}
    if input_col and input_col not in ("instruction", "output"):
        rename[input_col] = "input"
    df = df.rename(columns=rename)

    if "input" not in df.columns:
        df["input"] = ""
    return df[["instruction", "input", "output"]]


# ── Token approximation ───────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Rough token count: words * 1.3 (accounts for subword splits)."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return max(1, int(len(text.split()) * 1.3))


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext == ".jsonl":
        return pd.read_json(p, lines=True)
    if ext == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported format: {ext}")


# ── Histogram helper ──────────────────────────────────────────────────────────

def _histogram(series: pd.Series, bins: int = 20) -> dict:
    arr = series.dropna().values
    if len(arr) == 0:
        return {"buckets": [], "counts": []}
    counts, edges = np.histogram(arr, bins=bins)
    buckets = [int((edges[i] + edges[i + 1]) / 2) for i in range(len(edges) - 1)]
    return {"buckets": buckets, "counts": [int(c) for c in counts]}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eda(file_path: str, max_length: int = 2048) -> dict:
    """
    Analyse a dataset file and return a DatasetStats-compatible dict.

    Args:
        file_path:  absolute path to the dataset file
        max_length: token threshold above which samples are flagged

    Returns:
        dict with all DatasetStats fields
    """
    df_raw = _load(file_path)
    total_rows = len(df_raw)

    # Count nulls before normalisation
    null_count = int(df_raw.isnull().sum().sum())

    # Count exact duplicates
    duplicate_count = int(df_raw.duplicated().sum())

    # Normalise columns
    df = _normalise(df_raw)

    # Token counts per field
    df["_instr_tokens"]  = df["instruction"].apply(_approx_tokens)
    df["_input_tokens"]  = df["input"].apply(_approx_tokens)
    df["_output_tokens"] = df["output"].apply(_approx_tokens)
    df["_total_tokens"]  = df["_instr_tokens"] + df["_input_tokens"] + df["_output_tokens"]

    avg_instruction_tokens = float(df["_instr_tokens"].mean())
    avg_output_tokens      = float(df["_output_tokens"].mean())
    p95_total_tokens       = float(df["_total_tokens"].quantile(0.95))

    flagged_too_long  = int((df["_total_tokens"] > max_length).sum())
    flagged_too_short = int((df["_output_tokens"] < 10).sum())

    token_histogram = _histogram(df["_total_tokens"])

    return {
        "total_rows":              total_rows,
        "null_count":              null_count,
        "duplicate_count":         duplicate_count,
        "removed_count":           0,
        "avg_instruction_tokens":  round(avg_instruction_tokens, 2),
        "avg_output_tokens":       round(avg_output_tokens, 2),
        "p95_total_tokens":        round(p95_total_tokens, 2),
        "flagged_too_long":        flagged_too_long,
        "flagged_too_short":       flagged_too_short,
        "token_histogram":         token_histogram,
        "cleaning_report":         {},
    }
