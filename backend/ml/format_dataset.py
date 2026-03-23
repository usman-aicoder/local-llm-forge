"""
Dataset formatting — converts cleaned datasets to instruction-tuning format.

Supports: Alpaca and Chat/Messages formats.
Applies model-specific prompt templates.

No FastAPI, no DB imports. Pure Python + pandas.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.eda import _load, _normalise


# ── Model → template mapping ──────────────────────────────────────────────────

# Maps Ollama model names and HF model IDs to their prompt template style
MISTRAL_MODELS  = {"mistral", "mistralai"}
LLAMA3_MODELS   = {"llama3", "llama-3", "meta-llama"}
GEMMA_MODELS    = {"gemma"}
QWEN_MODELS     = {"qwen"}


def _detect_template(base_model: str) -> str:
    name = base_model.lower()
    if any(k in name for k in MISTRAL_MODELS):
        return "mistral"
    if any(k in name for k in LLAMA3_MODELS):
        return "llama3"
    if any(k in name for k in GEMMA_MODELS):
        return "gemma"
    if any(k in name for k in QWEN_MODELS):
        return "qwen"
    return "alpaca"


# ── Alpaca format ─────────────────────────────────────────────────────────────

def _alpaca_format(instruction: str, input_text: str, output: str) -> str:
    if input_text.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return prompt


# ── Mistral format ────────────────────────────────────────────────────────────

def _mistral_format(instruction: str, input_text: str, output: str) -> str:
    user_content = f"{instruction}\n{input_text}".strip() if input_text.strip() else instruction
    return f"<s>[INST] {user_content} [/INST] {output}</s>"


# ── LLaMA-3 format ────────────────────────────────────────────────────────────

def _llama3_format(instruction: str, input_text: str, output: str) -> str:
    user_content = f"{instruction}\n{input_text}".strip() if input_text.strip() else instruction
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{output}"
        "<|eot_id|>"
    )


# ── Gemma / Qwen format (ChatML-like) ─────────────────────────────────────────

def _chatml_format(instruction: str, input_text: str, output: str) -> str:
    user_content = f"{instruction}\n{input_text}".strip() if input_text.strip() else instruction
    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


# ── Chat / Messages format ────────────────────────────────────────────────────

def _chat_record(instruction: str, input_text: str, output: str) -> dict:
    user_content = f"{instruction}\n{input_text}".strip() if input_text.strip() else instruction
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


# ── Dispatch ──────────────────────────────────────────────────────────────────

_TEMPLATE_FNS = {
    "mistral": _mistral_format,
    "llama3":  _llama3_format,
    "gemma":   _chatml_format,
    "qwen":    _chatml_format,
    "alpaca":  _alpaca_format,
}


def _format_row(row: pd.Series, format_type: str, template: str) -> str | dict:
    instr  = str(row.get("instruction", "")).strip()
    inp    = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()

    if format_type == "chat":
        return _chat_record(instr, inp, output)

    fn = _TEMPLATE_FNS.get(template, _alpaca_format)
    return fn(instr, inp, output)


# ── Main ──────────────────────────────────────────────────────────────────────

def _dpo_normalise(file_path: str) -> pd.DataFrame:
    """Load a dataset and map to prompt/chosen/rejected columns for DPO."""
    df = _load(file_path)
    cols = {c.lower() for c in df.columns}

    # Map common column name variations
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("prompt", "instruction", "question", "input"):
            col_map[col] = "prompt"
        elif cl in ("chosen", "good_answer", "accepted", "positive", "best_answer"):
            col_map[col] = "chosen"
        elif cl in ("rejected", "bad_answer", "negative", "worse_answer"):
            col_map[col] = "rejected"
    df = df.rename(columns=col_map)

    missing = {"prompt", "chosen", "rejected"} - set(df.columns)
    if missing:
        raise ValueError(
            f"DPO format requires columns: prompt, chosen, rejected. "
            f"Missing: {missing}. Found: {list(df.columns)}"
        )
    return df[["prompt", "chosen", "rejected"]].dropna()


def _run_dpo_format(file_path: str, output_path: str, preview_count: int = 10) -> dict:
    df = _dpo_normalise(file_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preview = []
    with out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "prompt": str(row["prompt"]).strip(),
                "chosen": str(row["chosen"]).strip(),
                "rejected": str(row["rejected"]).strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(preview) < preview_count:
                preview.append(
                    f"prompt: {record['prompt'][:80]}...\n"
                    f"chosen: {record['chosen'][:60]}...\n"
                    f"rejected: {record['rejected'][:60]}..."
                )
    return {"formatted_path": str(out), "total_rows": len(df), "template": "dpo", "preview": preview}


def run_format(
    file_path: str,
    output_path: str,
    format_type: str = "alpaca",
    base_model: str = "mistral:7b",
    preview_count: int = 10,
) -> dict:
    """
    Format a cleaned dataset file and write JSONL output.

    Returns:
        {
          "formatted_path": str,
          "total_rows": int,
          "template": str,
          "preview": [str, ...]   # first preview_count formatted samples as text
        }
    """
    if format_type == "dpo":
        return _run_dpo_format(file_path, output_path, preview_count)

    df = _normalise(_load(file_path))
    template = _detect_template(base_model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    formatted_rows = []
    preview = []

    for _, row in df.iterrows():
        record = _format_row(row, format_type, template)
        if format_type == "chat":
            formatted_rows.append(record)
        else:
            formatted_rows.append({"text": record})

        if len(preview) < preview_count:
            preview.append(record if isinstance(record, str) else json.dumps(record, ensure_ascii=False))

    # Write JSONL
    with out.open("w", encoding="utf-8") as f:
        for r in formatted_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "formatted_path": str(out),
        "total_rows": len(formatted_rows),
        "template": template,
        "preview": preview,
    }


def get_preview(
    file_path: str,
    format_type: str = "alpaca",
    base_model: str = "mistral:7b",
    count: int = 10,
) -> list[str]:
    """Return formatted text for the first `count` samples without writing a file."""
    if format_type == "dpo":
        df = _dpo_normalise(file_path).head(count)
        return [
            f"prompt: {row['prompt'][:80]}...\nchosen: {row['chosen'][:60]}...\nrejected: {row['rejected'][:60]}..."
            for _, row in df.iterrows()
        ]
    df = _normalise(_load(file_path))
    template = _detect_template(base_model)
    preview = []
    for _, row in df.head(count).iterrows():
        record = _format_row(row, format_type, template)
        preview.append(record if isinstance(record, str) else json.dumps(record, ensure_ascii=False))
    return preview
