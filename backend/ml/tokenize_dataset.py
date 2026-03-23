"""
Tokenisation + train/val split.

Uses the HuggingFace tokenizer for the chosen model.
Downloads tokenizer-only files (small, ~few MB) if not cached locally.

No FastAPI, no DB imports. Pure Python + transformers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Ollama name → HF model ID (for tokenizer download)
OLLAMA_TO_HF: dict[str, str] = {
    "mistral:7b":      "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen3.5:9b":      "Qwen/Qwen2.5-7B-Instruct",
    "gemma2:2b":       "google/gemma-2-2b-it",
    "llama3.2:latest": "meta-llama/Llama-3.2-3B-Instruct",
    "gpt-oss:20b":     None,   # custom model — fallback to gpt2
}

FALLBACK_TOKENIZER = "gpt2"


def _get_tokenizer(model_path: str | None, base_model: str | None):
    from transformers import AutoTokenizer

    # 1. Try local HF weights directory
    if model_path and Path(model_path).exists():
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tok, str(model_path)

    # 2. Try mapping Ollama name → HF hub ID
    if base_model:
        hf_id = OLLAMA_TO_HF.get(base_model)
        if hf_id:
            try:
                tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                return tok, hf_id
            except Exception:
                pass

    # 3. Fallback to gpt2 (always available, no auth needed)
    tok = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)
    return tok, FALLBACK_TOKENIZER


def run_tokenize(
    formatted_path: str,
    output_dir: str,
    model_path: str | None = None,
    base_model: str | None = None,
    max_seq_len: int = 2048,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Tokenise a formatted JSONL dataset and split into train / val.

    Returns:
        {
          "train_path": str,
          "val_path": str,
          "tokenizer_used": str,
          "train_count": int,
          "val_count": int,
          "truncated_count": int,   # samples that hit max_seq_len
        }
    """
    tokenizer, tokenizer_used = _get_tokenizer(model_path, base_model)

    # Pad token fix for models without one (Mistral, LLaMA)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load formatted JSONL
    formatted_path_obj = Path(formatted_path)
    records = []
    with formatted_path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Tokenise and check lengths
    tokenised = []
    truncated_count = 0

    for rec in records:
        # Extract text from record
        if "text" in rec:
            text = rec["text"]
        elif "messages" in rec:
            # Simple concat for length check
            text = " ".join(m["content"] for m in rec["messages"])
        else:
            text = str(rec)

        ids = tokenizer.encode(text, truncation=False)
        if len(ids) > max_seq_len:
            truncated_count += 1

        tokenised.append({
            "text": text,
            "input_ids": ids[:max_seq_len],
            "length": min(len(ids), max_seq_len),
        })

    # Shuffle + split
    random.seed(seed)
    random.shuffle(tokenised)
    val_n = max(1, int(len(tokenised) * val_split))
    val_data   = tokenised[:val_n]
    train_data = tokenised[val_n:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    val_path   = out / "val.jsonl"

    def _write(path: Path, data: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                # Store text + length; input_ids excluded (regenerated at training)
                f.write(json.dumps({"text": item["text"], "length": item["length"]},
                                   ensure_ascii=False) + "\n")

    _write(train_path, train_data)
    _write(val_path, val_data)

    return {
        "train_path":      str(train_path),
        "val_path":        str(val_path),
        "tokenizer_used":  tokenizer_used,
        "train_count":     len(train_data),
        "val_count":       len(val_data),
        "truncated_count": truncated_count,
    }
