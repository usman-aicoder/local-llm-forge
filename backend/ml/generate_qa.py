"""
Generate instruction/output Q&A pairs from raw text using a local Ollama model.

Flow:
  1. Split text into overlapping chunks (~600 words each)
  2. For each chunk call Ollama with a structured prompt
  3. Parse JSON array from the response
  4. Return list of {"instruction": "...", "output": "..."} dicts

This is intentionally synchronous so it can be called from FastAPI
BackgroundTasks without asyncio nesting issues.
"""
from __future__ import annotations

import json
import re
from typing import Callable

import httpx


CHUNK_WORDS   = 600    # target words per chunk
CHUNK_OVERLAP = 60     # word overlap between consecutive chunks
MAX_CHUNKS    = 25     # cap to prevent runaway jobs
OLLAMA_TIMEOUT = 120   # seconds per Ollama call

_QA_PROMPT = """\
You are a training-data generator for fine-tuning language models.

Given the text below, produce exactly {n} question-and-answer pairs.

Rules:
- Each question must be answerable solely from the provided text.
- Answers should be complete, factual sentences (not just keywords).
- Do NOT include questions about "this document", "the author", or anything meta.
- Output ONLY a valid JSON array — no markdown, no extra text.

Text:
\"\"\"
{text}
\"\"\"

Output JSON array (no other text):
"""


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    step  = CHUNK_WORDS - CHUNK_OVERLAP
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        if len(chunk.strip()) > 80:   # skip tiny trailing chunks
            chunks.append(chunk)
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks


def _parse_pairs(raw: str) -> list[dict]:
    """Extract a JSON array from a potentially noisy Ollama response."""
    # Find the outermost [...] block
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        # Try stripping markdown code fences and retry
        cleaned = re.sub(r"```(?:json)?", "", raw[start : end + 1]).strip()
        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

    pairs = []
    for item in items:
        if isinstance(item, dict):
            instruction = (item.get("instruction") or item.get("question") or "").strip()
            output      = (item.get("output")      or item.get("answer")   or "").strip()
            if instruction and output:
                pairs.append({"instruction": instruction, "output": output})
    return pairs


def generate_qa_pairs(
    text: str,
    ollama_model: str,
    ollama_url: str,
    pairs_per_chunk: int = 3,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Args:
        text:            Raw extracted text (from PDF or web scraping)
        ollama_model:    Ollama model tag to use for generation (e.g. "gemma2:2b")
        ollama_url:      Base URL of Ollama API (e.g. "http://localhost:11434")
        pairs_per_chunk: How many Q&A pairs to request per chunk
        on_progress:     Optional callback(current_chunk, total_chunks)

    Returns:
        List of {"instruction": ..., "output": ...} dicts
    """
    chunks = _chunk_text(text)
    total  = len(chunks)
    all_pairs: list[dict] = []

    for i, chunk in enumerate(chunks):
        prompt = _QA_PROMPT.format(n=pairs_per_chunk, text=chunk)
        try:
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={
                    "model":  ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 1024},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            pairs = _parse_pairs(raw)
            all_pairs.extend(pairs)
        except Exception:
            # Skip failed chunks — don't abort the whole job
            pass

        if on_progress:
            on_progress(i + 1, total)

    return all_pairs


def save_as_jsonl(pairs: list[dict], output_path: str) -> int:
    """Write pairs to a JSONL file. Returns number of rows written."""
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)
