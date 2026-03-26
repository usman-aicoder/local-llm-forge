"""
Dataset augmentation via Ollama.

Uses a local LLM to generate paraphrased variants of existing instruction/output
pairs. Solves the small-dataset problem without requiring external APIs.

Input:  formatted JSONL with "instruction" and "output" keys (Alpaca format)
Output: augmented JSONL containing original rows + generated variants
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable


PARAPHRASE_PROMPT = """\
You are a training-data augmentation assistant.

Given the instruction-output pair below, generate {n} alternative versions.
Each version should ask the same question differently and give an equivalent answer.
Keep the same meaning but vary the wording, sentence structure, and phrasing.

Original instruction: {instruction}
Original output: {output}

Output ONLY a valid JSON array of objects with "instruction" and "output" keys.
Example format:
[
  {{"instruction": "...", "output": "..."}},
  {{"instruction": "...", "output": "..."}}
]
"""


def _parse_pairs(response: str) -> list[dict]:
    """Extract JSON array of instruction/output pairs from LLM response."""
    # Strip markdown code fences if present
    response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return [
                r for r in data
                if isinstance(r, dict) and "instruction" in r and "output" in r
            ]
    except json.JSONDecodeError:
        # Try to find a JSON array embedded in free text
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [
                        r for r in data
                        if isinstance(r, dict) and "instruction" in r and "output" in r
                    ]
            except json.JSONDecodeError:
                pass
    return []


def augment_dataset(
    input_path: str,
    output_path: str,
    ollama_url: str,
    model_name: str,
    paraphrases_per_row: int = 2,
    max_rows_to_augment: int = 200,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    """
    Augment a JSONL dataset using Ollama.

    Args:
        input_path:           Path to source JSONL (Alpaca format)
        output_path:          Path to write augmented JSONL
        ollama_url:           Ollama base URL (e.g. "http://localhost:11434")
        model_name:           Ollama model to use for paraphrasing
        paraphrases_per_row:  How many variants to generate per original row
        max_rows_to_augment:  Cap on how many rows get augmented (avoids very long runs)
        on_log:               Optional callback for progress messages

    Returns:
        {"original_rows": int, "augmented_rows": int, "new_rows_added": int}
    """
    import httpx

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    augmented: list[dict] = list(rows)   # always keep originals
    to_augment = rows[:max_rows_to_augment]
    failed = 0

    for i, row in enumerate(to_augment):
        instruction = row.get("instruction") or row.get("text", "")
        output = row.get("output") or row.get("response", "")

        if not instruction:
            continue

        if on_log:
            on_log(f"Augmenting row {i + 1}/{len(to_augment)}")

        prompt = PARAPHRASE_PROMPT.format(
            n=paraphrases_per_row,
            instruction=instruction,
            output=output,
        )

        try:
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            pairs = _parse_pairs(resp.json().get("response", ""))
            augmented.extend(pairs)
            if on_log and pairs:
                on_log(f"  +{len(pairs)} rows from row {i + 1}")
        except Exception as exc:
            failed += 1
            if on_log:
                on_log(f"  Row {i + 1} augmentation failed: {exc}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in augmented:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if on_log:
        on_log(
            f"Augmentation complete: {len(rows)} original + "
            f"{len(augmented) - len(rows)} new = {len(augmented)} total rows"
        )

    return {
        "original_rows": len(rows),
        "augmented_rows": len(augmented),
        "new_rows_added": len(augmented) - len(rows),
        "rows_failed": failed,
    }
