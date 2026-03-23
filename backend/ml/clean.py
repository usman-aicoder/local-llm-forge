"""
Cleaning pipeline for fine-tuning datasets.

Input:  file path (cleaned CSV stage)
Output: cleaned file path + removal report dict

No FastAPI, no DB imports. Pure Python + pandas.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ml.eda import _load, _normalise


# ── Cleaning steps ────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return text.strip()


def _normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", text).strip()


def _remove_urls(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"https?://\S+|www\.\S+", "", text).strip()


def _apply_to_text_cols(df: pd.DataFrame, fn) -> pd.DataFrame:
    # "input" holds the rejected column for DPO datasets — clean it too
    for col in ["instruction", "input", "output"]:
        if col in df.columns:
            df[col] = df[col].apply(fn)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_clean(
    file_path: str,
    output_path: str,
    strip_html: bool = True,
    normalize_whitespace: bool = True,
    remove_urls: bool = True,
    deduplicate: bool = True,
    filter_short: bool = True,
    min_instruction_len: int = 5,
    min_output_len: int = 20,
) -> dict:
    """
    Clean a dataset file and write the result to output_path.

    Returns:
        cleaning_report dict: {step_name: rows_removed}
    """
    df = _normalise(_load(file_path))
    report: dict[str, int] = {}
    before = len(df)

    if strip_html:
        df = _apply_to_text_cols(df, _strip_html)
        report["strip_html"] = 0  # structural change, not row removal

    if normalize_whitespace:
        df = _apply_to_text_cols(df, _normalize_whitespace)
        report["normalize_whitespace"] = 0

    if remove_urls:
        df = _apply_to_text_cols(df, _remove_urls)
        report["remove_urls"] = 0

    # Remove rows where required fields are null / empty after cleaning
    before_null = len(df)
    df = df[
        df["instruction"].str.strip().str.len() > 0
    ]
    df = df[
        df["output"].str.strip().str.len() > 0
    ]
    report["remove_empty"] = before_null - len(df)

    if deduplicate:
        before_dup = len(df)
        df = df.drop_duplicates(subset=["instruction", "output"])
        report["deduplicate"] = before_dup - len(df)

    if filter_short:
        before_short = len(df)
        df = df[df["instruction"].str.len() >= min_instruction_len]
        df = df[df["output"].str.len() >= min_output_len]
        # Remove samples where output == instruction (copy-paste error)
        df = df[df["output"] != df["instruction"]]
        report["filter_short"] = before_short - len(df)

    report["total_removed"] = before - len(df)
    report["rows_remaining"] = len(df)

    # Write cleaned file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_json(out, orient="records", lines=True, force_ascii=False)

    return report
