"""
Evaluation script.

Loads a fine-tuned PEFT model, generates predictions on the validation split,
and computes ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores.

The val.jsonl contains rows of {"text": "<full formatted prompt+response>"}.
We split on the model-specific response separator to get prompt vs ground-truth.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK punkt tokenizer once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Ordered by specificity — checked in order
_RESPONSE_MARKERS = [
    "<|start_header_id|>assistant<|end_header_id|>\n\n",  # LLaMA-3
    "<|im_start|>assistant\n",                             # Qwen / Gemma ChatML
    "[/INST]",                                             # Mistral
    "### Response:\n",                                     # Alpaca
]

_EOS_TOKENS = ["</s>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"]


def _split_prompt_response(text: str) -> tuple[str, str] | None:
    """Return (prompt_part, response_part) or None if no marker found."""
    for marker in _RESPONSE_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            prompt = text[: idx + len(marker)]
            response = text[idx + len(marker):]
            for eos in _EOS_TOKENS:
                response = response.replace(eos, "")
            return prompt, response.strip()
    return None


def run_evaluation(
    model_path: str,
    adapter_path: str,
    val_jsonl_path: str,
    max_samples: int = 50,
    max_new_tokens: int = 256,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Args:
        model_path:      HF model directory (base model)
        adapter_path:    LoRA adapter directory
        val_jsonl_path:  Path to val.jsonl (each line: {"text": "..."})
        max_samples:     Cap on number of samples to evaluate
        max_new_tokens:  Generation length cap per sample
        on_progress:     Optional callback(current, total)

    Returns dict with keys:
        rouge_1, rouge_2, rouge_l, bleu  — float or None
        sample_results                    — list of dicts
    """
    # ── Load samples ──────────────────────────────────────────────────────────
    samples: list[str] = []
    with open(val_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if text:
                samples.append(text)
            if len(samples) >= max_samples:
                break

    if not samples:
        return {
            "rouge_1": None, "rouge_2": None, "rouge_l": None,
            "bleu": None, "sample_results": [],
        }

    # ── Load model ────────────────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # ── Scoring ───────────────────────────────────────────────────────────────
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    r1_scores, r2_scores, rl_scores, bleu_scores = [], [], [], []
    sample_results = []

    total = len(samples)
    for i, text in enumerate(samples):
        split = _split_prompt_response(text)
        if split is None:
            if on_progress:
                on_progress(i + 1, total)
            continue

        prompt, ground_truth = split

        # Generate
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = out[0][input_ids.shape[1]:]
        generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # ROUGE
        scores = rouge.score(ground_truth, generated)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

        # BLEU
        ref_tokens = ground_truth.split()
        hyp_tokens = generated.split()
        bleu = sentence_bleu(
            [ref_tokens], hyp_tokens, smoothing_function=smoother
        ) if ref_tokens else 0.0
        bleu_scores.append(bleu)

        sample_results.append({
            "prompt": prompt[-500:],   # trim very long prompts for storage
            "response": generated,
            "ground_truth": ground_truth[:500],
            "accuracy": 0,
            "relevance": 0,
            "fluency": 0,
            "completeness": 0,
        })

        if on_progress:
            on_progress(i + 1, total)

    def _avg(lst: list[float]) -> float | None:
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "rouge_1": _avg(r1_scores),
        "rouge_2": _avg(r2_scores),
        "rouge_l": _avg(rl_scores),
        "bleu": _avg(bleu_scores),
        "sample_results": sample_results,
    }


def run_evaluation_via_ollama(
    formatted_jsonl_path: str,
    ollama_model_name: str,
    ollama_url: str,
    max_samples: int = 20,
    max_new_tokens: int = 256,
) -> dict:
    """
    Evaluate a model exported to Ollama using prompt/chosen pairs from a
    DPO-format JSONL (columns: prompt, chosen, rejected).

    Sends each prompt to the Ollama /api/generate endpoint and compares
    the response to the 'chosen' ground truth with ROUGE and BLEU.
    """
    import requests

    samples: list[dict] = []
    with open(formatted_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("instruction") or ""
            ground_truth = row.get("chosen") or row.get("output") or ""
            if prompt and ground_truth:
                samples.append({"prompt": prompt, "ground_truth": ground_truth})
            if len(samples) >= max_samples:
                break

    if not samples:
        return {
            "rouge_1": None, "rouge_2": None, "rouge_l": None,
            "bleu": None, "sample_results": [],
        }

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    r1_scores, r2_scores, rl_scores, bleu_scores = [], [], [], []
    sample_results = []

    for item in samples:
        prompt = item["prompt"]
        ground_truth = item["ground_truth"]

        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_new_tokens, "temperature": 0.0},
                },
                timeout=120,
            )
            generated = resp.json().get("response", "").strip()
        except Exception:
            generated = ""

        scores = rouge.score(ground_truth, generated)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

        ref_tokens = ground_truth.split()
        hyp_tokens = generated.split()
        bleu = sentence_bleu(
            [ref_tokens], hyp_tokens, smoothing_function=smoother
        ) if ref_tokens else 0.0
        bleu_scores.append(bleu)

        sample_results.append({
            "prompt": prompt[-500:],
            "response": generated,
            "ground_truth": ground_truth[:500],
            "accuracy": 0,
            "relevance": 0,
            "fluency": 0,
            "completeness": 0,
        })

    def _avg(lst: list[float]) -> float | None:
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "rouge_1": _avg(r1_scores),
        "rouge_2": _avg(r2_scores),
        "rouge_l": _avg(rl_scores),
        "bleu": _avg(bleu_scores),
        "sample_results": sample_results,
    }
