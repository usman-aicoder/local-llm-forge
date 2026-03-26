"""
Auto model card generator.

Produces a HuggingFace-compatible MODEL_CARD.md for a completed training job.
Written to the adapter directory so it travels with the adapter weights.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


_METHOD_DESCRIPTIONS = {
    "sft":  "Supervised Fine-Tuning (SFT) via TRL SFTTrainer",
    "dpo":  "Direct Preference Optimization (DPO) via TRL DPOTrainer",
    "orpo": "Identity Preference Optimization (IPO) via TRL DPOTrainer (loss_type=ipo) — equivalent to ORPO",
}


def generate_model_card(
    job: dict,
    dataset: dict | None = None,
    evaluation: dict | None = None,
) -> str:
    """
    Build the MODEL_CARD.md content as a string.

    Args:
        job:        TrainingJob document (plain dict from MongoDB)
        dataset:    Dataset document (optional, for row count / format info)
        evaluation: Evaluation result dict with rouge1/rouge2/rougeL/bleu keys (optional)

    Returns:
        Markdown string ready to write to MODEL_CARD.md
    """
    method = job.get("training_method", "sft")
    method_desc = _METHOD_DESCRIPTIONS.get(method, method.upper())
    base_model = job.get("base_model", job.get("model_path", "unknown"))
    job_name = job.get("name", "fine-tuned-model")
    created_at = job.get("created_at", datetime.utcnow())
    if isinstance(created_at, datetime):
        created_at = created_at.strftime("%Y-%m-%d")

    # ── YAML frontmatter ──────────────────────────────────────────────────────
    quant_tag = "qlora" if job.get("use_qlora") else "lora"
    frontmatter = f"""---
base_model: {base_model}
library_name: peft
tags:
  - lora
  - {quant_tag}
  - {method}
  - local-llm-forge
---"""

    # ── Dataset section ───────────────────────────────────────────────────────
    if dataset:
        dataset_section = f"""## Dataset

| Property | Value |
|----------|-------|
| Name | {dataset.get('name', 'N/A')} |
| Rows | {dataset.get('row_count', 'N/A')} |
| Format | {dataset.get('format_type', 'N/A')} |
"""
    else:
        dataset_section = ""

    # ── Evaluation section ────────────────────────────────────────────────────
    if evaluation:
        def _fmt(v) -> str:
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
        eval_section = f"""## Evaluation Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | {_fmt(evaluation.get('rouge_1'))} |
| ROUGE-2 | {_fmt(evaluation.get('rouge_2'))} |
| ROUGE-L | {_fmt(evaluation.get('rouge_l'))} |
| BLEU | {_fmt(evaluation.get('bleu'))} |
| Perplexity | {_fmt(evaluation.get('perplexity'))} |

Evaluated on {len(evaluation.get('sample_results') or [])} samples.
"""
    else:
        eval_section = "_Evaluation not run for this job._\n"

    # ── Full card ─────────────────────────────────────────────────────────────
    card = f"""{frontmatter}

# {job_name}

Fine-tuned from **{base_model}** using [Local LLM Forge](https://github.com/usman-aicoder/local-llm-forge).

## Training Details

| Property | Value |
|----------|-------|
| Base Model | {base_model} |
| Training Method | {method_desc} |
| Date | {created_at} |
| QLoRA (4-bit) | {"Yes" if job.get('use_qlora') else "No"} |
| Unsloth | {"Yes" if job.get('use_unsloth') else "No"} |

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | {job.get('lora_r', 16)} |
| Alpha | {job.get('lora_alpha', 32)} |
| Dropout | {job.get('lora_dropout', 0.05)} |
| Target Modules | {', '.join(job.get('target_modules') or [])} |

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | {job.get('learning_rate', 2e-4)} |
| Epochs | {job.get('epochs', 3)} |
| Batch Size | {job.get('batch_size', 2)} |
| Gradient Accumulation | {job.get('grad_accum', 8)} |
| Max Sequence Length | {job.get('max_seq_len', 2048)} |
| BF16 | {"Yes" if job.get('bf16') else "No"} |

{dataset_section}
## Evaluation

{eval_section}
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "{base_model}"
adapter_path = "<path-to-this-adapter>"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_path)

inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## License

Same as the base model: {base_model}
"""
    return card


def save_model_card(job: dict, adapter_path: str, **kwargs) -> str:
    """
    Generate and write MODEL_CARD.md to adapter_path directory.
    Returns the path to the written file.
    """
    content = generate_model_card(job, **kwargs)
    out = Path(adapter_path) / "MODEL_CARD.md"
    out.write_text(content, encoding="utf-8")
    return str(out)
