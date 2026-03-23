"""
Merge LoRA adapter into the base model to produce a standalone HF checkpoint.

No FastAPI/DB imports. Pure ML.
"""
from __future__ import annotations

from pathlib import Path


def run_merge(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
) -> dict:
    """
    Load base model + LoRA adapter, merge, and save as a full HF model.

    Returns: {"merged_path": str}
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # merge on CPU to avoid VRAM limits
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out}")
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    return {"merged_path": str(out)}
