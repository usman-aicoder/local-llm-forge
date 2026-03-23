"""
ORPO (Odds Ratio Preference Optimization) training.

Dataset must be JSONL with columns: prompt, chosen, rejected.
No separate tokenization step needed — DPOTrainer handles it internally.

In TRL >= 0.26, ORPO is implemented via DPOTrainer with loss_type="orpo".
ORPO combines SFT + preference alignment in a single pass — no reference model needed.
It's more efficient than DPO and works well with smaller datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def run_orpo_training(
    job_id: str,
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    adapter_output_dir: str,
    # LoRA
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    # Training
    orpo_alpha: float = 0.1,    # ORPO regularisation weight (beta in TRL DPOConfig)
    learning_rate: float = 8e-6,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_seq_len: int = 1024,
    bf16: bool = True,
    # Callbacks
    on_epoch_end: Callable[[int, float, float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import DPOTrainer, DPOConfig

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    if on_log:
        on_log(f"[ORPO] Loading model from {model_path}")

    # ── Quantisation ──────────────────────────────────────────────────────────
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    max_mem = {0: "6GiB", "cpu": "48GiB"} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_mem,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_files: dict = {"train": train_data_path}
    if Path(val_data_path).exists() and val_data_path != train_data_path:
        data_files["validation"] = val_data_path

    raw = load_dataset("json", data_files=data_files)

    required = {"prompt", "chosen", "rejected"}
    actual = set(raw["train"].column_names)
    missing = required - actual
    if missing:
        raise ValueError(
            f"ORPO dataset missing columns: {missing}. "
            f"Found: {actual}. "
            "Dataset must have: prompt, chosen, rejected."
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    class ProgressCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if on_epoch_end and state.log_history:
                last = {k: v for d in state.log_history for k, v in d.items()}
                tl = last.get("loss", last.get("train_loss", 0.0))
                el = last.get("eval_loss", tl)
                on_epoch_end(int(state.epoch), float(tl), float(el))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if on_log and logs:
                on_log(str(logs))

    # ── ORPO-style config via DPOConfig with loss_type="ipo" ─────────────────
    # TRL 0.29 removed loss_type="orpo"; IPO is the closest equivalent —
    # both are reference-free preference optimization methods.
    out = Path(adapter_output_dir)
    out.mkdir(parents=True, exist_ok=True)

    orpo_config = DPOConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        loss_type="ipo",
        beta=orpo_alpha,
        max_length=max_seq_len,
        bf16=bf16 and torch.cuda.is_bf16_supported(),
        fp16=not (bf16 and torch.cuda.is_bf16_supported()),
        logging_steps=1,
        eval_strategy="epoch" if "validation" in raw else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end="validation" in raw,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=orpo_config,
        train_dataset=raw["train"],
        eval_dataset=raw.get("validation"),
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[ProgressCallback()],
    )

    if on_log:
        on_log("ORPO training started")

    trainer.train()

    if on_log:
        on_log("Saving adapter...")
    trainer.model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    if on_log:
        on_log(f"Adapter saved to {out}")
        on_log("ORPO training completed successfully")

    return {"adapter_path": str(out)}
