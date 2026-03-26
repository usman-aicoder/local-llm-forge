"""
SFTTrainer training loop.

Called by workers/training_tasks.py. No FastAPI/DB imports.
Progress is reported via a callback that pushes to Redis.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable


def run_training(
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
    learning_rate: float = 2e-4,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_seq_len: int = 2048,
    bf16: bool = True,
    # Acceleration / resume
    use_unsloth: bool = False,
    resume_from_checkpoint: str | None = None,
    # Callbacks
    on_epoch_end: Callable[[int, float, float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    """
    Fine-tune a model using SFTTrainer.

    on_epoch_end(epoch, train_loss, eval_loss) — called after each epoch
    on_log(message)                            — called for each log line

    Returns: {"adapter_path": str}
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    from trl import SFTTrainer, SFTConfig

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    if on_log:
        on_log(f"Loading model from {model_path}")

    # ── Model loading: try Unsloth, fall back to standard HuggingFace ─────────
    backend = "standard"

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel  # type: ignore[import]
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_len,
                dtype=None,           # auto-detect
                load_in_4bit=use_qlora,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                use_gradient_checkpointing="unsloth",
                bias="none",
            )
            # Unsloth may set '<EOS_TOKEN>' as a template placeholder;
            # TRL 0.24 validates it must exist in the vocab — reset to real EOS.
            if tokenizer.eos_token not in tokenizer.get_vocab():
                tokenizer.eos_token = tokenizer.decode([tokenizer.eos_token_id])
            if tokenizer.pad_token is None or tokenizer.pad_token not in tokenizer.get_vocab():
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            backend = "unsloth"
        except ImportError:
            if on_log:
                on_log("Unsloth not installed — falling back to standard TRL")

    if backend == "standard":
        # ── Quantisation config ───────────────────────────────────────────────
        bnb_config = None
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Leave headroom for training: cap GPU usage at 6 GB, overflow to CPU RAM
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
        tokenizer.padding_side = "right"

        if use_qlora:
            model = prepare_model_for_kbit_training(model)

        # ── LoRA config ───────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    if on_log:
        on_log(f"Training backend: {backend}")

    trainable, total = model.get_nb_trainable_parameters()
    if on_log:
        on_log(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(
        "json",
        data_files={"train": train_data_path, "validation": val_data_path},
        split=None,
    )

    # ── Progress callback ─────────────────────────────────────────────────────
    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs and on_log:
                parts = []
                for k, v in logs.items():
                    parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
                on_log("  ".join(parts))

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if state.log_history and on_epoch_end:
                # Find the most recent train/eval loss entries
                train_loss = next(
                    (h["loss"] for h in reversed(state.log_history) if "loss" in h), 0.0
                )
                eval_loss = next(
                    (h["eval_loss"] for h in reversed(state.log_history) if "eval_loss" in h), 0.0
                )
                epoch = int(state.epoch or 0)
                on_epoch_end(epoch, train_loss, eval_loss)

    # ── Training args ─────────────────────────────────────────────────────────
    out = Path(adapter_output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.001,
        bf16=bf16 and torch.cuda.is_bf16_supported(),
        fp16=not (bf16 and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=10,
        max_length=max_seq_len,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[ProgressCallback()],
    )

    if on_log:
        on_log("Training started")
    if resume_from_checkpoint and on_log:
        on_log(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if on_log:
        on_log("Saving adapter...")

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    if on_log:
        on_log(f"Adapter saved to {out}")

    return {"adapter_path": str(out)}
