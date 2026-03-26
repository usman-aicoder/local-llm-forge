"""
Full Fine-Tuning (FFT) — trains ALL model parameters (no LoRA adapter).

Use cases:
  - Models ≤ 3B on consumer GPU (e.g. Qwen2.5-1.5B, Phi-3-mini)
  - Larger models on A100 / H100

Key differences from LoRA/QLoRA training:
  - No quantization (full precision)
  - No PEFT adapter — all weights updated
  - Output is a full HuggingFace model directory, not an adapter
  - No merge step needed before GGUF export

VRAM requirement (rough): ~2× the model size in bfloat16
  e.g. 1.5B model ≈ 3 GB model weights + optimizer states ≈ 8–10 GB total
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def run_fft_training(
    job_id: str,
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    # Training
    learning_rate: float = 1e-5,   # lower than LoRA — all params update
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 16,
    max_seq_len: int = 1024,
    bf16: bool = True,
    resume_from_checkpoint: str | None = None,
    # Callbacks
    on_epoch_end: Callable[[int, float, float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    """
    Full fine-tune: all parameters trainable, no LoRA adapter.

    Returns: {"model_path": str, "is_full_model": True}
    """
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    from trl import SFTTrainer, SFTConfig

    if on_log:
        on_log(f"[FFT] Loading model from {model_path} (full precision, no quantization)")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # No LoRA wrapping — all parameters are trainable
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if on_log:
        on_log(f"[FFT] Trainable params: {trainable:,} / {total:,} (100% — full fine-tune)")

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
                parts = [
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in logs.items()
                ]
                on_log("  ".join(parts))

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if state.log_history and on_epoch_end:
                train_loss = next(
                    (h["loss"] for h in reversed(state.log_history) if "loss" in h), 0.0
                )
                eval_loss = next(
                    (h["eval_loss"] for h in reversed(state.log_history) if "eval_loss" in h), 0.0
                )
                on_epoch_end(int(state.epoch or 0), train_loss, eval_loss)

    # ── Training config ───────────────────────────────────────────────────────
    out = Path(output_dir)
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
        weight_decay=0.01,
        bf16=bf16 and torch.cuda.is_bf16_supported(),
        fp16=not (bf16 and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
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
        on_log("[FFT] Training started")
    if resume_from_checkpoint and on_log:
        on_log(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if on_log:
        on_log("[FFT] Saving full model...")

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    if on_log:
        on_log(f"[FFT] Full model saved to {out}")

    return {"model_path": str(out), "is_full_model": True}
