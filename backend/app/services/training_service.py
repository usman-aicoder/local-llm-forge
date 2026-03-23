"""
Training service — VRAM estimation and job dispatch utilities.
"""
from __future__ import annotations

import subprocess


# Approximate model sizes in billions of parameters, keyed by Ollama model name
MODEL_PARAM_BILLIONS: dict[str, float] = {
    "mistral:7b":      7.0,
    "qwen3.5:9b":      9.0,
    "gemma2:2b":       2.0,
    "llama3.2:latest": 3.2,
    "gpt-oss:20b":     20.0,
}

# Transformer layers per model (approximate, for LoRA overhead calculation)
MODEL_LAYERS: dict[str, int] = {
    "mistral:7b":      32,
    "qwen3.5:9b":      32,
    "gemma2:2b":       18,
    "llama3.2:latest": 28,
    "gpt-oss:20b":     40,
}


def estimate_vram_gb(
    base_model: str,
    lora_r: int = 16,
    use_qlora: bool = True,
) -> dict:
    """
    Estimate GPU VRAM required for fine-tuning.

    Returns:
        {
          "model_gb": float,   base model memory
          "lora_gb":  float,   adapter overhead
          "total_gb": float,   total estimate
          "warning":  str | None
        }
    """
    params_b = MODEL_PARAM_BILLIONS.get(base_model, 7.0)

    # Base model VRAM:
    #   QLoRA 4-bit ≈ params_b × 0.5 GB
    #   LoRA bf16   ≈ params_b × 2.0 GB
    model_gb = params_b * (0.5 if use_qlora else 2.0)

    # LoRA adapter overhead (small but non-zero)
    layers = MODEL_LAYERS.get(base_model, 32)
    # 7 target modules × 2 matrices × r × hidden_dim × 2 bytes ÷ 1e9
    hidden_dim = 4096  # conservative estimate
    lora_gb = (7 * 2 * lora_r * hidden_dim * 2 * layers) / 1e9

    # Activation + optimiser overhead: ~20% buffer
    total_gb = (model_gb + lora_gb) * 1.2

    warning = None
    vram_available = get_gpu_vram_gb()
    if vram_available and total_gb > vram_available:
        warning = (
            f"Estimated {total_gb:.1f} GB exceeds available GPU VRAM "
            f"({vram_available:.1f} GB). Reduce batch size or use QLoRA."
        )

    return {
        "model_gb":  round(model_gb, 1),
        "lora_gb":   round(lora_gb, 2),
        "total_gb":  round(total_gb, 1),
        "warning":   warning,
    }


def get_gpu_vram_gb() -> float | None:
    """Return total GPU VRAM in GB, or None if no GPU found."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        mb = int(out.decode().strip().split("\n")[0])
        return round(mb / 1024, 1)
    except Exception:
        return None


def get_gpu_utilization() -> dict:
    """Return current GPU utilization stats."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5,
        )
        line = out.decode().strip().split("\n")[0]
        util_pct, mem_used_mb, mem_total_mb = [int(x.strip()) for x in line.split(",")]
        return {
            "gpu_util_pct":  util_pct,
            "mem_used_gb":   round(mem_used_mb / 1024, 1),
            "mem_total_gb":  round(mem_total_mb / 1024, 1),
        }
    except Exception:
        return {"gpu_util_pct": None, "mem_used_gb": None, "mem_total_gb": None}
