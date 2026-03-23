from pathlib import Path
from app.config import settings

# Known mapping from Ollama model names to HuggingFace model IDs
OLLAMA_TO_HF: dict[str, str] = {
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen3.5:9b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma2:2b": "google/gemma-2-2b-it",
    "llama3.2:latest": "meta-llama/Llama-3.2-3B-Instruct",
}


def scan_local_models() -> list[dict]:
    """Scan storage/models/hf/ and return info about each downloaded model."""
    hf_dir = settings.abs(settings.models_hf_dir)
    if not hf_dir.exists():
        return []

    models = []
    for entry in sorted(hf_dir.iterdir()):
        if not entry.is_dir():
            continue
        # A valid HF model directory contains config.json
        config_file = entry / "config.json"
        if not config_file.exists():
            continue

        size_bytes = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        models.append({
            "name": entry.name,
            "path": str(entry),
            "size_gb": round(size_bytes / (1024 ** 3), 2),
        })
    return models


def get_model_path(model_name: str) -> Path | None:
    """Return the absolute path to a downloaded HF model, or None if not found."""
    hf_dir = settings.abs(settings.models_hf_dir)
    candidate = hf_dir / model_name
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate
    return None


def hf_id_for_ollama(ollama_name: str) -> str | None:
    return OLLAMA_TO_HF.get(ollama_name)
