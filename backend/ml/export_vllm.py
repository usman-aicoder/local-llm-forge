"""
vLLM export helper.

vLLM reads HuggingFace format natively — no conversion needed.
This module validates the merged model directory and generates
a ready-to-run launch command.
"""
from __future__ import annotations

from pathlib import Path


def prepare_vllm_export(merged_model_dir: str, port: int = 8000) -> dict:
    """
    Validate the merged HF model directory and generate a vLLM launch command.

    vLLM needs only the merged HuggingFace model directory (no GGUF conversion).
    The directory must contain config.json produced by the merge step.

    Returns:
        {
            "vllm_model_path": str,
            "launch_command": str,   # copy-paste to start vLLM server
        }
    """
    src = Path(merged_model_dir)
    if not src.exists():
        raise FileNotFoundError(
            f"Merged model directory not found: {merged_model_dir}. "
            "Run the merge step first."
        )

    config_file = src / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"config.json not found in {merged_model_dir}. "
            "The merge may not have completed successfully."
        )

    launch_cmd = (
        f"python -m vllm.entrypoints.openai.api_server \\\n"
        f"  --model {src} \\\n"
        f"  --port {port} \\\n"
        f"  --dtype bfloat16 \\\n"
        f"  --max-model-len 4096"
    )

    return {
        "vllm_model_path": str(src),
        "launch_command": launch_cmd,
    }
