"""
Export task: merge LoRA adapter → full HF model → GGUF → ollama create.
"""
from __future__ import annotations

import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from bson import ObjectId
from pymongo import MongoClient

from app.config import settings
from workers.celery_app import celery_app


def _get_db():
    from pymongo import MongoClient
    client = MongoClient(settings.mongo_url)
    return client[settings.mongo_db_name]


@celery_app.task(bind=True, name="workers.export_tasks.run_merge_task")
def run_merge_task(self, job_id: str) -> dict:
    db = _get_db()
    try:
        job = db["training_jobs"].find_one({"_id": ObjectId(job_id)})
        if not job or not job.get("adapter_path"):
            return {"error": "No adapter found — training must complete first"}

        from ml.merge import run_merge

        merged_dir = settings.abs(settings.merged_models_dir) / job_id
        result = run_merge(
            base_model_path=job["model_path"],
            adapter_path=job["adapter_path"],
            output_path=str(merged_dir),
        )

        db["training_jobs"].update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"merged_path": result["merged_path"], "updated_at": datetime.utcnow()}},
        )
        return result

    except Exception:
        err = traceback.format_exc()
        db["training_jobs"].update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"error_message": err}},
        )
        return {"error": err}


@celery_app.task(bind=True, name="workers.export_tasks.run_export_task")
def run_export_task(self, job_id: str) -> dict:
    """Convert merged model to GGUF and register with Ollama."""
    db = _get_db()
    try:
        job = db["training_jobs"].find_one({"_id": ObjectId(job_id)})
        if not job or not job.get("merged_path"):
            return {"error": "No merged model found — run merge first"}

        merged_path = job["merged_path"]
        gguf_dir = settings.abs(settings.gguf_exports_dir)
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / f"{job_id}.gguf"

        # Fix tokenizer_config.json: extra_special_tokens must be a dict, not a list
        import json
        tok_cfg_path = Path(merged_path) / "tokenizer_config.json"
        if tok_cfg_path.exists():
            tok_cfg = json.loads(tok_cfg_path.read_text())
            if isinstance(tok_cfg.get("extra_special_tokens"), list):
                tok_cfg["extra_special_tokens"] = {}
                tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2))

        # Convert to GGUF using llama.cpp (must be installed separately)
        convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            return {"error": "llama.cpp not found at ~/llama.cpp — export unavailable"}

        # Use gguf_patch.py wrapper to fix Qwen2 tokenizer bug in transformers
        venv_python = str(Path(sys.executable))
        wrapper = Path(__file__).resolve().parent.parent / "ml" / "gguf_patch.py"
        subprocess.run(
            [venv_python, str(wrapper), str(convert_script), merged_path,
             "--outfile", str(gguf_path), "--outtype", "q8_0"],
            check=True,
        )

        # Build Modelfile
        model_name = f"llmplatform/{job_id[:8]}"
        modelfile_path = gguf_dir / f"{job_id}.Modelfile"
        modelfile_path.write_text(
            f"FROM {gguf_path}\n"
            f'PARAMETER temperature 0.7\n'
            f'PARAMETER repeat_penalty 1.1\n'
        )

        # Register with Ollama
        subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            check=True,
        )

        db["training_jobs"].update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {
                "gguf_path": str(gguf_path),
                "ollama_model_name": model_name,
                "updated_at": datetime.utcnow(),
            }},
        )
        return {"gguf_path": str(gguf_path), "ollama_model_name": model_name}

    except Exception:
        err = traceback.format_exc()
        return {"error": err}
