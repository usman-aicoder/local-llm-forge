"""
Celery evaluation task.

Runs auto-evaluation on a completed training job:
  - SFT jobs: loads PEFT model, generates on val.jsonl, computes ROUGE/BLEU
  - DPO/ORPO jobs: uses Ollama API with the formatted JSONL (prompt→chosen pairs)
  - Perplexity taken from the final checkpoint
  - Saves/upserts an Evaluation document in MongoDB
"""
from __future__ import annotations

import traceback
from datetime import datetime

from bson import ObjectId, DBRef
from pymongo import MongoClient

from app.config import settings
from workers.celery_app import celery_app


def _get_db():
    client = MongoClient(settings.mongo_url)
    return client[settings.mongo_db_name]


def _extract_id(ref) -> str:
    """Extract plain ObjectId string from either an ObjectId or a pymongo DBRef."""
    if isinstance(ref, DBRef):
        return str(ref.id)
    if hasattr(ref, "id"):
        return str(ref.id)
    return str(ref)


@celery_app.task(bind=True, name="workers.evaluation_tasks.run_auto_eval_task")
def run_auto_eval_task(self, job_id: str) -> dict:
    db = _get_db()

    try:
        # ── Load job ──────────────────────────────────────────────────────────
        job = db["training_jobs"].find_one({"_id": ObjectId(job_id)})
        if not job:
            return {"error": "Job not found"}
        if job["status"] != "completed":
            return {"error": "Job must be completed before evaluation"}

        adapter_path = job.get("adapter_path")
        model_path   = job.get("model_path")
        dataset_id   = _extract_id(job["dataset_id"])
        training_method = job.get("training_method", "sft")
        ollama_model    = job.get("ollama_model_name")

        if not adapter_path:
            return {"error": "Job has no adapter_path"}
        if not model_path:
            return {"error": "Job has no model_path"}

        # ── Perplexity from final checkpoint ──────────────────────────────────
        final_ckpt = db["checkpoints"].find_one(
            {"job_id": ObjectId(job_id)},
            sort=[("epoch", -1)],
        )
        perplexity = final_ckpt["perplexity"] if final_ckpt else None

        # ── Route: SFT uses PEFT model + val.jsonl
        #          DPO/ORPO uses Ollama + formatted JSONL ────────────────────
        val_path = settings.abs(settings.datasets_tokenized_dir) / dataset_id / "val.jsonl"

        if val_path.exists() and training_method == "sft":
            # SFT: generate with PEFT model loaded directly
            from ml.evaluate import run_evaluation
            result = run_evaluation(
                model_path=model_path,
                adapter_path=adapter_path,
                val_jsonl_path=str(val_path),
                max_samples=50,
                max_new_tokens=256,
            )

        else:
            # DPO/ORPO (or SFT without val.jsonl): use Ollama for generation
            formatted_path = settings.abs(settings.datasets_formatted_dir) / f"{dataset_id}.jsonl"

            if not formatted_path.exists():
                return {"error": f"No evaluation data found. Expected: {formatted_path}"}

            if not ollama_model:
                return {"error": "Job has no ollama_model_name — export to Ollama first."}

            from ml.evaluate import run_evaluation_via_ollama
            result = run_evaluation_via_ollama(
                formatted_jsonl_path=str(formatted_path),
                ollama_model_name=f"{ollama_model}:latest",
                ollama_url=settings.ollama_url,
                max_samples=20,
                max_new_tokens=256,
            )

        # ── Upsert Evaluation doc ─────────────────────────────────────────────
        eval_doc = {
            "job_id": ObjectId(job_id),
            "rouge_1": result["rouge_1"],
            "rouge_2": result["rouge_2"],
            "rouge_l": result["rouge_l"],
            "bleu": result["bleu"],
            "perplexity": perplexity,
            "human_avg_score": None,
            "sample_results": result["sample_results"],
            "created_at": datetime.utcnow(),
        }

        db["evaluations"].replace_one(
            {"job_id": ObjectId(job_id)},
            eval_doc,
            upsert=True,
        )

        return {
            "status": "completed",
            "rouge_l": result["rouge_l"],
            "bleu": result["bleu"],
        }

    except Exception:
        return {"error": traceback.format_exc()}
