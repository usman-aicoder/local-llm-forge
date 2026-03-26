"""
Celery training task.

Runs SFTTrainer on the GPU worker process.
Progress (epoch, train_loss, eval_loss) pushed to Redis.
Log lines pushed to Redis list, capped at 1000 entries.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Ensure the backend root is on sys.path so 'ml' package is importable
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ml.train import run_training  # noqa: E402 — must come after sys.path fix
from pymongo import MongoClient
from bson import ObjectId
import redis

from app.config import settings
from workers.celery_app import celery_app

LOG_MAX_LEN = 1000


def _get_db():
    client = MongoClient(settings.mongo_url)
    return client[settings.mongo_db_name]


def _get_redis():
    return redis.Redis.from_url(settings.redis_url)


def _set_status(db, job_id: str, status: str, extra: dict | None = None):
    update: dict = {"status": status, "updated_at": datetime.utcnow()}
    if extra:
        update.update(extra)
    db["training_jobs"].update_one({"_id": ObjectId(job_id)}, {"$set": update})


def _push_progress(r, job_id: str, epoch: int, train_loss: float, eval_loss: float):
    perplexity = round(2.718281828 ** eval_loss, 4)
    payload = json.dumps({
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "eval_loss": round(eval_loss, 6),
        "perplexity": perplexity,
    })
    r.set(f"job:{job_id}:progress", payload)

    # Also write checkpoint to MongoDB
    db = _get_db()
    db["checkpoints"].insert_one({
        "job_id": ObjectId(job_id),
        "epoch": epoch,
        "step": 0,
        "train_loss": round(train_loss, 6),
        "eval_loss": round(eval_loss, 6),
        "perplexity": perplexity,
        "file_path": "",
        "created_at": datetime.utcnow(),
    })


def _push_log(r, job_id: str, line: str):
    key = f"job:{job_id}:logs"
    r.rpush(key, line)
    # Keep only the last LOG_MAX_LEN entries
    r.ltrim(key, -LOG_MAX_LEN, -1)


@celery_app.task(bind=True, name="workers.training_tasks.run_training_task")
def run_training_task(self, job_id: str) -> dict:
    db = _get_db()
    r = _get_redis()

    try:
        # Load job config
        job = db["training_jobs"].find_one({"_id": ObjectId(job_id)})
        if not job:
            return {"error": "Job not found"}

        _set_status(db, job_id, "running", {"started_at": datetime.utcnow()})
        _push_log(r, job_id, f"Job {job_id} started")

        # Resolve dataset_id (may be a Beanie DBRef)
        raw_ds_id = job["dataset_id"]
        if hasattr(raw_ds_id, "id"):
            raw_ds_id = raw_ds_id.id

        training_method = job.get("training_method", "sft")

        # SFT uses tokenized data; DPO/ORPO use formatted JSONL directly
        if training_method in ("dpo", "orpo"):
            formatted_path = settings.abs(settings.datasets_formatted_dir) / f"{raw_ds_id}.jsonl"
            if not formatted_path.exists():
                raise FileNotFoundError(f"Formatted dataset not found: {formatted_path}")
            train_path = formatted_path
            val_path   = formatted_path   # DPOTrainer handles splits internally
        else:
            tokenized_dir = settings.abs(settings.datasets_tokenized_dir) / str(raw_ds_id)
            train_path = tokenized_dir / "train.jsonl"
            val_path   = tokenized_dir / "val.jsonl"
            if not train_path.exists():
                raise FileNotFoundError(f"Tokenized training data not found: {train_path}")

        adapter_dir = settings.abs(settings.checkpoints_dir) / job_id
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve resume checkpoint path ────────────────────────────────────
        resume_path: str | None = None
        resume_from_job_id = job.get("resume_from_job_id")
        if resume_from_job_id:
            source_job = db["training_jobs"].find_one({"_id": ObjectId(resume_from_job_id)})
            if source_job and source_job.get("adapter_path"):
                resume_path = source_job["adapter_path"]
            else:
                # Fall back to latest checkpoint directory
                checkpoint_dir = settings.abs(settings.checkpoints_dir) / resume_from_job_id
                if checkpoint_dir.exists():
                    checkpoints = sorted(
                        checkpoint_dir.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[1]),
                    )
                    if checkpoints:
                        resume_path = str(checkpoints[-1])

        def on_epoch_end(epoch: int, train_loss: float, eval_loss: float):
            _push_progress(r, job_id, epoch, train_loss, eval_loss)
            _push_log(r, job_id, f"Epoch {epoch} — train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}")

        def on_log(msg: str):
            _push_log(r, job_id, msg)

        common = dict(
            job_id=job_id,
            model_path=job["model_path"],
            train_data_path=str(train_path),
            val_data_path=str(val_path),
            adapter_output_dir=str(adapter_dir),
            use_qlora=job.get("use_qlora", True),
            lora_r=job.get("lora_r", 16),
            lora_alpha=job.get("lora_alpha", 32),
            lora_dropout=job.get("lora_dropout", 0.05),
            target_modules=job.get("target_modules"),
            learning_rate=job.get("learning_rate", 2e-4),
            epochs=job.get("epochs", 3),
            batch_size=job.get("batch_size", 2),
            grad_accum=job.get("grad_accum", 8),
            max_seq_len=job.get("max_seq_len", 2048),
            bf16=job.get("bf16", True),
            use_unsloth=job.get("use_unsloth", False),
            resume_from_checkpoint=resume_path,
            on_epoch_end=on_epoch_end,
            on_log=on_log,
        )

        if training_method == "dpo":
            from ml.train_dpo import run_dpo_training
            result = run_dpo_training(**common)
        elif training_method == "orpo":
            from ml.train_orpo import run_orpo_training
            result = run_orpo_training(**common)
        else:
            result = run_training(**common)

        _set_status(db, job_id, "completed", {
            "adapter_path": result["adapter_path"],
            "completed_at": datetime.utcnow(),
        })
        _push_log(r, job_id, "Training completed successfully")
        r.set(f"job:{job_id}:done", "1")
        return result

    except Exception:
        err = traceback.format_exc()
        _set_status(db, job_id, "failed", {"error_message": err})
        _push_log(r, job_id, f"ERROR: {err}")
        r.set(f"job:{job_id}:done", "1")
        return {"error": err}
