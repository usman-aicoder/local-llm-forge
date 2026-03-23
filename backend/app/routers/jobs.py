"""
Jobs router — create, list, monitor, cancel, merge, export.
SSE endpoint streams live training progress from Redis.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime

import redis.asyncio as aioredis
from beanie import PydanticObjectId
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.models.dataset import Dataset
from app.models.job import TrainingJob
from app.models.project import Project
from app.services.training_service import estimate_vram_gb, get_gpu_utilization

router = APIRouter(tags=["jobs"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class JobCreate(BaseModel):
    dataset_id: str
    name: str
    base_model: str = "mistral:7b"
    model_path: str
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    training_method: str = "sft"   # "sft" | "dpo" | "orpo"
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    max_seq_len: int = 2048
    bf16: bool = True


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _require_job(job_id: str) -> TrainingJob:
    job = await TrainingJob.get(PydanticObjectId(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/jobs")
async def list_jobs(project_id: str):
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    jobs = await TrainingJob.find(
        TrainingJob.project_id.id == PydanticObjectId(project_id)  # type: ignore[attr-defined]
    ).to_list()
    return {"jobs": [j.model_dump(mode="json") for j in jobs]}


@router.post("/projects/{project_id}/jobs", status_code=201)
async def create_job(project_id: str, body: JobCreate):
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = await Dataset.get(PydanticObjectId(body.dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    is_preference = body.training_method in ("dpo", "orpo")
    if is_preference and dataset.status not in ("formatted", "tokenized"):
        raise HTTPException(status_code=400, detail="DPO/ORPO requires a formatted dataset (prompt/chosen/rejected columns).")
    if not is_preference and dataset.status != "tokenized":
        raise HTTPException(status_code=400, detail="SFT requires a tokenized dataset.")

    job = TrainingJob(
        project_id=project,   # type: ignore[arg-type]
        dataset_id=dataset,   # type: ignore[arg-type]
        name=body.name,
        base_model=body.base_model,
        model_path=body.model_path,
        use_qlora=body.use_qlora,
        training_method=body.training_method,
        lora_r=body.lora_r,
        lora_alpha=body.lora_alpha,
        lora_dropout=body.lora_dropout,
        target_modules=body.target_modules,
        learning_rate=body.learning_rate,
        epochs=body.epochs,
        batch_size=body.batch_size,
        grad_accum=body.grad_accum,
        max_seq_len=body.max_seq_len,
        bf16=body.bf16,
        status="queued",
    )
    await job.insert()

    # Dispatch to Celery
    from workers.training_tasks import run_training_task
    task = run_training_task.delay(str(job.id))
    job.celery_task_id = task.id
    await job.save()

    return job.model_dump(mode="json")


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await _require_job(job_id)
    return job.model_dump(mode="json")


@router.get("/jobs/{job_id}/checkpoints")
async def get_checkpoints(job_id: str):
    from app.models.checkpoint import Checkpoint
    await _require_job(job_id)
    checkpoints = await Checkpoint.find(
        Checkpoint.job_id.id == PydanticObjectId(job_id)  # type: ignore[attr-defined]
    ).sort("+epoch").to_list()
    return {"checkpoints": [c.model_dump(mode="json") for c in checkpoints]}


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    import shutil
    job = await _require_job(job_id)
    # Cancel any running celery task first
    if job.celery_task_id and job.status in ("queued", "running"):
        from workers.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)
    # Remove checkpoint/adapter files
    for path_attr in ("adapter_path", "merged_path"):
        p = getattr(job, path_attr, None)
        if p:
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
    # Remove GGUF export file
    if job.gguf_path:
        try:
            Path(job.gguf_path).unlink(missing_ok=True)
        except Exception:
            pass
    # Remove associated checkpoints from DB
    from app.models.checkpoint import Checkpoint
    await Checkpoint.find(Checkpoint.job_id.id == PydanticObjectId(job_id)).delete()  # type: ignore[attr-defined]
    await job.delete()


@router.delete("/jobs/{job_id}/cancel", status_code=204)
async def cancel_job(job_id: str):
    job = await _require_job(job_id)
    if job.celery_task_id:
        from workers.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)
    job.status = "cancelled"
    job.updated_at = datetime.utcnow() if hasattr(job, "updated_at") else None
    await job.save()


@router.post("/jobs/{job_id}/merge")
async def merge_job(job_id: str):
    job = await _require_job(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before merging.")

    from workers.export_tasks import run_merge_task
    task = run_merge_task.delay(job_id)
    return {"task_id": task.id}


@router.post("/jobs/{job_id}/export")
async def export_job(job_id: str):
    job = await _require_job(job_id)
    if not job.merged_path:
        raise HTTPException(status_code=400, detail="Merge the model before exporting to Ollama.")

    from workers.export_tasks import run_export_task
    task = run_export_task.delay(job_id)
    return {"task_id": task.id}


@router.get("/jobs/{job_id}/gpu")
async def gpu_status(job_id: str):
    return get_gpu_utilization()


@router.get("/models/vram-estimate")
async def vram_estimate(base_model: str = "mistral:7b", lora_r: int = 16, use_qlora: bool = True):
    return estimate_vram_gb(base_model, lora_r, use_qlora)


# ── SSE stream ────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """
    Server-Sent Events endpoint.
    Streams training progress and logs from Redis until job completes.
    Events:
      progress  — {epoch, train_loss, eval_loss, perplexity}
      log       — {line: str}
      done      — {status: str}
    """
    await _require_job(job_id)

    async def event_gen():
        r = aioredis.from_url(settings.redis_url)
        log_cursor = 0

        try:
            while True:
                # Progress
                raw = await r.get(f"job:{job_id}:progress")
                if raw:
                    yield f"event: progress\ndata: {raw.decode()}\n\n"

                # New log lines
                lines = await r.lrange(f"job:{job_id}:logs", log_cursor, -1)
                for line in lines:
                    payload = json.dumps({"line": line.decode()})
                    yield f"event: log\ndata: {payload}\n\n"
                log_cursor += len(lines)

                # Check if done
                done = await r.get(f"job:{job_id}:done")
                if done:
                    job = await TrainingJob.get(PydanticObjectId(job_id))
                    status = job.status if job else "unknown"
                    yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                    break

                await asyncio.sleep(2)

        finally:
            await r.aclose()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
