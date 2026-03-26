"""
Jobs router — create, list, monitor, cancel, merge, export.
SSE endpoint streams live training progress from Redis.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import redis.asyncio as aioredis
import yaml
from beanie import PydanticObjectId
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import Response, StreamingResponse
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
    use_unsloth: bool = False
    resume_from_job_id: str | None = None
    webhook_url: str | None = None
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
    is_fft = body.training_method == "fft"
    if is_preference and dataset.status not in ("formatted", "tokenized"):
        raise HTTPException(status_code=400, detail="DPO/ORPO requires a formatted dataset (prompt/chosen/rejected columns).")
    if not is_preference and not is_fft and dataset.status != "tokenized":
        raise HTTPException(status_code=400, detail="SFT requires a tokenized dataset.")
    if is_fft and dataset.status != "tokenized":
        raise HTTPException(status_code=400, detail="FFT requires a tokenized dataset.")

    job = TrainingJob(
        project_id=project,   # type: ignore[arg-type]
        dataset_id=dataset,   # type: ignore[arg-type]
        name=body.name,
        base_model=body.base_model,
        model_path=body.model_path,
        use_qlora=body.use_qlora,
        training_method=body.training_method,
        use_unsloth=body.use_unsloth,
        resume_from_job_id=body.resume_from_job_id,
        webhook_url=body.webhook_url,
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


# ── System capabilities ───────────────────────────────────────────────────────

@router.get("/system/capabilities", tags=["system"])
async def system_capabilities():
    """Report which optional acceleration libraries are available."""
    return {
        "unsloth": importlib.util.find_spec("unsloth") is not None,
        "bitsandbytes": importlib.util.find_spec("bitsandbytes") is not None,
    }


# ── Config export / import ────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/config")
async def export_job_config(job_id: str):
    """Export all training hyperparameters for a job as a YAML file."""
    job = await _require_job(job_id)
    config = {
        "base_model": job.base_model,
        "model_path": job.model_path,
        "training_method": job.training_method,
        "lora": {
            "use_qlora": job.use_qlora,
            "r": job.lora_r,
            "alpha": job.lora_alpha,
            "dropout": job.lora_dropout,
            "target_modules": job.target_modules,
        },
        "training": {
            "learning_rate": job.learning_rate,
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "grad_accum": job.grad_accum,
            "max_seq_len": job.max_seq_len,
            "bf16": job.bf16,
        },
        "use_unsloth": job.use_unsloth,
    }
    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    return Response(
        content=yaml_content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f"attachment; filename=job-{job_id[:8]}-config.yaml"},
    )


@router.post("/jobs/from-config", status_code=200)
async def create_job_from_config(
    project_id: str,
    dataset_id: str,
    name: str,
    config_file: UploadFile = File(...),
):
    """Create a new job pre-filled from an uploaded YAML config file."""
    raw = await config_file.read()
    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    lora = config.get("lora", {})
    training = config.get("training", {})

    body = JobCreate(
        dataset_id=dataset_id,
        name=name,
        base_model=config.get("base_model", "mistral:7b"),
        model_path=config.get("model_path", ""),
        training_method=config.get("training_method", "sft"),
        use_qlora=lora.get("use_qlora", True),
        lora_r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        learning_rate=training.get("learning_rate", 2e-4),
        epochs=training.get("epochs", 3),
        batch_size=training.get("batch_size", 2),
        grad_accum=training.get("grad_accum", 8),
        max_seq_len=training.get("max_seq_len", 2048),
        bf16=training.get("bf16", True),
        use_unsloth=config.get("use_unsloth", False),
    )
    return await create_job(project_id, body)


# ── vLLM export ───────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/export-vllm")
async def export_vllm(job_id: str):
    """
    Generate a vLLM launch command for a merged model.
    Requires the merge step to have completed first.
    Stores the launch command on the job document.
    """
    job = await _require_job(job_id)
    if not job.merged_path:
        raise HTTPException(
            status_code=400,
            detail="Merge the model first (POST /jobs/{id}/merge).",
        )

    from ml.export_vllm import prepare_vllm_export
    try:
        result = prepare_vllm_export(job.merged_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    job.vllm_launch_cmd = result["launch_command"]
    await job.save()
    return result


# ── Model card ────────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/model-card")
async def generate_model_card(job_id: str):
    """
    Auto-generate MODEL_CARD.md and save it to the adapter directory.
    Optionally includes evaluation results if a completed evaluation exists.
    """
    job = await _require_job(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed first.")
    if not job.adapter_path:
        raise HTTPException(status_code=400, detail="No adapter path found on this job.")

    from app.models.dataset import Dataset
    from app.models.evaluation import Evaluation
    from ml.model_card import save_model_card

    # Fetch dataset metadata
    dataset = None
    try:
        raw_ds_id = job.dataset_id
        ds_id = raw_ds_id.ref.id if hasattr(raw_ds_id, "ref") else raw_ds_id
        dataset_doc = await Dataset.get(ds_id)
        if dataset_doc:
            dataset = dataset_doc.model_dump(mode="json")
    except Exception:
        pass

    # Fetch latest evaluation if available
    evaluation = None
    try:
        evals = await Evaluation.find(
            Evaluation.job_id.id == job.id  # type: ignore[attr-defined]
        ).sort("-created_at").limit(1).to_list()
        if evals:
            evaluation = evals[0].model_dump(mode="json")
    except Exception:
        pass

    card_path = save_model_card(
        job=job.model_dump(mode="json"),
        adapter_path=job.adapter_path,
        dataset=dataset,
        evaluation=evaluation,
    )

    job.model_card_path = card_path
    await job.save()

    # Return the markdown content so the UI can render it inline
    card_content = Path(card_path).read_text(encoding="utf-8")
    return {"model_card_path": card_path, "content": card_content}


@router.get("/jobs/{job_id}/model-card")
async def get_model_card(job_id: str):
    """Return the current MODEL_CARD.md content for a job."""
    job = await _require_job(job_id)
    if not job.model_card_path:
        raise HTTPException(
            status_code=404,
            detail="No model card generated yet. POST /jobs/{id}/model-card to generate one.",
        )
    card_path = Path(job.model_card_path)
    if not card_path.exists():
        raise HTTPException(status_code=404, detail="Model card file not found on disk.")
    return {"model_card_path": str(card_path), "content": card_path.read_text(encoding="utf-8")}


# ── HuggingFace Hub push ──────────────────────────────────────────────────────

class HubPushBody(BaseModel):
    repo_id: str
    private: bool = True
    commit_message: str = "Upload fine-tuned model"


@router.post("/jobs/{job_id}/push-to-hub")
async def push_to_hub(job_id: str, body: HubPushBody):
    """
    Push the model adapter (or full FFT model) to HuggingFace Hub.
    Requires HF_TOKEN to be set in .env / environment.
    Uses merged_path if available, otherwise adapter_path.
    """
    job = await _require_job(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before pushing to hub.")

    push_dir = job.merged_path or job.adapter_path
    if not push_dir:
        raise HTTPException(
            status_code=400,
            detail="No model artifacts found. Run merge first (POST /jobs/{id}/merge).",
        )
    if not Path(push_dir).exists():
        raise HTTPException(status_code=400, detail=f"Model directory not found on disk: {push_dir}")

    token = settings.hf_token
    if not token:
        raise HTTPException(status_code=400, detail="HF_TOKEN is not configured. Add it to your .env file.")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=body.repo_id, private=body.private, exist_ok=True)
        api.upload_folder(
            folder_path=push_dir,
            repo_id=body.repo_id,
            commit_message=body.commit_message,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HuggingFace Hub push failed: {exc}")

    job.hf_repo_id = body.repo_id
    await job.save()

    return {
        "repo_id": body.repo_id,
        "url": f"https://huggingface.co/{body.repo_id}",
    }


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
