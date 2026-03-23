"""
Inference routes.

GET  /inference/models   — grouped model list (base Ollama + fine-tuned exports)
POST /inference/generate — SSE: stream tokens from Ollama
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.ollama_service import ollama_service

router = APIRouter(prefix="/inference", tags=["inference"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512
    repeat_penalty: float = 1.1


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/models")
async def inference_models():
    """
    Returns two groups:
      base_models  — all models currently installed in Ollama
      fine_tuned   — completed jobs that have been exported (have ollama_model_name)
    """
    from app.models.job import TrainingJob

    base_models = []
    try:
        base_models = await ollama_service.list_models()
    except Exception:
        pass

    exported_jobs = await TrainingJob.find(
        TrainingJob.ollama_model_name != None  # noqa: E711
    ).to_list()

    fine_tuned = [
        {
            "job_id": str(j.id),
            "job_name": j.name,
            "ollama_model_name": j.ollama_model_name,
            "base_model": j.base_model,
        }
        for j in exported_jobs
        if j.ollama_model_name
    ]

    return {"base_models": base_models, "fine_tuned": fine_tuned}


@router.post("/generate")
async def generate(body: GenerateRequest):
    """
    SSE endpoint. Streams tokens from Ollama one by one.

    Events:
      data: {"token": "..."}
      data: {"done": true}

    Client reads via fetch() + ReadableStream (EventSource doesn't support POST).
    """
    async def event_gen():
        try:
            async for token in ollama_service.generate_stream(
                model=body.model,
                prompt=body.prompt,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                repeat_penalty=body.repeat_penalty,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
