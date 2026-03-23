"""
Evaluation routes.

POST /jobs/{job_id}/evaluation/auto     — dispatch auto-eval Celery task
GET  /jobs/{job_id}/evaluation          — fetch Evaluation doc
POST /jobs/{job_id}/evaluation/human    — submit human scores
GET  /projects/{project_id}/evaluations — comparison table
"""
from __future__ import annotations

from datetime import datetime

from beanie import PydanticObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.evaluation import Evaluation
from app.models.job import TrainingJob
from app.models.project import Project

router = APIRouter(tags=["evaluations"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class HumanSample(BaseModel):
    prompt: str
    response: str
    ground_truth: str = ""
    accuracy: int
    relevance: int
    fluency: int
    completeness: int


class HumanEvalBody(BaseModel):
    results: list[HumanSample]


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _require_job(job_id: str) -> TrainingJob:
    job = await TrainingJob.get(PydanticObjectId(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/evaluation/auto", status_code=202)
async def run_auto_eval(job_id: str):
    job = await _require_job(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before running evaluation.")

    from workers.evaluation_tasks import run_auto_eval_task
    task = run_auto_eval_task.delay(job_id)
    return {"task_id": task.id}


@router.get("/jobs/{job_id}/evaluation")
async def get_evaluation(job_id: str):
    await _require_job(job_id)
    ev = await Evaluation.find_one(
        Evaluation.job_id.id == PydanticObjectId(job_id)  # type: ignore[attr-defined]
    )
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return ev.model_dump(mode="json")


@router.post("/jobs/{job_id}/evaluation/human")
async def submit_human_eval(job_id: str, body: HumanEvalBody):
    await _require_job(job_id)

    ev = await Evaluation.find_one(
        Evaluation.job_id.id == PydanticObjectId(job_id)  # type: ignore[attr-defined]
    )
    if not ev:
        raise HTTPException(status_code=404, detail="Run auto evaluation first.")

    results = [r.model_dump() for r in body.results]

    # Average across all four dimensions and all samples
    scores = [
        (r["accuracy"] + r["relevance"] + r["fluency"] + r["completeness"]) / 4
        for r in results
    ]
    human_avg = round(sum(scores) / len(scores), 2) if scores else None

    ev.sample_results = results
    ev.human_avg_score = human_avg
    await ev.save()

    return ev.model_dump(mode="json")


@router.get("/projects/{project_id}/evaluations")
async def list_evaluations(project_id: str):
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all completed jobs for the project
    jobs = await TrainingJob.find(
        TrainingJob.project_id.id == PydanticObjectId(project_id),  # type: ignore[attr-defined]
        TrainingJob.status == "completed",
    ).to_list()

    if not jobs:
        return {"evaluations": []}

    job_map = {str(j.id): j for j in jobs}
    job_ids = [PydanticObjectId(j_id) for j_id in job_map]

    # Fetch evaluations whose job_id is in the set
    all_evals = await Evaluation.find(
        {"job_id": {"$in": job_ids}}
    ).to_list()

    rows = []
    for ev in all_evals:
        job_id_str = str(ev.job_id.ref.id)  # type: ignore[union-attr]
        job = job_map.get(job_id_str)
        if not job:
            continue
        rows.append({
            "job_id": job_id_str,
            "job_name": job.name,
            "base_model": job.base_model,
            "epochs": job.epochs,
            "rouge_l": ev.rouge_l,
            "rouge_1": ev.rouge_1,
            "rouge_2": ev.rouge_2,
            "bleu": ev.bleu,
            "perplexity": ev.perplexity,
            "human_avg_score": ev.human_avg_score,
            "created_at": ev.created_at.isoformat() if ev.created_at else None,
        })

    return {"evaluations": rows}
