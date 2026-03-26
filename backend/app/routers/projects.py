from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from beanie import PydanticObjectId

from app.models.project import Project
from app.models.job import TrainingJob
from app.models.evaluation import Evaluation

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


@router.get("")
async def list_projects():
    projects = await Project.find_all().to_list()
    return {"projects": [p.model_dump(mode="json") for p in projects]}


@router.post("", status_code=201)
async def create_project(body: ProjectCreate):
    project = Project(name=body.name, description=body.description)
    await project.insert()
    return project.model_dump(mode="json")


@router.get("/{project_id}")
async def get_project(project_id: PydanticObjectId):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump(mode="json")


@router.patch("/{project_id}")
async def update_project(project_id: PydanticObjectId, body: ProjectUpdate):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    update_data = body.model_dump(exclude_none=True)
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await project.set(update_data)
    return project.model_dump(mode="json")


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: PydanticObjectId):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await project.delete()


@router.get("/{project_id}/experiments/summary")
async def experiments_summary(project_id: str):
    """
    Return all completed jobs for a project with hyperparameters and
    evaluation metrics joined — formatted for charting/comparison.
    """
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    jobs = await TrainingJob.find(
        TrainingJob.project_id.id == PydanticObjectId(project_id),  # type: ignore[attr-defined]
        TrainingJob.status == "completed",
    ).to_list()

    experiments = []
    for job in jobs:
        eval_doc = await Evaluation.find_one(
            Evaluation.job_id.id == job.id  # type: ignore[attr-defined]
        )
        metrics = {
            "rouge_1":    eval_doc.rouge_1    if eval_doc else None,
            "rouge_2":    eval_doc.rouge_2    if eval_doc else None,
            "rouge_l":    eval_doc.rouge_l    if eval_doc else None,
            "bleu":       eval_doc.bleu       if eval_doc else None,
            "perplexity": eval_doc.perplexity if eval_doc else None,
        }
        experiments.append({
            "id":              str(job.id),
            "name":            job.name,
            "training_method": job.training_method,
            "base_model":      job.base_model,
            "hyperparams": {
                "learning_rate": job.learning_rate,
                "epochs":        job.epochs,
                "batch_size":    job.batch_size,
                "grad_accum":    job.grad_accum,
                "lora_r":        job.lora_r,
                "lora_alpha":    job.lora_alpha,
                "use_qlora":     job.use_qlora,
                "use_unsloth":   job.use_unsloth,
                "max_seq_len":   job.max_seq_len,
            },
            "metrics":      metrics,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "created_at":   job.created_at.isoformat()   if job.created_at   else None,
        })

    return {"experiments": experiments}
