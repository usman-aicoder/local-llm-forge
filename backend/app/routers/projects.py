from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from beanie import PydanticObjectId

from app.models.project import Project

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
