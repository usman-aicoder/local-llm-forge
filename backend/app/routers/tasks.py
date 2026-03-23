from beanie import PydanticObjectId
from fastapi import APIRouter, HTTPException

from app.models.task import TaskRecord

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    try:
        task = await TaskRecord.get(PydanticObjectId(task_id))
    except Exception:
        raise HTTPException(status_code=404, detail="Task not found")
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.model_dump(mode="json")
