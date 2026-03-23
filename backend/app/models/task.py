from datetime import datetime
from typing import Literal, Any
from beanie import Document
from pydantic import Field


class TaskRecord(Document):
    """Tracks background task status for dataset pipeline operations."""
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: dict | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "tasks"
