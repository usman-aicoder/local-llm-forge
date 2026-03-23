from datetime import datetime
from beanie import Document, Link
from pydantic import Field

from app.models.job import TrainingJob


class Checkpoint(Document):
    job_id: Link[TrainingJob]
    epoch: int
    step: int
    train_loss: float
    eval_loss: float
    perplexity: float   # exp(eval_loss)
    file_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "checkpoints"
