from datetime import datetime
from beanie import Document, Link
from pydantic import Field

from app.models.job import TrainingJob


class Evaluation(Document):
    job_id: Link[TrainingJob]
    rouge_l: float | None = None
    rouge_1: float | None = None
    rouge_2: float | None = None
    bleu: float | None = None
    perplexity: float | None = None
    human_avg_score: float | None = None
    # [{prompt, response, accuracy, relevance, fluency, completeness}]
    sample_results: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "evaluations"
