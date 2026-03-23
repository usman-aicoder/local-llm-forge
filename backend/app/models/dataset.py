from datetime import datetime
from typing import Literal
from beanie import Document, Link
from pydantic import BaseModel, Field

from app.models.project import Project


class DatasetStats(BaseModel):
    total_rows: int = 0
    null_count: int = 0
    duplicate_count: int = 0
    removed_count: int = 0
    avg_instruction_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    p95_total_tokens: float = 0.0
    flagged_too_long: int = 0
    flagged_too_short: int = 0
    token_histogram: dict = Field(default_factory=dict)   # {buckets: [], counts: []}
    cleaning_report: dict = Field(default_factory=dict)   # {step_name: rows_removed}


class Dataset(Document):
    project_id: Link[Project]
    name: str
    source_type: Literal["upload", "huggingface", "synthetic", "pdf", "web"] = "upload"
    source_url: str | None = None          # populated for web-scraped datasets
    file_path: str
    row_count: int | None = None
    status: Literal["generating", "uploaded", "inspected", "cleaned", "formatted", "tokenized", "failed"] = "uploaded"
    generation_model: str | None = None    # Ollama model used for Q&A generation
    format_type: Literal["alpaca", "chat", "dpo"] | None = None
    base_model_name: str | None = None
    stats: DatasetStats | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "datasets"
