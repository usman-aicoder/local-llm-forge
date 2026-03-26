from datetime import datetime
from typing import Literal
from beanie import Document, Link
from pydantic import Field

from app.models.project import Project
from app.models.dataset import Dataset


class TrainingJob(Document):
    project_id: Link[Project]
    dataset_id: Link[Dataset]
    name: str

    # Model
    base_model: str
    model_path: str
    use_qlora: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training method
    training_method: Literal["sft", "dpo", "orpo"] = "sft"

    # Acceleration / resume
    use_unsloth: bool = False
    resume_from_job_id: str | None = None

    # Training
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    max_seq_len: int = 2048
    bf16: bool = True

    # Runtime
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    celery_task_id: str | None = None
    error_message: str | None = None

    # Artifacts
    adapter_path: str | None = None
    merged_path: str | None = None
    gguf_path: str | None = None
    ollama_model_name: str | None = None
    vllm_launch_cmd: str | None = None
    model_card_path: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "training_jobs"
