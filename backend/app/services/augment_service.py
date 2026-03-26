"""
Augment service — background task runner for dataset augmentation.

Creates a new Dataset document for the augmented output so the
original is always preserved.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from beanie import PydanticObjectId

from app.config import settings
from app.models.dataset import Dataset
from app.models.task import TaskRecord


async def run_augment(
    task_id: str,
    dataset_id: str,
    ollama_model: str,
    paraphrases_per_row: int,
    max_rows_to_augment: int,
) -> None:
    """
    Background task: augment dataset and save as a new Dataset document.
    """
    from motor.motor_asyncio import AsyncIOMotorClient
    from beanie import init_beanie
    from app.models import ALL_MODELS

    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    await init_beanie(database=db, document_models=ALL_MODELS)

    task = await TaskRecord.get(PydanticObjectId(task_id))

    try:
        ds = await Dataset.get(PydanticObjectId(dataset_id))
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")

        logs: list[str] = []

        def on_log(msg: str) -> None:
            logs.append(msg)

        # Output path for the augmented JSONL
        formatted_dir = settings.abs(settings.datasets_formatted_dir)
        formatted_dir.mkdir(parents=True, exist_ok=True)

        # We'll create the new Dataset doc first to get its ID for the filename
        new_ds = Dataset(
            project_id=ds.project_id,  # type: ignore[arg-type]
            name=f"{ds.name} (augmented ×{paraphrases_per_row})",
            source_type="augmented",
            file_path="",  # filled in after augmentation
            status="uploaded",
            format_type=ds.format_type,
            base_model_name=ds.base_model_name,
        )
        await new_ds.insert()

        output_path = formatted_dir / f"{new_ds.id}.jsonl"

        from ml.augment import augment_dataset
        result = augment_dataset(
            input_path=ds.file_path,
            output_path=str(output_path),
            ollama_url=settings.ollama_url,
            model_name=ollama_model,
            paraphrases_per_row=paraphrases_per_row,
            max_rows_to_augment=max_rows_to_augment,
            on_log=on_log,
        )

        new_ds.file_path = str(output_path)
        new_ds.row_count = result["augmented_rows"]
        new_ds.status = "formatted"
        await new_ds.save()

        if task:
            task.status = "completed"
            task.result = {
                "new_dataset_id": str(new_ds.id),
                "original_rows": result["original_rows"],
                "augmented_rows": result["augmented_rows"],
                "new_rows_added": result["new_rows_added"],
                "logs": logs[-20:],  # last 20 log lines
            }
            await task.save()

    except Exception as exc:
        import traceback
        if task:
            task.status = "failed"
            task.result = {"error": traceback.format_exc()}
            await task.save()

    finally:
        motor.close()
