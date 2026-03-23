"""
Dataset service — background task runners for each pipeline stage.
Called by the datasets router via FastAPI BackgroundTasks.
Each function updates the TaskRecord and Dataset documents directly.
"""
from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

from beanie import PydanticObjectId

from app.config import settings
from app.models.dataset import Dataset, DatasetStats
from app.models.task import TaskRecord


async def _fail(task: TaskRecord, msg: str) -> None:
    task.status = "failed"
    task.error = msg
    task.updated_at = datetime.utcnow()
    await task.save()


async def run_eda(task_id: str, dataset_id: str) -> None:
    task = await TaskRecord.get(PydanticObjectId(task_id))
    dataset = await Dataset.get(PydanticObjectId(dataset_id))
    if not task or not dataset:
        return

    try:
        task.status = "running"
        task.updated_at = datetime.utcnow()
        await task.save()

        from ml.eda import run_eda as _run_eda
        stats_dict = _run_eda(dataset.file_path)

        dataset.stats = DatasetStats(**stats_dict)
        dataset.row_count = stats_dict["total_rows"]
        dataset.status = "inspected"
        dataset.updated_at = datetime.utcnow()
        await dataset.save()

        task.status = "completed"
        task.result = stats_dict
        task.updated_at = datetime.utcnow()
        await task.save()

    except Exception:
        await _fail(task, traceback.format_exc())


async def run_clean(task_id: str, dataset_id: str, config: dict) -> None:
    task = await TaskRecord.get(PydanticObjectId(task_id))
    dataset = await Dataset.get(PydanticObjectId(dataset_id))
    if not task or not dataset:
        return

    try:
        task.status = "running"
        task.updated_at = datetime.utcnow()
        await task.save()

        from ml.clean import run_clean as _run_clean

        cleaned_dir = settings.abs(settings.datasets_cleaned_dir)
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        out_path = cleaned_dir / f"{dataset_id}{Path(dataset.file_path).suffix}"

        report = _run_clean(
            file_path=dataset.file_path,
            output_path=str(out_path),
            **config,
        )

        # Update dataset file_path to cleaned file, patch stats
        dataset.file_path = str(out_path)
        dataset.status = "cleaned"
        dataset.row_count = report.get("rows_remaining", dataset.row_count)
        if dataset.stats:
            dataset.stats.removed_count = report.get("total_removed", 0)
            dataset.stats.cleaning_report = {k: v for k, v in report.items()
                                             if k not in ("total_removed", "rows_remaining")}
        dataset.updated_at = datetime.utcnow()
        await dataset.save()

        task.status = "completed"
        task.result = report
        task.updated_at = datetime.utcnow()
        await task.save()

    except Exception:
        await _fail(task, traceback.format_exc())


async def run_format(
    task_id: str,
    dataset_id: str,
    format_type: str,
    base_model: str,
) -> None:
    task = await TaskRecord.get(PydanticObjectId(task_id))
    dataset = await Dataset.get(PydanticObjectId(dataset_id))
    if not task or not dataset:
        return

    try:
        task.status = "running"
        task.updated_at = datetime.utcnow()
        await task.save()

        from ml.format_dataset import run_format as _run_format

        formatted_dir = settings.abs(settings.datasets_formatted_dir)
        formatted_dir.mkdir(parents=True, exist_ok=True)
        out_path = formatted_dir / f"{dataset_id}.jsonl"

        result = _run_format(
            file_path=dataset.file_path,
            output_path=str(out_path),
            format_type=format_type,
            base_model=base_model,
        )

        dataset.file_path = str(out_path)
        dataset.format_type = format_type
        dataset.base_model_name = base_model
        dataset.status = "formatted"
        dataset.updated_at = datetime.utcnow()
        await dataset.save()

        task.status = "completed"
        task.result = result
        task.updated_at = datetime.utcnow()
        await task.save()

    except Exception:
        await _fail(task, traceback.format_exc())


async def run_tokenize(
    task_id: str,
    dataset_id: str,
    max_seq_len: int,
    val_split: float,
) -> None:
    task = await TaskRecord.get(PydanticObjectId(task_id))
    dataset = await Dataset.get(PydanticObjectId(dataset_id))
    if not task or not dataset:
        return

    try:
        task.status = "running"
        task.updated_at = datetime.utcnow()
        await task.save()

        from ml.tokenize_dataset import run_tokenize as _run_tokenize
        from app.services.hf_model_service import OLLAMA_TO_HF

        hf_id = OLLAMA_TO_HF.get(dataset.base_model_name or "")
        tokenized_dir = settings.abs(settings.datasets_tokenized_dir) / dataset_id
        tokenized_dir.mkdir(parents=True, exist_ok=True)

        result = _run_tokenize(
            formatted_path=dataset.file_path,
            output_dir=str(tokenized_dir),
            model_path=None,
            base_model=dataset.base_model_name,
            max_seq_len=max_seq_len,
            val_split=val_split,
        )

        dataset.status = "tokenized"
        dataset.updated_at = datetime.utcnow()
        await dataset.save()

        task.status = "completed"
        task.result = result
        task.updated_at = datetime.utcnow()
        await task.save()

    except Exception:
        await _fail(task, traceback.format_exc())


# ── PDF → Q&A generation ──────────────────────────────────────────────────────

async def run_from_pdf(
    task_id: str,
    dataset_id: str,
    ollama_model: str,
    pairs_per_chunk: int,
) -> None:
    import asyncio

    task = await TaskRecord.get(PydanticObjectId(task_id))
    dataset = await Dataset.get(PydanticObjectId(dataset_id))
    if not task or not dataset:
        return

    try:
        task.status = "running"
        task.updated_at = datetime.utcnow()
        await task.save()

        from ml.pdf_extract import extract_pdf_text
        from ml.generate_qa import generate_qa_pairs, save_as_jsonl

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, extract_pdf_text, dataset.file_path)

        raw_dir = settings.abs(settings.datasets_raw_dir)
        out_path = raw_dir / f"{dataset_id}.jsonl"

        pairs = await loop.run_in_executor(
            None,
            lambda: generate_qa_pairs(
                text=text,
                ollama_model=ollama_model,
                ollama_url=settings.ollama_url,
                pairs_per_chunk=pairs_per_chunk,
            ),
        )

        if not pairs:
            raise ValueError("Ollama returned no Q&A pairs — check the model and PDF content.")

        row_count = await loop.run_in_executor(None, save_as_jsonl, pairs, str(out_path))

        dataset.file_path = str(out_path)
        dataset.row_count = row_count
        dataset.status = "uploaded"
        dataset.generation_model = ollama_model
        dataset.updated_at = datetime.utcnow()
        await dataset.save()

        task.status = "completed"
        task.result = {"row_count": row_count, "output_path": str(out_path)}
        task.updated_at = datetime.utcnow()
        await task.save()

    except Exception:
        err = traceback.format_exc()
        dataset.status = "failed"
        dataset.updated_at = datetime.utcnow()
        await dataset.save()
        await _fail(task, err)


# ── URL scrape → Q&A generation ───────────────────────────────────────────────

async def run_from_url(
    task_id: str,
    dataset_id: str,
    url: str,
    ollama_model: str,
    pairs_per_chunk: int,
) -> None:
    import asyncio

    task = await TaskRecord.get(PydanticObjectId(task_id))
    dataset = await Dataset.get(PydanticObjectId(dataset_id))
    if not task or not dataset:
        return

    try:
        task.status = "running"
        task.updated_at = datetime.utcnow()
        await task.save()

        from ml.web_scrape import scrape_url
        from ml.generate_qa import generate_qa_pairs, save_as_jsonl

        loop = asyncio.get_event_loop()
        title, text = await loop.run_in_executor(None, scrape_url, url)

        raw_dir = settings.abs(settings.datasets_raw_dir)
        out_path = raw_dir / f"{dataset_id}.jsonl"

        pairs = await loop.run_in_executor(
            None,
            lambda: generate_qa_pairs(
                text=text,
                ollama_model=ollama_model,
                ollama_url=settings.ollama_url,
                pairs_per_chunk=pairs_per_chunk,
            ),
        )

        if not pairs:
            raise ValueError("Ollama returned no Q&A pairs — check the model and URL content.")

        row_count = await loop.run_in_executor(None, save_as_jsonl, pairs, str(out_path))

        dataset.file_path = str(out_path)
        dataset.row_count = row_count
        dataset.status = "uploaded"
        dataset.generation_model = ollama_model
        dataset.updated_at = datetime.utcnow()
        await dataset.save()

        task.status = "completed"
        task.result = {"row_count": row_count, "output_path": str(out_path), "title": title}
        task.updated_at = datetime.utcnow()
        await task.save()

    except Exception:
        err = traceback.format_exc()
        dataset.status = "failed"
        dataset.updated_at = datetime.utcnow()
        await dataset.save()
        await _fail(task, err)
