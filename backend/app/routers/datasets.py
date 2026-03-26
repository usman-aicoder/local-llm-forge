from pathlib import Path
from datetime import datetime
import json
import shutil

import aiofiles
from beanie import PydanticObjectId
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.config import settings
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.task import TaskRecord

router = APIRouter(tags=["datasets"])

ALLOWED_EXTENSIONS      = {".csv", ".json", ".jsonl"}
ALLOWED_PDF_EXTENSIONS  = {".pdf"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class CleanConfig(BaseModel):
    strip_html: bool = True
    normalize_whitespace: bool = True
    remove_urls: bool = True
    deduplicate: bool = True
    filter_short: bool = True


class FormatConfig(BaseModel):
    format_type: str = "alpaca"   # "alpaca" | "chat"
    base_model: str = "mistral:7b"


class TokenizeConfig(BaseModel):
    max_seq_len: int = 2048
    val_split: float = 0.1


class FromUrlConfig(BaseModel):
    url: str
    name: str
    ollama_model: str = "gemma2:2b"
    pairs_per_chunk: int = 3


class PresetImportConfig(BaseModel):
    preset_id: str   # e.g. "orca_dpo_pairs"
    variant_id: str  # e.g. "sft" | "dpo" | "eval"
    name: str | None = None  # override dataset name


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _dispatch(coro) -> dict:
    """Create a TaskRecord, dispatch background coro, return task_id."""
    task = TaskRecord()
    await task.insert()
    return str(task.id)


async def _require_dataset(dataset_id: str) -> Dataset:
    ds = await Dataset.get(PydanticObjectId(dataset_id))
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/projects/{project_id}/datasets/upload", status_code=201)
async def upload_dataset(project_id: str, file: UploadFile = File(...)):
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Use CSV, JSON, or JSONL.")

    raw_dir = settings.abs(settings.datasets_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create Dataset doc first to get its id for the filename
    dataset = Dataset(
        project_id=project,  # type: ignore[arg-type]
        name=Path(file.filename or "dataset").stem,
        file_path="",  # filled below
        source_type="upload",
    )
    await dataset.insert()

    out_path = raw_dir / f"{dataset.id}{suffix}"
    async with aiofiles.open(out_path, "wb") as f_out:
        content = await file.read()
        await f_out.write(content)

    dataset.file_path = str(out_path)
    await dataset.save()

    return dataset.model_dump(mode="json")


@router.get("/projects/{project_id}/datasets")
async def list_datasets(project_id: str):
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    datasets = await Dataset.find(Dataset.project_id.id == PydanticObjectId(project_id)).to_list()  # type: ignore[attr-defined]
    return {"datasets": [d.model_dump(mode="json") for d in datasets]}


# NOTE: /datasets/presets must be defined before /datasets/{dataset_id}
# so FastAPI doesn't interpret "presets" as a dataset_id.
@router.get("/datasets/presets")
async def list_presets():
    """Return all preset datasets that have been prepared locally."""
    demo_root = settings.abs("./storage/demo")
    presets = []
    if demo_root.exists():
        for manifest_path in sorted(demo_root.rglob("manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
                manifest["_dir"] = str(manifest_path.parent)
                presets.append(manifest)
            except Exception:
                continue
    return {"presets": presets}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    ds = await _require_dataset(dataset_id)
    return ds.model_dump(mode="json")


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str):
    ds = await _require_dataset(dataset_id)
    # Remove the raw file if it exists
    if ds.file_path:
        try:
            Path(ds.file_path).unlink(missing_ok=True)
        except Exception:
            pass
    await ds.delete()


@router.post("/datasets/{dataset_id}/inspect")
async def inspect_dataset(dataset_id: str, bg: BackgroundTasks):
    ds = await _require_dataset(dataset_id)

    task = TaskRecord()
    await task.insert()
    task_id = str(task.id)

    from app.services.dataset_service import run_eda
    bg.add_task(run_eda, task_id, dataset_id)

    return {"task_id": task_id}


@router.post("/datasets/{dataset_id}/clean")
async def clean_dataset(dataset_id: str, config: CleanConfig, bg: BackgroundTasks):
    ds = await _require_dataset(dataset_id)
    if ds.status not in ("inspected", "cleaned"):
        raise HTTPException(status_code=400, detail="Run EDA inspection before cleaning.")

    task = TaskRecord()
    await task.insert()
    task_id = str(task.id)

    from app.services.dataset_service import run_clean
    bg.add_task(run_clean, task_id, dataset_id, config.model_dump())

    return {"task_id": task_id}


@router.get("/datasets/{dataset_id}/format/preview")
async def format_preview(dataset_id: str, format_type: str = "alpaca", base_model: str = "mistral:7b"):
    ds = await _require_dataset(dataset_id)

    from ml.format_dataset import get_preview
    try:
        samples = get_preview(ds.file_path, format_type=format_type, base_model=base_model, count=10)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"samples": samples, "format_type": format_type, "base_model": base_model}


@router.post("/datasets/{dataset_id}/format")
async def format_dataset(dataset_id: str, config: FormatConfig, bg: BackgroundTasks):
    ds = await _require_dataset(dataset_id)
    if ds.status not in ("cleaned", "inspected", "formatted"):
        raise HTTPException(status_code=400, detail="Dataset must be at least inspected before formatting.")

    task = TaskRecord()
    await task.insert()
    task_id = str(task.id)

    from app.services.dataset_service import run_format
    bg.add_task(run_format, task_id, dataset_id, config.format_type, config.base_model)

    return {"task_id": task_id}


@router.post("/datasets/{dataset_id}/augment")
async def augment_dataset(
    dataset_id: str,
    bg: BackgroundTasks,
    ollama_model: str = "llama3.2:latest",
    paraphrases_per_row: int = 2,
    max_rows_to_augment: int = 100,
):
    """
    Augment a formatted dataset using Ollama to generate paraphrased variants.
    Requires dataset status = formatted (Alpaca JSONL with instruction/output columns).
    Creates a NEW dataset document; the original is preserved.
    """
    ds = await _require_dataset(dataset_id)
    if ds.status not in ("formatted", "tokenized"):
        raise HTTPException(
            status_code=400,
            detail="Dataset must be formatted before augmenting.",
        )
    if not ds.file_path or not Path(ds.file_path).exists():
        raise HTTPException(status_code=400, detail="Dataset file not found on disk.")

    task = TaskRecord()
    await task.insert()
    task_id = str(task.id)

    from app.services.augment_service import run_augment
    bg.add_task(
        run_augment,
        task_id=task_id,
        dataset_id=dataset_id,
        ollama_model=ollama_model,
        paraphrases_per_row=paraphrases_per_row,
        max_rows_to_augment=max_rows_to_augment,
    )
    return {"task_id": task_id}


@router.post("/datasets/{dataset_id}/tokenize")
async def tokenize_dataset(dataset_id: str, config: TokenizeConfig, bg: BackgroundTasks):
    ds = await _require_dataset(dataset_id)
    if ds.status != "formatted":
        raise HTTPException(status_code=400, detail="Dataset must be formatted before tokenizing.")

    task = TaskRecord()
    await task.insert()
    task_id = str(task.id)

    from app.services.dataset_service import run_tokenize
    bg.add_task(run_tokenize, task_id, dataset_id, config.max_seq_len, config.val_split)

    return {"task_id": task_id}


@router.post("/projects/{project_id}/datasets/from-pdf", status_code=201)
async def create_from_pdf(
    project_id: str,
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(...),
    ollama_model: str = Form("gemma2:2b"),
    pairs_per_chunk: int = Form(3),
):
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_PDF_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    raw_dir = settings.abs(settings.datasets_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(
        project_id=project,  # type: ignore[arg-type]
        name=name,
        file_path="",
        source_type="pdf",
        status="generating",
        generation_model=ollama_model,
    )
    await dataset.insert()

    pdf_path = raw_dir / f"{dataset.id}.pdf"
    async with aiofiles.open(pdf_path, "wb") as f_out:
        await f_out.write(await file.read())

    dataset.file_path = str(pdf_path)
    await dataset.save()

    task = TaskRecord()
    await task.insert()

    from app.services.dataset_service import run_from_pdf
    bg.add_task(run_from_pdf, str(task.id), str(dataset.id), ollama_model, pairs_per_chunk)

    return {"dataset_id": str(dataset.id), "task_id": str(task.id)}


@router.post("/projects/{project_id}/datasets/from-url", status_code=201)
async def create_from_url(project_id: str, body: FromUrlConfig, bg: BackgroundTasks):
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = Dataset(
        project_id=project,  # type: ignore[arg-type]
        name=body.name,
        file_path="",
        source_type="web",
        source_url=body.url,
        status="generating",
        generation_model=body.ollama_model,
    )
    await dataset.insert()

    task = TaskRecord()
    await task.insert()

    from app.services.dataset_service import run_from_url
    bg.add_task(
        run_from_url,
        str(task.id),
        str(dataset.id),
        body.url,
        body.ollama_model,
        body.pairs_per_chunk,
    )

    return {"dataset_id": str(dataset.id), "task_id": str(task.id)}


@router.post("/projects/{project_id}/datasets/from-preset", status_code=201)
async def import_preset(project_id: str, body: PresetImportConfig):
    """Copy a preset variant file into the raw datasets dir and register it."""
    project = await Project.get(PydanticObjectId(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Find the manifest
    demo_root = settings.abs("./storage/demo")
    preset_dir = demo_root / body.preset_id
    manifest_path = preset_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{body.preset_id}' not found. Run scripts/prepare_demo_dataset.py first.")

    manifest = json.loads(manifest_path.read_text())
    variant = next((v for v in manifest["variants"] if v["id"] == body.variant_id), None)
    if not variant:
        raise HTTPException(status_code=404, detail=f"Variant '{body.variant_id}' not found in preset '{body.preset_id}'.")

    src_file = preset_dir / variant["file"]
    if not src_file.exists():
        raise HTTPException(status_code=404, detail=f"Preset file not found: {variant['file']}. Re-run prepare_demo_dataset.py.")

    raw_dir = settings.abs(settings.datasets_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = body.name or f"{manifest['name']} — {variant['label']}"

    dataset = Dataset(
        project_id=project,  # type: ignore[arg-type]
        name=dataset_name,
        file_path="",
        source_type="upload",
        row_count=variant.get("rows"),
        status="uploaded",
    )
    await dataset.insert()

    dest_file = raw_dir / f"{dataset.id}.jsonl"
    shutil.copy2(src_file, dest_file)

    dataset.file_path = str(dest_file)
    await dataset.save()

    return dataset.model_dump(mode="json")
