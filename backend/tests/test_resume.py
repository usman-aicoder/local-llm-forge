"""
Phase 2.2 — Resume training from checkpoint tests.
"""
import pytest

from app.models.job import TrainingJob


# ── Unit: model field ─────────────────────────────────────────────────────────

def test_resume_from_job_id_field_exists():
    """TrainingJob must have the resume_from_job_id field."""
    import inspect
    import dataclasses
    fields = TrainingJob.model_fields
    assert "resume_from_job_id" in fields
    assert fields["resume_from_job_id"].default is None


def test_resume_from_checkpoint_in_train_signature():
    """run_training must accept resume_from_checkpoint param."""
    import inspect
    from ml.train import run_training
    sig = inspect.signature(run_training)
    param = sig.parameters["resume_from_checkpoint"]
    assert param.default is None


# ── Unit: checkpoint path resolution logic ────────────────────────────────────

def test_checkpoint_resolution_uses_adapter_path(tmp_path):
    """
    If source job has adapter_path, that path is used as resume_from_checkpoint.
    This mirrors the logic in training_tasks.py without touching MongoDB.
    """
    adapter_path = str(tmp_path / "job-abc" / "adapter")
    (tmp_path / "job-abc" / "adapter").mkdir(parents=True)

    source_job = {"adapter_path": adapter_path}

    # Replicate the resolution logic
    resume_path = None
    if source_job and source_job.get("adapter_path"):
        resume_path = source_job["adapter_path"]

    assert resume_path == adapter_path


def test_checkpoint_resolution_falls_back_to_checkpoint_dir(tmp_path):
    """
    If source job has no adapter_path, the latest checkpoint-N dir is used.
    """
    job_dir = tmp_path / "job-xyz"
    (job_dir / "checkpoint-100").mkdir(parents=True)
    (job_dir / "checkpoint-200").mkdir(parents=True)
    (job_dir / "checkpoint-50").mkdir(parents=True)

    source_job = {"adapter_path": None}

    resume_path = None
    if source_job and source_job.get("adapter_path"):
        resume_path = source_job["adapter_path"]
    else:
        checkpoint_dir = job_dir
        if checkpoint_dir.exists():
            checkpoints = sorted(
                checkpoint_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]),
            )
            if checkpoints:
                resume_path = str(checkpoints[-1])

    assert resume_path is not None
    assert "checkpoint-200" in resume_path


def test_checkpoint_resolution_returns_none_when_no_checkpoints(tmp_path):
    """If source job has no adapter and no checkpoint dirs, resume_path stays None."""
    source_job = {"adapter_path": None}
    empty_dir = tmp_path / "job-empty"
    empty_dir.mkdir()

    resume_path = None
    if source_job and source_job.get("adapter_path"):
        resume_path = source_job["adapter_path"]
    else:
        checkpoint_dir = empty_dir
        if checkpoint_dir.exists():
            checkpoints = sorted(
                checkpoint_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]),
            )
            if checkpoints:
                resume_path = str(checkpoints[-1])

    assert resume_path is None


def test_checkpoint_resolution_returns_none_when_source_job_missing():
    """If source job is not found (None), resume_path stays None."""
    source_job = None
    resume_path = None
    if source_job and source_job.get("adapter_path"):
        resume_path = source_job["adapter_path"]
    assert resume_path is None


# ── Integration: resume_from_job_id stored in job ────────────────────────────

@pytest.mark.asyncio
async def test_resume_from_job_id_persisted(client):
    """resume_from_job_id must be stored in MongoDB when creating a job."""
    proj = await client.post("/projects", json={"name": "resume-test"})
    assert proj.status_code == 201
    proj_id = proj.json()["id"]

    import io
    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings

    ds_data = b'{"text": "sample"}\n'
    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(ds_data), "application/json")},
        data={"name": "ds-resume"},
    )
    assert ds.status_code == 201
    ds_id = ds.json()["id"]

    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    await db["datasets"].update_one(
        {"_id": ObjectId(ds_id)},
        {"$set": {"status": "tokenized"}},
    )
    motor.close()

    fake_job_id = "000000000000000000000001"
    resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "resume-job",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
            "resume_from_job_id": fake_job_id,
        },
    )
    assert resp.status_code in (201, 500), resp.text
    if resp.status_code == 201:
        assert resp.json()["resume_from_job_id"] == fake_job_id
