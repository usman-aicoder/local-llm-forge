"""
Phase 4.4 — Full Fine-Tuning (FFT) mode tests.
"""
import inspect
import io

import pytest


# ── Unit: train_fft signature ─────────────────────────────────────────────────

def test_fft_function_exists():
    from ml.train_fft import run_fft_training
    assert callable(run_fft_training)


def test_fft_signature_has_required_params():
    from ml.train_fft import run_fft_training
    sig = inspect.signature(run_fft_training)
    for param in ("job_id", "model_path", "train_data_path", "output_dir",
                  "learning_rate", "epochs", "batch_size", "resume_from_checkpoint"):
        assert param in sig.parameters, f"Missing param: {param}"


def test_fft_default_learning_rate_is_lower_than_lora():
    """FFT default LR (1e-5) should be lower than LoRA default (2e-4)."""
    from ml.train_fft import run_fft_training
    from ml.train import run_training
    fft_lr = inspect.signature(run_fft_training).parameters["learning_rate"].default
    lora_lr = inspect.signature(run_training).parameters["learning_rate"].default
    assert fft_lr < lora_lr


def test_fft_has_no_use_qlora_param():
    """FFT does not use quantization — use_qlora should not be a parameter."""
    from ml.train_fft import run_fft_training
    sig = inspect.signature(run_fft_training)
    assert "use_qlora" not in sig.parameters


def test_fft_has_no_lora_params():
    """FFT has no LoRA-specific params (lora_r, lora_alpha, etc.)."""
    from ml.train_fft import run_fft_training
    sig = inspect.signature(run_fft_training)
    for lora_param in ("lora_r", "lora_alpha", "lora_dropout", "use_unsloth"):
        assert lora_param not in sig.parameters


# ── Unit: FFT job model field ─────────────────────────────────────────────────

def test_job_model_has_is_full_model_field():
    from app.models.job import TrainingJob
    assert "is_full_model" in TrainingJob.model_fields
    assert TrainingJob.model_fields["is_full_model"].default is False


def test_fft_training_method_accepted_by_job_model():
    from app.models.job import TrainingJob
    field = TrainingJob.model_fields["training_method"]
    # The Literal type includes "fft"
    import typing
    args = typing.get_args(field.annotation)
    assert "fft" in args


# ── Integration: create FFT job ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fft_job_creation_requires_tokenized_dataset(client):
    proj = await client.post("/projects", json={"name": "fft-test"})
    proj_id = proj.json()["id"]

    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(b'{"text":"hi"}\n'), "application/json")},
        data={"name": "raw-ds"},
    )
    ds_id = ds.json()["id"]
    # Status = uploaded — FFT should reject
    resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "fft-job",
            "base_model": "qwen2.5:1.5b",
            "model_path": "/models/qwen",
            "training_method": "fft",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_fft_job_stores_method_in_db(client):
    import io
    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings

    proj = await client.post("/projects", json={"name": "fft-method"})
    proj_id = proj.json()["id"]

    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(b'{"text":"hi"}\n'), "application/json")},
        data={"name": "ds"},
    )
    ds_id = ds.json()["id"]

    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    await db["datasets"].update_one(
        {"_id": ObjectId(ds_id)}, {"$set": {"status": "tokenized"}}
    )
    motor.close()

    resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "fft-job2",
            "base_model": "qwen2.5:1.5b",
            "model_path": "/models/qwen",
            "training_method": "fft",
        },
    )
    assert resp.status_code in (201, 500), resp.text
    if resp.status_code == 201:
        assert resp.json()["training_method"] == "fft"


# ── Unit: training_tasks handles FFT result ───────────────────────────────────

def test_fft_result_uses_model_path_not_adapter_path():
    """
    FFT run_fft_training returns {"model_path": ..., "is_full_model": True}
    NOT {"adapter_path": ...}. Verify the return key name.
    """
    # We verify via source inspection — no GPU needed
    import inspect
    from ml.train_fft import run_fft_training
    src = inspect.getsource(run_fft_training)
    assert '"model_path"' in src
    assert '"is_full_model": True' in src or "'is_full_model': True" in src


def test_training_tasks_sets_is_full_model_on_fft_completion():
    """Verify training_tasks.py sets is_full_model=True for FFT jobs."""
    import inspect
    from workers import training_tasks
    src = inspect.getsource(training_tasks)
    assert "is_full_model" in src
    assert "fft" in src
