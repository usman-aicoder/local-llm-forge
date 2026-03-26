"""
Phase 5.1 — Experiment tracking summary tests.
"""
import io

import pytest


# ── Unit: endpoint structure ──────────────────────────────────────────────────

def test_experiments_summary_endpoint_exists():
    """Verify the route is registered in projects router."""
    import inspect
    from app.routers import projects
    src = inspect.getsource(projects)
    assert "experiments/summary" in src


def test_experiments_summary_returns_experiments_key():
    """Source must return a dict with 'experiments' key."""
    import inspect
    from app.routers import projects
    src = inspect.getsource(projects)
    assert '"experiments"' in src or "'experiments'" in src


def test_experiments_hyperparams_keys():
    """Verify all required hyperparameter keys appear in the source."""
    import inspect
    from app.routers import projects
    src = inspect.getsource(projects)
    for key in ("learning_rate", "epochs", "batch_size", "grad_accum",
                "lora_r", "lora_alpha", "use_qlora", "use_unsloth"):
        assert key in src, f"Missing hyperparams key: {key}"


def test_experiments_metrics_keys():
    """Verify all metric keys appear in the source."""
    import inspect
    from app.routers import projects
    src = inspect.getsource(projects)
    for key in ("rouge_1", "rouge_2", "rouge_l", "bleu", "perplexity"):
        assert key in src, f"Missing metrics key: {key}"


# ── Integration: /projects/{id}/experiments/summary ──────────────────────────

@pytest.mark.asyncio
async def test_summary_404_for_unknown_project(client):
    resp = await client.get("/projects/000000000000000000000001/experiments/summary")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_summary_empty_for_new_project(client):
    proj = await client.post("/projects", json={"name": "exp-empty"})
    proj_id = proj.json()["id"]
    resp = await client.get(f"/projects/{proj_id}/experiments/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "experiments" in data
    assert data["experiments"] == []


@pytest.mark.asyncio
async def test_summary_excludes_non_completed_jobs(client):
    """Queued / running jobs must not appear in the summary."""
    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings

    proj = await client.post("/projects", json={"name": "exp-queued"})
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
            "name": "queued-job",
            "base_model": "qwen2.5:1.5b",
            "model_path": "/models/qwen",
        },
    )
    assert resp.status_code in (201, 500)

    # Summary must still be empty — job is queued, not completed
    summary = await client.get(f"/projects/{proj_id}/experiments/summary")
    assert summary.status_code == 200
    assert summary.json()["experiments"] == []


@pytest.mark.asyncio
async def test_summary_includes_completed_job(client):
    """A job manually set to 'completed' in DB must appear in summary."""
    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings

    proj = await client.post("/projects", json={"name": "exp-completed"})
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

    job_resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "done-job",
            "base_model": "qwen2.5:1.5b",
            "model_path": "/models/qwen",
        },
    )
    if job_resp.status_code != 201:
        pytest.skip("Job creation failed (Celery unavailable)")

    job_id = job_resp.json()["id"]

    # Force status to completed in DB
    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    from datetime import datetime
    await db["training_jobs"].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "completed", "completed_at": datetime.utcnow()}},
    )
    motor.close()

    summary = await client.get(f"/projects/{proj_id}/experiments/summary")
    assert summary.status_code == 200
    experiments = summary.json()["experiments"]
    assert len(experiments) == 1

    exp = experiments[0]
    assert exp["id"] == job_id
    assert exp["name"] == "done-job"
    assert "hyperparams" in exp
    assert "metrics" in exp
    assert exp["hyperparams"]["learning_rate"] == pytest.approx(2e-4)
    assert exp["metrics"]["rouge_1"] is None   # no evaluation yet


@pytest.mark.asyncio
async def test_summary_includes_evaluation_metrics(client):
    """When an Evaluation document exists, metrics must be populated."""
    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings
    from datetime import datetime

    proj = await client.post("/projects", json={"name": "exp-with-eval"})
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

    job_resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "eval-job",
            "base_model": "qwen2.5:1.5b",
            "model_path": "/models/qwen",
        },
    )
    if job_resp.status_code != 201:
        pytest.skip("Job creation failed (Celery unavailable)")

    job_id = job_resp.json()["id"]

    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    await db["training_jobs"].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "completed", "completed_at": datetime.utcnow()}},
    )
    # Insert a fake evaluation — Beanie stores Link fields as DBRefs
    await db["evaluations"].insert_one({
        "job_id": {"$ref": "training_jobs", "$id": ObjectId(job_id)},
        "rouge_1": 0.42,
        "rouge_2": 0.21,
        "rouge_l": 0.38,
        "bleu": 0.15,
        "perplexity": 12.3,
        "created_at": datetime.utcnow(),
    })
    motor.close()

    summary = await client.get(f"/projects/{proj_id}/experiments/summary")
    assert summary.status_code == 200
    experiments = summary.json()["experiments"]
    assert len(experiments) >= 1

    exp = next(e for e in experiments if e["id"] == job_id)
    assert exp["metrics"]["rouge_1"] == pytest.approx(0.42)
    assert exp["metrics"]["bleu"] == pytest.approx(0.15)
    assert exp["metrics"]["perplexity"] == pytest.approx(12.3)
