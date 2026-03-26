"""
Phase 3.2 — vLLM export tests.
"""
import io
import json
from pathlib import Path

import pytest


# ── Unit: prepare_vllm_export ─────────────────────────────────────────────────

def test_vllm_export_generates_launch_command(tmp_path):
    from ml.export_vllm import prepare_vllm_export
    # Create a minimal merged model directory with config.json
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    result = prepare_vllm_export(str(tmp_path))
    assert "launch_command" in result
    assert "vllm_model_path" in result
    assert "vllm.entrypoints.openai.api_server" in result["launch_command"]
    assert str(tmp_path) in result["launch_command"]


def test_vllm_export_launch_command_contains_model_path(tmp_path):
    from ml.export_vllm import prepare_vllm_export
    (tmp_path / "config.json").write_text("{}")
    result = prepare_vllm_export(str(tmp_path))
    assert "--model" in result["launch_command"]
    assert str(tmp_path) in result["vllm_model_path"]


def test_vllm_export_raises_if_dir_missing():
    from ml.export_vllm import prepare_vllm_export
    with pytest.raises(FileNotFoundError, match="not found"):
        prepare_vllm_export("/nonexistent/path/to/model")


def test_vllm_export_raises_if_config_missing(tmp_path):
    from ml.export_vllm import prepare_vllm_export
    # Directory exists but no config.json
    with pytest.raises(FileNotFoundError, match="config.json"):
        prepare_vllm_export(str(tmp_path))


def test_vllm_export_custom_port(tmp_path):
    from ml.export_vllm import prepare_vllm_export
    (tmp_path / "config.json").write_text("{}")
    result = prepare_vllm_export(str(tmp_path), port=9000)
    assert "--port 9000" in result["launch_command"]


# ── Integration: /jobs/{id}/export-vllm endpoint ─────────────────────────────

@pytest.mark.asyncio
async def test_export_vllm_requires_merged_model(client):
    """Endpoint must return 400 if merged_path is not set."""
    # Create a completed job with adapter_path but no merged_path
    proj = await client.post("/projects", json={"name": "vllm-test"})
    proj_id = proj.json()["id"]

    ds_data = b'{"text": "hello"}\n'
    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(ds_data), "application/json")},
        data={"name": "ds"},
    )
    ds_id = ds.json()["id"]

    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings
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
            "name": "vllm-job",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
        },
    )
    if job_resp.status_code != 201:
        pytest.skip("Job creation failed (Celery not running)")

    job_id = job_resp.json()["id"]
    resp = await client.post(f"/jobs/{job_id}/export-vllm")
    assert resp.status_code == 400
    assert "merge" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_export_vllm_stores_launch_cmd_on_job(client, tmp_path):
    """When merged_path exists with config.json, launch_cmd is saved to the job."""
    proj = await client.post("/projects", json={"name": "vllm-test2"})
    proj_id = proj.json()["id"]

    ds_data = b'{"text": "hi"}\n'
    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(ds_data), "application/json")},
        data={"name": "ds2"},
    )
    ds_id = ds.json()["id"]

    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings
    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    await db["datasets"].update_one(
        {"_id": ObjectId(ds_id)}, {"$set": {"status": "tokenized"}}
    )

    job_resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "vllm-job2",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
        },
    )
    if job_resp.status_code != 201:
        motor.close()
        pytest.skip("Job creation failed (Celery not running)")

    job_id = job_resp.json()["id"]

    # Inject a fake merged_path with config.json
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "config.json").write_text("{}")
    await db["training_jobs"].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"merged_path": str(merged_dir), "status": "completed"}},
    )
    motor.close()

    resp = await client.post(f"/jobs/{job_id}/export-vllm")
    assert resp.status_code == 200
    data = resp.json()
    assert "launch_command" in data
    assert "vllm_model_path" in data
    assert "vllm.entrypoints.openai.api_server" in data["launch_command"]
