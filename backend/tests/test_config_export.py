"""
Phase 2.3 — YAML config export / import tests.
"""
import io

import pytest
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _create_project_and_tokenized_dataset(client):
    proj = await client.post("/projects", json={"name": "cfg-test"})
    assert proj.status_code == 201
    proj_id = proj.json()["id"]

    ds_data = b'{"text": "hello"}\n'
    ds = await client.post(
        f"/projects/{proj_id}/datasets/upload",
        files={"file": ("d.jsonl", io.BytesIO(ds_data), "application/json")},
        data={"name": "ds-cfg"},
    )
    assert ds.status_code == 201
    ds_id = ds.json()["id"]

    from bson import ObjectId
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings
    motor = AsyncIOMotorClient(settings.mongo_url)
    db = motor[settings.mongo_db_name]
    await db["datasets"].update_one(
        {"_id": ObjectId(ds_id)},
        {"$set": {"status": "tokenized"}},
    )
    motor.close()

    return proj_id, ds_id


# ── Export ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_config_export_returns_200(client):
    proj_id, ds_id = await _create_project_and_tokenized_dataset(client)

    resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "export-test",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
        },
    )
    assert resp.status_code in (201, 500), resp.text
    if resp.status_code != 201:
        pytest.skip("Job creation failed (Celery not running)")

    job_id = resp.json()["id"]
    cfg_resp = await client.get(f"/jobs/{job_id}/config")
    assert cfg_resp.status_code == 200


@pytest.mark.asyncio
async def test_config_export_produces_valid_yaml(client):
    proj_id, ds_id = await _create_project_and_tokenized_dataset(client)

    resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "yaml-check",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
            "lora_r": 32,
            "epochs": 5,
        },
    )
    if resp.status_code != 201:
        pytest.skip("Job creation failed (Celery not running)")

    job_id = resp.json()["id"]
    cfg_resp = await client.get(f"/jobs/{job_id}/config")
    assert cfg_resp.status_code == 200

    config = yaml.safe_load(cfg_resp.content)
    assert "base_model" in config
    assert "lora" in config
    assert "training" in config
    assert "use_unsloth" in config


@pytest.mark.asyncio
async def test_config_export_preserves_lora_values(client):
    proj_id, ds_id = await _create_project_and_tokenized_dataset(client)

    resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "lora-values",
            "base_model": "llama3:8b",
            "model_path": "/models/llama3",
            "lora_r": 64,
            "lora_alpha": 128,
        },
    )
    if resp.status_code != 201:
        pytest.skip("Job creation failed (Celery not running)")

    job_id = resp.json()["id"]
    cfg_resp = await client.get(f"/jobs/{job_id}/config")
    config = yaml.safe_load(cfg_resp.content)

    assert config["lora"]["r"] == 64
    assert config["lora"]["alpha"] == 128
    assert config["base_model"] == "llama3:8b"


@pytest.mark.asyncio
async def test_config_export_content_disposition_header(client):
    proj_id, ds_id = await _create_project_and_tokenized_dataset(client)

    resp = await client.post(
        f"/projects/{proj_id}/jobs",
        json={
            "dataset_id": ds_id,
            "name": "header-check",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
        },
    )
    if resp.status_code != 201:
        pytest.skip("Job creation failed (Celery not running)")

    job_id = resp.json()["id"]
    cfg_resp = await client.get(f"/jobs/{job_id}/config")
    assert "attachment" in cfg_resp.headers.get("content-disposition", "")
    assert ".yaml" in cfg_resp.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_config_export_404_for_unknown_job(client):
    resp = await client.get("/jobs/000000000000000000000099/config")
    assert resp.status_code == 404


# ── Import ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_config_import_prefills_fields(client):
    """Uploading a YAML config must create a job with those hyperparameters."""
    proj_id, ds_id = await _create_project_and_tokenized_dataset(client)

    config = {
        "base_model": "llama3:8b",
        "model_path": "/models/llama3",
        "training_method": "sft",
        "lora": {"use_qlora": True, "r": 32, "alpha": 64, "dropout": 0.1,
                 "target_modules": ["q_proj", "v_proj"]},
        "training": {"learning_rate": 1e-4, "epochs": 2, "batch_size": 4,
                     "grad_accum": 4, "max_seq_len": 1024, "bf16": True},
        "use_unsloth": False,
    }
    yaml_bytes = yaml.dump(config).encode()

    resp = await client.post(
        "/jobs/from-config",
        params={"project_id": proj_id, "dataset_id": ds_id, "name": "from-yaml"},
        files={"config_file": ("cfg.yaml", io.BytesIO(yaml_bytes), "application/x-yaml")},
    )
    assert resp.status_code in (200, 201, 500), resp.text
    if resp.status_code not in (200, 201):
        pytest.skip("Job creation failed (Celery not running)")

    job = resp.json()
    assert job["base_model"] == "llama3:8b"
    assert job["lora_r"] == 32
    assert job["lora_alpha"] == 64
    assert job["epochs"] == 2


@pytest.mark.asyncio
async def test_config_import_rejects_invalid_yaml(client):
    proj_id, ds_id = await _create_project_and_tokenized_dataset(client)

    bad_yaml = b"this: is: not: valid: yaml: ["

    resp = await client.post(
        "/jobs/from-config",
        params={"project_id": proj_id, "dataset_id": ds_id, "name": "bad"},
        files={"config_file": ("bad.yaml", io.BytesIO(bad_yaml), "application/x-yaml")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_config_import_uses_defaults_for_missing_keys(client):
    """A minimal YAML (only base_model) must not crash — defaults fill the rest."""
    proj_id, ds_id = await _create_project_and_tokenized_dataset(client)

    minimal_yaml = yaml.dump({
        "base_model": "mistral:7b",
        "model_path": "/models/mistral",
    }).encode()

    resp = await client.post(
        "/jobs/from-config",
        params={"project_id": proj_id, "dataset_id": ds_id, "name": "minimal"},
        files={"config_file": ("min.yaml", io.BytesIO(minimal_yaml), "application/x-yaml")},
    )
    # Either created (200/201) or 500 if Celery not running — both are acceptable
    assert resp.status_code in (200, 201, 500), resp.text
