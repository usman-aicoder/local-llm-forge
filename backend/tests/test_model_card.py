"""
Phase 3.3 — Model card generation tests.
"""
import io
from pathlib import Path

import pytest


# ── Unit: generate_model_card ─────────────────────────────────────────────────

def _sample_job(**overrides):
    base = {
        "name": "test-job",
        "base_model": "mistral:7b",
        "model_path": "/models/mistral",
        "training_method": "sft",
        "use_qlora": True,
        "use_unsloth": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 2,
        "grad_accum": 8,
        "max_seq_len": 2048,
        "bf16": True,
    }
    base.update(overrides)
    return base


def test_model_card_contains_job_name():
    from ml.model_card import generate_model_card
    card = generate_model_card(_sample_job(name="my-custom-model"))
    assert "my-custom-model" in card


def test_model_card_contains_base_model():
    from ml.model_card import generate_model_card
    card = generate_model_card(_sample_job(base_model="llama3:8b"))
    assert "llama3:8b" in card


def test_model_card_contains_training_method():
    from ml.model_card import generate_model_card
    card = generate_model_card(_sample_job(training_method="dpo"))
    assert "DPO" in card or "dpo" in card


def test_model_card_contains_lora_params():
    from ml.model_card import generate_model_card
    card = generate_model_card(_sample_job(lora_r=64, lora_alpha=128))
    assert "64" in card
    assert "128" in card


def test_model_card_has_yaml_frontmatter():
    from ml.model_card import generate_model_card
    card = generate_model_card(_sample_job())
    assert card.startswith("---")
    assert "library_name: peft" in card
    assert "local-llm-forge" in card


def test_model_card_qlora_tag():
    from ml.model_card import generate_model_card
    card_qlora = generate_model_card(_sample_job(use_qlora=True))
    card_lora  = generate_model_card(_sample_job(use_qlora=False))
    assert "qlora" in card_qlora
    # non-qlora card should not have the qlora tag in frontmatter
    assert "  - lora\n" in card_lora


def test_model_card_with_evaluation():
    from ml.model_card import generate_model_card
    evaluation = {
        "rouge_1": 0.45,
        "rouge_2": 0.22,
        "rouge_l": 0.40,
        "bleu": 0.18,
        "perplexity": 12.5,
        "sample_results": [{}] * 10,
    }
    card = generate_model_card(_sample_job(), evaluation=evaluation)
    assert "0.4500" in card
    assert "0.1800" in card
    assert "12.5000" in card
    assert "10 samples" in card


def test_model_card_with_dataset():
    from ml.model_card import generate_model_card
    dataset = {"name": "Orca DPO", "row_count": 300, "format_type": "alpaca"}
    card = generate_model_card(_sample_job(), dataset=dataset)
    assert "Orca DPO" in card
    assert "300" in card


def test_model_card_without_evaluation_shows_placeholder():
    from ml.model_card import generate_model_card
    card = generate_model_card(_sample_job(), evaluation=None)
    assert "Evaluation not run" in card or "not run" in card.lower()


def test_model_card_contains_usage_snippet():
    from ml.model_card import generate_model_card
    card = generate_model_card(_sample_job())
    assert "from peft import PeftModel" in card
    assert "PeftModel.from_pretrained" in card


def test_save_model_card_writes_file(tmp_path):
    from ml.model_card import save_model_card
    path = save_model_card(_sample_job(), adapter_path=str(tmp_path))
    assert Path(path).exists()
    content = Path(path).read_text()
    assert "test-job" in content


def test_save_model_card_filename(tmp_path):
    from ml.model_card import save_model_card
    path = save_model_card(_sample_job(), adapter_path=str(tmp_path))
    assert Path(path).name == "MODEL_CARD.md"


# ── Integration: /jobs/{id}/model-card endpoint ───────────────────────────────

@pytest.mark.asyncio
async def test_model_card_404_when_not_generated(client):
    proj = await client.post("/projects", json={"name": "card-test"})
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
            "name": "card-job",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
        },
    )
    if job_resp.status_code != 201:
        pytest.skip("Job creation failed (Celery not running)")

    job_id = job_resp.json()["id"]
    resp = await client.get(f"/jobs/{job_id}/model-card")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_model_card_generate_and_retrieve(client, tmp_path):
    """POST generates the card; GET retrieves it with correct content."""
    proj = await client.post("/projects", json={"name": "card-test2"})
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
            "name": "card-gen",
            "base_model": "mistral:7b",
            "model_path": "/models/mistral",
        },
    )
    if job_resp.status_code != 201:
        motor.close()
        pytest.skip("Job creation failed (Celery not running)")

    job_id = job_resp.json()["id"]

    # Fake adapter directory and completed status
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    await db["training_jobs"].update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {
            "status": "completed",
            "adapter_path": str(adapter_dir),
        }},
    )
    motor.close()

    # Generate
    post_resp = await client.post(f"/jobs/{job_id}/model-card")
    assert post_resp.status_code == 200
    data = post_resp.json()
    assert "content" in data
    assert "model_card_path" in data
    assert "card-gen" in data["content"]
    assert (adapter_dir / "MODEL_CARD.md").exists()

    # Retrieve
    get_resp = await client.get(f"/jobs/{job_id}/model-card")
    assert get_resp.status_code == 200
    assert get_resp.json()["content"] == data["content"]
