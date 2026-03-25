"""
Tests for dataset pipeline status gate enforcement.

The pipeline has strict ordering:
  uploaded → inspected → cleaned → formatted → tokenized

Each step rejects requests if the dataset is not in the correct preceding state.
These tests verify those gates without running actual ML operations.
"""
import pytest


# ── Clean gate ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clean_rejected_on_uploaded_status(client, uploaded_dataset):
    """Clean requires 'inspected' status — must reject plain 'uploaded'."""
    resp = await client.post(
        f"/datasets/{uploaded_dataset['id']}/clean",
        json={
            "strip_html": True,
            "normalize_whitespace": True,
            "remove_urls": True,
            "deduplicate": True,
            "filter_short": True,
        },
    )
    assert resp.status_code == 400
    assert "inspect" in resp.json()["detail"].lower()


# ── Format gate ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_format_rejected_on_uploaded_status(client, uploaded_dataset):
    """Format requires at least 'inspected' status."""
    resp = await client.post(
        f"/datasets/{uploaded_dataset['id']}/format",
        json={"format_type": "alpaca", "base_model": "llama3.2:latest"},
    )
    assert resp.status_code == 400


# ── Tokenize gate ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tokenize_rejected_on_uploaded_status(client, uploaded_dataset):
    """Tokenize requires 'formatted' status."""
    resp = await client.post(
        f"/datasets/{uploaded_dataset['id']}/tokenize",
        json={"max_seq_len": 512, "val_split": 0.1},
    )
    assert resp.status_code == 400
    assert "formatted" in resp.json()["detail"].lower()


# ── Job creation gates ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sft_job_rejected_on_non_tokenized_dataset(client, project, uploaded_dataset):
    """SFT requires a tokenized dataset."""
    resp = await client.post(
        f"/projects/{project['id']}/jobs",
        json={
            "dataset_id": uploaded_dataset["id"],
            "name": "Test SFT",
            "base_model": "llama3.2:latest",
            "model_path": "meta-llama/Llama-3.2-1B",
            "training_method": "sft",
        },
    )
    assert resp.status_code == 400
    assert "tokenized" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_dpo_job_rejected_on_uploaded_dataset(client, project, uploaded_dataset):
    """DPO requires at minimum a formatted dataset."""
    resp = await client.post(
        f"/projects/{project['id']}/jobs",
        json={
            "dataset_id": uploaded_dataset["id"],
            "name": "Test DPO",
            "base_model": "llama3.2:latest",
            "model_path": "meta-llama/Llama-3.2-1B",
            "training_method": "dpo",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_job_creation_with_nonexistent_dataset_returns_404(client, project):
    resp = await client.post(
        f"/projects/{project['id']}/jobs",
        json={
            "dataset_id": "000000000000000000000000",
            "name": "Ghost Job",
            "base_model": "llama3.2:latest",
            "model_path": "meta-llama/Llama-3.2-1B",
            "training_method": "sft",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_job_creation_for_nonexistent_project_returns_404(client, uploaded_dataset):
    resp = await client.post(
        "/projects/000000000000000000000000/jobs",
        json={
            "dataset_id": uploaded_dataset["id"],
            "name": "Ghost",
            "base_model": "llama3.2:latest",
            "model_path": "meta-llama/Llama-3.2-1B",
            "training_method": "sft",
        },
    )
    assert resp.status_code == 404


# ── Jobs list ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_jobs_empty(client, project):
    resp = await client.get(f"/projects/{project['id']}/jobs")
    assert resp.status_code == 200
    assert resp.json()["jobs"] == []


@pytest.mark.asyncio
async def test_list_jobs_for_nonexistent_project_returns_404(client):
    resp = await client.get("/projects/000000000000000000000000/jobs")
    assert resp.status_code == 404
