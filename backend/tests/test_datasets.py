"""
Tests for dataset creation, listing, deletion, and file handling.
"""
import pytest


# ── Upload ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_jsonl_returns_201(client, project, tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"instruction": "q", "output": "a"}\n')
    with open(f, "rb") as fh:
        resp = await client.post(
            f"/projects/{project['id']}/datasets/upload",
            files={"file": ("d.jsonl", fh, "application/octet-stream")},
        )
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_upload_dataset_has_correct_fields(client, project, tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"instruction": "q", "output": "a"}\n')
    with open(f, "rb") as fh:
        resp = await client.post(
            f"/projects/{project['id']}/datasets/upload",
            files={"file": ("d.jsonl", fh, "application/octet-stream")},
        )
    data = resp.json()
    assert "id" in data
    assert data["status"] == "uploaded"
    assert data["source_type"] == "upload"


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_extension(client, project, tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello")
    with open(f, "rb") as fh:
        resp = await client.post(
            f"/projects/{project['id']}/datasets/upload",
            files={"file": ("data.txt", fh, "text/plain")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_to_nonexistent_project_returns_404(client, tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"instruction": "q", "output": "a"}\n')
    with open(f, "rb") as fh:
        resp = await client.post(
            "/projects/000000000000000000000000/datasets/upload",
            files={"file": ("d.jsonl", fh, "application/octet-stream")},
        )
    assert resp.status_code == 404


# ── List ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_datasets_empty(client, project):
    resp = await client.get(f"/projects/{project['id']}/datasets")
    assert resp.status_code == 200
    assert resp.json()["datasets"] == []


@pytest.mark.asyncio
async def test_list_datasets_returns_uploaded(client, project, uploaded_dataset):
    resp = await client.get(f"/projects/{project['id']}/datasets")
    ids = [d["id"] for d in resp.json()["datasets"]]
    assert uploaded_dataset["id"] in ids


@pytest.mark.asyncio
async def test_list_datasets_for_nonexistent_project_returns_404(client):
    resp = await client.get("/projects/000000000000000000000000/datasets")
    assert resp.status_code == 404


# ── Get ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_dataset_by_id(client, uploaded_dataset):
    resp = await client.get(f"/datasets/{uploaded_dataset['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == uploaded_dataset["id"]


@pytest.mark.asyncio
async def test_get_nonexistent_dataset_returns_404(client):
    resp = await client.get("/datasets/000000000000000000000000")
    assert resp.status_code == 404


# ── Delete ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_dataset_returns_204(client, uploaded_dataset):
    resp = await client.delete(f"/datasets/{uploaded_dataset['id']}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_dataset_removes_from_list(client, project, uploaded_dataset):
    await client.delete(f"/datasets/{uploaded_dataset['id']}")
    resp = await client.get(f"/projects/{project['id']}/datasets")
    ids = [d["id"] for d in resp.json()["datasets"]]
    assert uploaded_dataset["id"] not in ids


@pytest.mark.asyncio
async def test_delete_nonexistent_dataset_returns_404(client):
    resp = await client.delete("/datasets/000000000000000000000000")
    assert resp.status_code == 404
