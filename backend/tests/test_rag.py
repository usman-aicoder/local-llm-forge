"""
Tests for RAG collections and documents — CRUD operations.
Does NOT test the actual embedding or vector search (those require GPU/model download).
"""
import pytest


# ── Collections ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_collection_returns_200(client, project):
    resp = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Test Docs", "embedding_model": "BAAI/bge-small-en-v1.5"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_create_collection_has_correct_fields(client, project):
    resp = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "My Docs"},
    )
    data = resp.json()
    assert "id" in data
    assert data["name"] == "My Docs"
    assert "qdrant_collection" in data
    assert data["document_count"] == 0


@pytest.mark.asyncio
async def test_create_collection_for_nonexistent_project_returns_404(client):
    resp = await client.post(
        "/projects/000000000000000000000000/rag/collections",
        json={"name": "Orphan"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_collections_empty(client, project):
    resp = await client.get(f"/projects/{project['id']}/rag/collections")
    assert resp.status_code == 200
    assert resp.json()["collections"] == []


@pytest.mark.asyncio
async def test_list_collections_returns_created(client, project):
    await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Col A"},
    )
    await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Col B"},
    )
    resp = await client.get(f"/projects/{project['id']}/rag/collections")
    names = [c["name"] for c in resp.json()["collections"]]
    assert "Col A" in names
    assert "Col B" in names


@pytest.mark.asyncio
async def test_delete_collection_returns_200(client, project):
    create = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "To Delete"},
    )
    col_id = create.json()["id"]
    resp = await client.delete(f"/rag/collections/{col_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@pytest.mark.asyncio
async def test_delete_collection_removes_from_list(client, project):
    create = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Gone"},
    )
    col_id = create.json()["id"]
    await client.delete(f"/rag/collections/{col_id}")
    resp = await client.get(f"/projects/{project['id']}/rag/collections")
    ids = [c["id"] for c in resp.json()["collections"]]
    assert col_id not in ids


@pytest.mark.asyncio
async def test_delete_nonexistent_collection_returns_404(client):
    resp = await client.delete("/rag/collections/000000000000000000000000")
    assert resp.status_code == 404


# ── Documents ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_document_returns_2xx(client, project, tmp_path):
    col = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Docs"},
    )
    col_id = col.json()["id"]

    txt = tmp_path / "test.txt"
    txt.write_text("Hello world. This is a test document.")
    with open(txt, "rb") as fh:
        resp = await client.post(
            f"/rag/collections/{col_id}/documents",
            files={"file": ("test.txt", fh, "text/plain")},
        )
    assert resp.status_code in (200, 201)


@pytest.mark.asyncio
async def test_upload_document_initial_status_is_uploaded(client, project, tmp_path):
    col = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Docs"},
    )
    col_id = col.json()["id"]

    txt = tmp_path / "test.txt"
    txt.write_text("Some content.")
    with open(txt, "rb") as fh:
        resp = await client.post(
            f"/rag/collections/{col_id}/documents",
            files={"file": ("test.txt", fh, "text/plain")},
        )
    assert resp.json()["status"] == "uploaded"


@pytest.mark.asyncio
async def test_list_documents_empty(client, project):
    col = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Empty"},
    )
    col_id = col.json()["id"]
    resp = await client.get(f"/rag/collections/{col_id}/documents")
    assert resp.status_code == 200
    assert resp.json()["documents"] == []


@pytest.mark.asyncio
async def test_delete_document_removes_from_list(client, project, tmp_path):
    col = await client.post(
        f"/projects/{project['id']}/rag/collections",
        json={"name": "Docs"},
    )
    col_id = col.json()["id"]

    txt = tmp_path / "file.txt"
    txt.write_text("Content.")
    with open(txt, "rb") as fh:
        upload = await client.post(
            f"/rag/collections/{col_id}/documents",
            files={"file": ("file.txt", fh, "text/plain")},
        )
    doc_id = upload.json()["id"]

    del_resp = await client.delete(f"/rag/documents/{doc_id}")
    assert del_resp.status_code == 200

    list_resp = await client.get(f"/rag/collections/{col_id}/documents")
    ids = [d["id"] for d in list_resp.json()["documents"]]
    assert doc_id not in ids


@pytest.mark.asyncio
async def test_delete_nonexistent_document_returns_404(client):
    resp = await client.delete("/rag/documents/000000000000000000000000")
    assert resp.status_code == 404
