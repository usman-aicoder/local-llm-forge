"""
Tests for /projects — CRUD operations.
"""
import pytest


# ── Create ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_project_returns_201(client):
    resp = await client.post("/projects", json={"name": "My Project"})
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_create_project_returns_id_and_name(client):
    resp = await client.post("/projects", json={"name": "Alpha", "description": "Test"})
    data = resp.json()
    assert "id" in data
    assert data["name"] == "Alpha"
    assert data["description"] == "Test"


@pytest.mark.asyncio
async def test_create_project_without_description(client):
    resp = await client.post("/projects", json={"name": "No Desc"})
    assert resp.status_code == 201
    assert resp.json()["description"] is None


@pytest.mark.asyncio
async def test_create_project_missing_name_returns_422(client):
    resp = await client.post("/projects", json={"description": "No name"})
    assert resp.status_code == 422


# ── List ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_projects_empty(client):
    resp = await client.get("/projects")
    assert resp.status_code == 200
    assert resp.json() == {"projects": []}


@pytest.mark.asyncio
async def test_list_projects_returns_created_projects(client):
    await client.post("/projects", json={"name": "P1"})
    await client.post("/projects", json={"name": "P2"})
    resp = await client.get("/projects")
    names = [p["name"] for p in resp.json()["projects"]]
    assert "P1" in names
    assert "P2" in names


@pytest.mark.asyncio
async def test_list_projects_count(client):
    for i in range(3):
        await client.post("/projects", json={"name": f"Project {i}"})
    resp = await client.get("/projects")
    assert len(resp.json()["projects"]) == 3


# ── Get ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_project_by_id(client, project):
    resp = await client.get(f"/projects/{project['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == project["id"]


@pytest.mark.asyncio
async def test_get_nonexistent_project_returns_404(client):
    resp = await client.get("/projects/000000000000000000000000")
    assert resp.status_code == 404


# ── Update ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_project_name(client, project):
    resp = await client.patch(f"/projects/{project['id']}", json={"name": "Renamed"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Renamed"


@pytest.mark.asyncio
async def test_update_project_description(client, project):
    resp = await client.patch(f"/projects/{project['id']}", json={"description": "New desc"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "New desc"


@pytest.mark.asyncio
async def test_update_nonexistent_project_returns_404(client):
    resp = await client.patch("/projects/000000000000000000000000", json={"name": "X"})
    assert resp.status_code == 404


# ── Delete ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_project_returns_204(client, project):
    resp = await client.delete(f"/projects/{project['id']}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_project_removes_from_list(client, project):
    await client.delete(f"/projects/{project['id']}")
    resp = await client.get("/projects")
    ids = [p["id"] for p in resp.json()["projects"]]
    assert project["id"] not in ids


@pytest.mark.asyncio
async def test_delete_nonexistent_project_returns_404(client):
    resp = await client.delete("/projects/000000000000000000000000")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_deleted_project_returns_404(client, project):
    await client.delete(f"/projects/{project['id']}")
    resp = await client.get(f"/projects/{project['id']}")
    assert resp.status_code == 404
