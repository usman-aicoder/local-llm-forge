"""
Tests for the /health endpoint.

Verifies the response shape and that the API itself is reachable.
Does NOT require Redis or Ollama to be running — those sub-checks
are allowed to return "error" in a test environment.
"""
import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_has_required_keys(client):
    resp = await client.get("/health")
    data = resp.json()
    assert "status" in data
    assert "mongo" in data
    assert "redis" in data
    assert "ollama" in data


@pytest.mark.asyncio
async def test_health_mongo_is_ok(client):
    """MongoDB must be reachable since our tests depend on it."""
    resp = await client.get("/health")
    data = resp.json()
    assert data["mongo"] == "ok", (
        "MongoDB is not reachable. Ensure mongod is running on localhost:27017."
    )


@pytest.mark.asyncio
async def test_health_status_field_is_ok(client):
    """Top-level status reflects overall health."""
    resp = await client.get("/health")
    assert resp.json()["status"] == "ok"
