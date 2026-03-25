"""
Shared test fixtures for the LLM Platform backend.

Uses a real MongoDB instance with a dedicated test database (test_llmplatform).
All fixtures are function-scoped — each test gets a fresh DB state.

Run: cd backend && pytest tests/ -v
"""
import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# Must be set before importing the app so Settings picks up the test DB
os.environ["MONGO_DB_NAME"] = "test_llmplatform"

from app.main import app           # noqa: E402
from app.models import ALL_MODELS  # noqa: E402

TEST_MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
TEST_DB_NAME   = "test_llmplatform"


# ── Per-test DB setup + teardown ─────────────────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    """
    Before each test: connect, init Beanie, yield.
    After each test: drop all collections so the next test starts empty.
    """
    motor = AsyncIOMotorClient(TEST_MONGO_URL)
    db = motor[TEST_DB_NAME]
    await init_beanie(database=db, document_models=ALL_MODELS)
    yield
    # Wipe every collection
    for name in await db.list_collection_names():
        await db[name].drop()
    motor.close()


# ── Async HTTP client ─────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(clean_db):
    """HTTPX AsyncClient wired to the FastAPI ASGI app — no real TCP."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ── Convenience: pre-created project ─────────────────────────────────────────

@pytest_asyncio.fixture
async def project(client):
    resp = await client.post("/projects", json={"name": "Test Project", "description": "unit test"})
    assert resp.status_code == 201
    return resp.json()


# ── Convenience: uploaded dataset ────────────────────────────────────────────

@pytest_asyncio.fixture
async def uploaded_dataset(client, project, tmp_path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        '{"instruction": "What is 2+2?", "output": "4"}\n'
        '{"instruction": "Capital of France?", "output": "Paris"}\n'
        '{"instruction": "Hello?", "output": "Hi there!"}\n'
    )
    with open(jsonl, "rb") as fh:
        resp = await client.post(
            f"/projects/{project['id']}/datasets/upload",
            files={"file": ("data.jsonl", fh, "application/octet-stream")},
        )
    assert resp.status_code == 201
    return resp.json()
