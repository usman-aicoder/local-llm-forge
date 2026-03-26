"""
Phase 3.1 — OpenAI-compatible API tests.
"""
import pytest


# ── /v1/models ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models_returns_200(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_models_response_shape(client):
    resp = await client.get("/v1/models")
    data = resp.json()
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_list_models_items_have_required_fields(client):
    """Each model entry must have id, object, created, owned_by."""
    resp = await client.get("/v1/models")
    models = resp.json()["data"]
    for m in models:
        assert "id" in m
        assert "object" in m
        assert m["object"] == "model"
        assert "created" in m
        assert "owned_by" in m
        assert m["owned_by"] == "local-llm-forge"


# ── /v1/chat/completions (non-streaming) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_completions_missing_model_returns_400(client):
    resp = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_invalid_json_returns_400(client):
    resp = await client.post(
        "/v1/chat/completions",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_response_shape_when_ollama_up(client):
    """
    When Ollama is running, response must follow OpenAI format.
    Skipped if Ollama is not available.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:11434/api/tags")
            if r.status_code != 200:
                pytest.skip("Ollama not running")
    except Exception:
        pytest.skip("Ollama not running")

    # Get a model name to use
    models = (await client.get("/v1/models")).json()["data"]
    if not models:
        pytest.skip("No Ollama models installed")

    model_name = models[0]["id"]
    resp = await client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Say 'ok' only."}],
        "max_tokens": 5,
        "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]


# ── Helper: _messages_to_prompt ───────────────────────────────────────────────

def test_messages_to_prompt_formats_roles():
    from app.routers.openai_compat import _messages_to_prompt
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    prompt = _messages_to_prompt(msgs)
    assert "[SYSTEM]" in prompt
    assert "[USER]" in prompt
    assert "[ASSISTANT]" in prompt
    assert "You are helpful." in prompt
    assert "Hello" in prompt


def test_messages_to_prompt_empty_returns_empty_string():
    from app.routers.openai_compat import _messages_to_prompt
    assert _messages_to_prompt([]) == ""


def test_wrap_as_openai_response_shape():
    from app.routers.openai_compat import _wrap_as_openai_response
    resp = _wrap_as_openai_response("mistral:7b", "hello world")
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "mistral:7b"
    assert resp["choices"][0]["message"]["content"] == "hello world"
    assert resp["choices"][0]["finish_reason"] == "stop"
