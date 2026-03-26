"""
OpenAI-compatible API proxy.

Translates OpenAI /v1/chat/completions → Ollama → back to OpenAI format.
Any tool that supports the OpenAI API can point at http://localhost:8010/v1.

Supports:
  GET  /v1/models
  POST /v1/chat/completions  (streaming and non-streaming)
"""
from __future__ import annotations

import json
import time
import uuid

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings

router = APIRouter(prefix="/v1", tags=["openai-compat"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert OpenAI chat messages list to a single Ollama prompt string."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM] {content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT] {content}")
        else:
            parts.append(f"[USER] {content}")
    return "\n".join(parts)


def _wrap_as_openai_response(model: str, text: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


async def _stream_ollama_as_openai(payload: dict):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", f"{settings.ollama_url}/api/generate", json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("response", "")
                done = data.get("done", False)
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": "stop" if done else None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """Return all Ollama models in OpenAI-compatible format."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception:
        models = []

    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-llm-forge",
            }
            for m in models
        ],
    }


@router.post("/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy chat completions to Ollama in OpenAI format.

    Supports both streaming (stream=true) and non-streaming responses.
    Compatible with: LangChain, LlamaIndex, Continue.dev, Cursor, Open WebUI, etc.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required")

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 512)

    prompt = _messages_to_prompt(messages)

    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if stream:
        return StreamingResponse(
            _stream_ollama_as_openai(ollama_payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/generate", json=ollama_payload
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")

    text = resp.json().get("response", "")
    return _wrap_as_openai_response(model, text)
