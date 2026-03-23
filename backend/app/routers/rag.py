"""RAG Pipeline API — collections, documents, and streaming query."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiofiles
from beanie import PydanticObjectId
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.models.project import Project
from app.models.rag import RAGCollection, RAGDocument
from workers.rag_tasks import run_rag_ingest_task

router = APIRouter(tags=["rag"])


# ── Collections ───────────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@router.get("/projects/{project_id}/rag/collections")
async def list_collections(project_id: str):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    cols = await RAGCollection.find(
        RAGCollection.project_id.id == PydanticObjectId(project_id)
    ).to_list()
    return {"collections": [c.model_dump(mode="json") for c in cols]}


@router.post("/projects/{project_id}/rag/collections")
async def create_collection(project_id: str, body: CollectionCreate):
    project = await Project.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    qdrant_col = f"proj_{project_id[:8]}_{body.name.lower().replace(' ', '_')[:20]}"
    col = RAGCollection(
        project_id=project,
        name=body.name,
        embedding_model=body.embedding_model,
        qdrant_collection=qdrant_col,
    )
    await col.insert()
    return col.model_dump(mode="json")


@router.delete("/rag/collections/{collection_id}")
async def delete_collection(collection_id: str):
    col = await RAGCollection.get(collection_id)
    if not col:
        raise HTTPException(404, "Collection not found")
    docs = await RAGDocument.find(
        RAGDocument.collection_id.id == PydanticObjectId(collection_id)
    ).to_list()
    for doc in docs:
        try:
            Path(doc.file_path).unlink(missing_ok=True)
        except Exception:
            pass
        await doc.delete()
    try:
        from ml.rag_embed import delete_collection as qdrant_delete_col
        qdrant_path = str(settings.abs("./storage/qdrant"))
        qdrant_delete_col(col.qdrant_collection, qdrant_path)
    except Exception:
        pass
    await col.delete()
    return {"ok": True}


# ── Documents ─────────────────────────────────────────────────────────────────

@router.get("/rag/collections/{collection_id}/documents")
async def list_documents(collection_id: str):
    col = await RAGCollection.get(collection_id)
    if not col:
        raise HTTPException(404, "Collection not found")
    docs = await RAGDocument.find(
        RAGDocument.collection_id.id == PydanticObjectId(collection_id)
    ).to_list()
    return {"documents": [d.model_dump(mode="json") for d in docs]}


@router.post("/rag/collections/{collection_id}/documents")
async def upload_document(collection_id: str, file: UploadFile = File(...)):
    col = await RAGCollection.get(collection_id)
    if not col:
        raise HTTPException(404, "Collection not found")

    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed)}")

    rag_dir = settings.abs(settings.rag_documents_dir) / collection_id
    rag_dir.mkdir(parents=True, exist_ok=True)
    file_path = rag_dir / (file.filename or "upload")

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    doc = RAGDocument(
        collection_id=col,
        filename=file.filename or "upload",
        file_path=str(file_path),
        status="uploaded",
    )
    await doc.insert()
    run_rag_ingest_task.delay(str(doc.id))
    return doc.model_dump(mode="json")


@router.delete("/rag/documents/{document_id}")
async def delete_document(document_id: str):
    doc = await RAGDocument.get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    # Extract the ObjectId from the Beanie Link (stored as DBRef internally)
    _ref = doc.collection_id
    if hasattr(_ref, "ref"):
        col_oid = _ref.ref.id          # Link → DBRef → ObjectId
    elif hasattr(_ref, "id"):
        col_oid = _ref.id
    else:
        col_oid = _ref
    col = await RAGCollection.get(PydanticObjectId(str(col_oid)))
    if col:
        try:
            from ml.rag_embed import delete_document_chunks
            qdrant_path = str(settings.abs("./storage/qdrant"))
            delete_document_chunks(document_id, col.qdrant_collection, qdrant_path)
        except Exception:
            pass

    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception:
        pass
    await doc.delete()
    return {"ok": True}


# ── Query (SSE stream) ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    model: str = "llama3.2:latest"
    top_k: int = 5
    temperature: float = 0.7


@router.post("/rag/collections/{collection_id}/query")
async def query_collection(collection_id: str, body: QueryRequest):
    col = await RAGCollection.get(collection_id)
    if not col:
        raise HTTPException(404, "Collection not found")

    async def stream():
        try:
            loop = asyncio.get_event_loop()
            from ml.rag_embed import search
            qdrant_path = str(settings.abs("./storage/qdrant"))
            chunks = await loop.run_in_executor(
                None,
                lambda: search(body.question, col.qdrant_collection, qdrant_path, body.top_k),
            )
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'sources': chunks})}\n\n"

        if not chunks:
            yield f"data: {json.dumps({'token': 'No relevant context found in this collection.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        context = "\n\n---\n\n".join(c["text"] for c in chunks)

        import httpx
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{settings.ollama_url}/api/chat",
                    json={
                        "model": body.model,
                        "stream": True,
                        "options": {"temperature": body.temperature},
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful assistant that answers questions using only "
                                    "the provided document context. Be concise and accurate. "
                                    "If the answer is not in the context, say so clearly."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Context from documents:\n\n{context}\n\n"
                                    f"Question: {body.question}"
                                ),
                            },
                        ],
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if data.get("done"):
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                break
                        except Exception:
                            continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
