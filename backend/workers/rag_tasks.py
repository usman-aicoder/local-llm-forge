"""Celery task: ingest a RAG document (chunk + embed + store in Qdrant)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from bson import ObjectId
from pymongo import MongoClient

from app.config import settings
from workers.celery_app import celery_app


def _get_db():
    client = MongoClient(settings.mongo_url)
    return client[settings.mongo_db_name]


@celery_app.task(bind=True, name="workers.rag_tasks.run_rag_ingest_task")
def run_rag_ingest_task(self, document_id: str) -> dict:
    db = _get_db()
    try:
        doc = db["rag_documents"].find_one({"_id": ObjectId(document_id)})
        if not doc:
            return {"error": "Document not found"}

        # Resolve collection_id (may be Beanie DBRef)
        col_id = doc["collection_id"]
        if hasattr(col_id, "id"):
            col_id = col_id.id

        col = db["rag_collections"].find_one({"_id": col_id})
        if not col:
            return {"error": "Collection not found"}

        db["rag_documents"].update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"status": "processing"}},
        )

        from ml.rag_embed import index_document
        qdrant_path = str(settings.abs("./storage/qdrant"))

        chunk_count = index_document(
            file_path=doc["file_path"],
            collection_name=col["qdrant_collection"],
            document_id=document_id,
            qdrant_path=qdrant_path,
        )

        db["rag_documents"].update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"status": "indexed", "chunk_count": chunk_count}},
        )

        # Update collection document count
        total = db["rag_documents"].count_documents(
            {"collection_id": col_id, "status": "indexed"}
        )
        db["rag_collections"].update_one(
            {"_id": col_id},
            {"$set": {"document_count": total}},
        )

        return {"chunk_count": chunk_count}

    except Exception:
        err = traceback.format_exc()
        db["rag_documents"].update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"status": "failed"}},
        )
        return {"error": err}
