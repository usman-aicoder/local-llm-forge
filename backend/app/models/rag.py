from datetime import datetime
from typing import Literal
from beanie import Document, Link
from pydantic import Field

from app.models.project import Project


class RAGCollection(Document):
    project_id: Link[Project]
    name: str
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    qdrant_collection: str
    document_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "rag_collections"


class RAGDocument(Document):
    collection_id: Link[RAGCollection]
    filename: str
    file_path: str
    chunk_count: int | None = None
    status: Literal["uploaded", "processing", "indexed", "failed"] = "uploaded"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "rag_documents"
