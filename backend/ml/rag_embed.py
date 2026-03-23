"""
RAG embedding pipeline: extract text → chunk → embed → store in Qdrant (local).

Uses sentence-transformers BAAI/bge-small-en-v1.5 (CPU-friendly, ~130 MB).
Qdrant runs in local file mode — no separate server required.
"""
from __future__ import annotations

import uuid
from pathlib import Path

CHUNK_SIZE = 400     # words per chunk
CHUNK_OVERLAP = 40  # words overlap between chunks
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384     # bge-small output dimension


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(file_path: str) -> str:
    p = Path(file_path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        if not pages:
            raise ValueError("PDF appears to be image-only — no extractable text")
        return "\n\n".join(pages)
    else:
        return p.read_text(encoding="utf-8", errors="ignore")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    vecs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _get_client(qdrant_path: str):
    from qdrant_client import QdrantClient
    return QdrantClient(path=qdrant_path)


def _ensure_collection(client, collection_name: str) -> None:
    from qdrant_client.models import Distance, VectorParams
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


# ── Public API ────────────────────────────────────────────────────────────────

def index_document(
    file_path: str,
    collection_name: str,
    document_id: str,
    qdrant_path: str,
) -> int:
    """Extract, chunk, embed, and upsert a document. Returns chunk count."""
    text = _extract_text(file_path)
    chunks = _chunk_text(text)
    if not chunks:
        return 0

    vectors = embed_texts(chunks)

    client = _get_client(qdrant_path)
    _ensure_collection(client, collection_name)

    from qdrant_client.models import PointStruct
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "document_id": document_id,
                "chunk_index": i,
                "text": chunks[i],
            },
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(chunks)


def search(
    query: str,
    collection_name: str,
    qdrant_path: str,
    top_k: int = 5,
) -> list[dict]:
    """Search for top-k relevant chunks. Returns list of {text, score, document_id, chunk_index}."""
    vecs = embed_texts([query])
    query_vec = vecs[0]

    client = _get_client(qdrant_path)
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        return []

    # qdrant-client >= 1.7 uses query_points(); older versions used search()
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vec,
            limit=top_k,
        )
        hits = response.points
    except AttributeError:
        # fallback for older qdrant-client
        hits = client.search(  # type: ignore[attr-defined]
            collection_name=collection_name,
            query_vector=query_vec,
            limit=top_k,
        )

    return [
        {
            "text": r.payload["text"],
            "score": round(r.score, 4),
            "document_id": r.payload["document_id"],
            "chunk_index": r.payload["chunk_index"],
        }
        for r in hits
    ]


def delete_document_chunks(
    document_id: str,
    collection_name: str,
    qdrant_path: str,
) -> None:
    """Remove all chunks belonging to a document from Qdrant."""
    client = _get_client(qdrant_path)
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        return
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )


def delete_collection(collection_name: str, qdrant_path: str) -> None:
    client = _get_client(qdrant_path)
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name)
