"""
Knowledge base: ChromaDB over data/corpus/ markdown files.

Ingestion pipeline:
  1. Load markdown files via src.corpus.load_corpus()
  2. Chunk each doc into ~400-token passages with 50-token overlap
  3. Embed with sentence-transformers all-MiniLM-L6-v2
  4. Store in ChromaDB at data/vector_db/

Crisis resources are tagged crisis_resource=True in metadata.
Retrieval bypasses similarity search for crisis queries — metadata filter only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_DEFAULT_CORPUS_PATH = Path(__file__).parent.parent.parent / "data" / "corpus"
_DEFAULT_VECTOR_DB_PATH = Path(__file__).parent.parent.parent / "data" / "vector_db"
_COLLECTION_NAME = "student_mental_health_kb"
_CHUNK_WORDS = 300         # ~400 tokens
_CHUNK_OVERLAP_WORDS = 37  # ~50 tokens


def _chunk_text(text: str, chunk_words: int = _CHUNK_WORDS, overlap_words: int = _CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


class KnowledgeBase:
    """ChromaDB-backed knowledge base with crisis metadata routing."""

    def __init__(
        self,
        vector_db_path: str | Path = _DEFAULT_VECTOR_DB_PATH,
        collection_name: str = _COLLECTION_NAME,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        import chromadb
        from chromadb.utils import embedding_functions

        self._vector_db_path = Path(vector_db_path)
        self._vector_db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._vector_db_path))
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        crisis_only: bool = False,
    ) -> list[dict]:
        """
        Semantic search over KB.

        crisis_only=True: metadata-filtered retrieval for crisis_resource=true docs only.
        This bypasses similarity search — used when crisis layer fires.
        """
        where_filter = {"crisis_resource": True} if crisis_only else None

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self._collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[knowledge_base] query error: {e}")
            return []

        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "relevance": float(1.0 - dist),
                }
            )
        return docs

    def get_crisis_resources(self) -> list[dict]:
        """Return all crisis-tagged documents (for sidebar/persistent display)."""
        try:
            results = self._collection.get(
                where={"crisis_resource": True},
                include=["documents", "metadatas"],
            )
            return [
                {"text": doc, "metadata": meta}
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
        except Exception as e:
            print(f"[knowledge_base] get_crisis_resources error: {e}")
            return []


def build_knowledge_base(
    corpus_path: str | Path = _DEFAULT_CORPUS_PATH,
    vector_db_path: str | Path = _DEFAULT_VECTOR_DB_PATH,
    collection_name: str = _COLLECTION_NAME,
    embedding_model: str = "all-MiniLM-L6-v2",
    force_rebuild: bool = False,
) -> KnowledgeBase:
    """
    Ingest corpus into ChromaDB. Idempotent — skips if already built unless force_rebuild.
    Returns loaded KnowledgeBase.
    """
    from src.corpus import load_corpus

    kb = KnowledgeBase(
        vector_db_path=vector_db_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    if kb.doc_count > 0 and not force_rebuild:
        print(f"[knowledge_base] KB already built ({kb.doc_count} chunks). Use force_rebuild=True to re-ingest.")
        return kb

    if force_rebuild and kb.doc_count > 0:
        kb._client.delete_collection(collection_name)
        kb._collection = kb._client.create_collection(
            name=collection_name,
            embedding_function=kb._ef,
            metadata={"hnsw:space": "cosine"},
        )
        print("[knowledge_base] Existing collection deleted — rebuilding")

    docs = load_corpus(corpus_path)
    print(f"[knowledge_base] Ingesting {len(docs)} documents...")

    all_texts, all_ids, all_metas = [], [], []

    for doc in docs:
        if not doc.content.strip():
            continue

        chunks = _chunk_text(doc.content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.path.stem}_{i}"
            meta = {
                "title": doc.title,
                "category": doc.category,
                "source_url": doc.source_url,
                "last_verified": doc.last_verified,
                "crisis_resource": doc.crisis_resource,
                "doc_stem": doc.path.stem,
                "chunk_index": i,
                "n_chunks": len(chunks),
            }
            all_texts.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append(meta)

    # Batch upsert (ChromaDB default batch = 5461)
    batch_size = 500
    for i in range(0, len(all_texts), batch_size):
        kb._collection.upsert(
            documents=all_texts[i : i + batch_size],
            ids=all_ids[i : i + batch_size],
            metadatas=all_metas[i : i + batch_size],
        )
        print(f"  Upserted chunks {i} – {min(i + batch_size, len(all_texts))}")

    print(f"[knowledge_base] Built: {kb.doc_count} chunks from {len(docs)} documents")
    return kb
