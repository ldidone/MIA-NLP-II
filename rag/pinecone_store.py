"""Pinecone vector store with integrated embeddings (llama-text-embed-v2)."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


EMBED_MODEL = "llama-text-embed-v2"
RERANK_MODEL = "bge-reranker-v2-m3"
TEXT_FIELD = "content"
DEFAULT_INDEX = "cv-rag"
CLOUD = "aws"
REGION = "us-east-1"

TEXT_BATCH_LIMIT = 96
INDEX_READY_TIMEOUT_S = 120
INDEX_POLL_INTERVAL_S = 2


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk along with its score and source metadata."""

    id: str
    score: float
    content: str
    source: str
    chunk_index: int


def namespace_for(file_bytes: bytes, prefix: str = "cv") -> str:
    """Deterministic namespace derived from CV bytes (so re-uploads are idempotent)."""
    digest = hashlib.sha1(file_bytes).hexdigest()[:12]
    return f"{prefix}_{digest}"


class PineconeStore:
    """Thin wrapper around the Pinecone SDK for the CV RAG use case."""

    def __init__(self, index_name: Optional[str] = None) -> None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set in the environment.")

        self.index_name = index_name or os.getenv("PINECONE_INDEX", DEFAULT_INDEX)
        self.pc = Pinecone(api_key=api_key)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self) -> None:
        if self.pc.has_index(self.index_name):
            return

        self.pc.create_index_for_model(
            name=self.index_name,
            cloud=CLOUD,
            region=REGION,
            embed={"model": EMBED_MODEL, "field_map": {"text": TEXT_FIELD}},
        )

        deadline = time.time() + INDEX_READY_TIMEOUT_S
        while time.time() < deadline:
            description = self.pc.describe_index(self.index_name)
            if getattr(description.status, "ready", False):
                return
            time.sleep(INDEX_POLL_INTERVAL_S)
        raise TimeoutError(
            f"Pinecone index {self.index_name!r} did not become ready in "
            f"{INDEX_READY_TIMEOUT_S}s."
        )

    def namespace_exists(self, namespace: str) -> bool:
        try:
            stats = self.index.describe_index_stats()
        except Exception:
            return False
        namespaces = getattr(stats, "namespaces", None) or {}
        ns_info = namespaces.get(namespace)
        if ns_info is None:
            return False
        count = getattr(ns_info, "vector_count", None)
        if count is None and isinstance(ns_info, dict):
            count = ns_info.get("vector_count", 0)
        return bool(count)

    def upsert_chunks(
        self,
        namespace: str,
        chunks: List[str],
        source: str,
        wait_for_indexing: bool = True,
    ) -> int:
        """Upsert chunks into the given namespace. Returns the number written.

        Records use the integrated embedding model: only the `content` field is
        embedded; remaining fields are stored as metadata.
        """
        if not chunks:
            return 0

        records = [
            {
                "_id": f"{namespace}_chunk_{i}",
                TEXT_FIELD: chunk,
                "source": source,
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]

        for batch in _batched(records, TEXT_BATCH_LIMIT):
            self.index.upsert_records(namespace, batch)

        if wait_for_indexing:
            time.sleep(10)

        return len(records)

    def search(
        self,
        namespace: str,
        question: str,
        top_k: int = 5,
        rerank: bool = True,
    ) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for `question` from `namespace`."""
        if not question.strip():
            return []

        query: dict = {
            "top_k": top_k * 2 if rerank else top_k,
            "inputs": {"text": question},
        }

        kwargs: dict = {"namespace": namespace, "query": query}
        if rerank:
            kwargs["rerank"] = {
                "model": RERANK_MODEL,
                "top_n": top_k,
                "rank_fields": [TEXT_FIELD],
            }

        results = self.index.search(**kwargs)
        hits = results["result"]["hits"]

        retrieved: List[RetrievedChunk] = []
        for hit in hits:
            fields = hit.get("fields", {}) or {}
            retrieved.append(
                RetrievedChunk(
                    id=hit["_id"],
                    score=float(hit.get("_score", 0.0)),
                    content=fields.get(TEXT_FIELD, ""),
                    source=fields.get("source", ""),
                    chunk_index=int(fields.get("chunk_index", -1)),
                )
            )
        return retrieved

    def delete_namespace(self, namespace: str) -> None:
        try:
            self.index.delete(namespace=namespace, delete_all=True)
        except Exception:
            pass


def _batched(items: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
