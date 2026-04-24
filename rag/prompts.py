"""Prompt templates for the CV-grounded RAG chatbot."""

from __future__ import annotations

from typing import List

from .pinecone_store import RetrievedChunk


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly about the "
    "candidate's CV using only the provided context. "
    "If the answer is not contained in the context, say you don't know. "
    "Be concise, factual, and cite the relevant context numbers in square "
    "brackets like [1], [2] when useful. Answer in the same language as the question."
)


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered context block."""
    if not chunks:
        return "(no context retrieved)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] (source: {c.source}, chunk #{c.chunk_index}, score={c.score:.3f})"
        blocks.append(f"{header}\n{c.content}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context = format_context(chunks)
    return (
        "Use the following CV excerpts to answer the question.\n\n"
        f"=== CV CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"Question: {question}\n\n"
        "Answer using only the CV context above. "
        "If the information is not present, reply that the CV does not contain it."
    )
