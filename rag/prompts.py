"""Prompt templates for the CV-grounded RAG chatbot."""

from __future__ import annotations

from typing import List, Optional

from .pinecone_store import RetrievedChunk


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the "
    "candidate's CV. Ground facts exclusively in the provided CV context. "
    "You may use the prior session messages only to resolve pronouns, "
    "follow-ups, and what the user is referring to—never to invent experience "
    "or facts. If the answer is not contained in the CV context, say you do "
    "not have that information. Be concise, factual, and cite the relevant "
    "context numbers in square brackets like [1], [2] when useful. "
    "Answer in the same language as the question."
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


def build_search_query(question: str, prior_messages: Optional[List[dict]] = None) -> str:
    """Merge recent turns with the question so vector search handles follow-ups."""
    q = question.strip()
    if not prior_messages:
        return q
    history_text = format_conversation_history(
        prior_messages, max_messages=4
    )
    if not history_text:
        return q
    return (
        f"Recent conversation (for disambiguation):\n{history_text}\n\n"
        f"Current question: {q}"
    )


def format_conversation_history(
    prior_messages: List[dict],
    max_messages: int = 12,
) -> Optional[str]:
    """Format earlier user/assistant text for follow-up questions."""
    if not prior_messages:
        return None
    recent = prior_messages[-max_messages:]
    lines: List[str] = []
    for m in recent:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    if not lines:
        return None
    return "\n\n".join(lines)


def build_user_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    prior_messages: Optional[List[dict]] = None,
    max_history_messages: int = 12,
) -> str:
    context = format_context(chunks)
    history_text = None
    if prior_messages:
        history_text = format_conversation_history(
            prior_messages, max_messages=max_history_messages
        )

    parts: List[str] = [
        "Use the CV context below to answer the current question. "
        "Use prior session messages only to interpret follow-up wording; "
        "all factual claims must come from the CV context.\n\n",
    ]
    if history_text:
        parts.append(
            "=== PRIOR MESSAGES (this session) ===\n"
            f"{history_text}\n"
            "=== END PRIOR MESSAGES ===\n\n"
        )
    parts.append(
        f"=== CV CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"Current question: {question}\n\n"
        "If the information is not in the CV context, say the CV does not contain it. "
        "For follow-ups, use prior messages to clarify intent when needed."
    )
    return "".join(parts)
