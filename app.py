"""Streamlit chat UI for CV question answering with RAG (Pinecone + OpenAI/Groq)."""

from __future__ import annotations

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

from rag.chunker import chunk_text
from rag.llm import (
    LLMError,
    PROVIDER_GROQ,
    PROVIDER_OPENAI,
    available_providers,
    generate,
)
from rag.loader import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError, extract_text
from rag.pinecone_store import (
    PineconeStore,
    PineconeUnauthorized,
    RetrievedChunk,
    namespace_for,
)
from rag.prompts import SYSTEM_PROMPT, build_search_query, build_user_prompt


load_dotenv()

st.set_page_config(page_title="CV RAG Chatbot", page_icon=None, layout="wide")


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_store() -> PineconeStore:
    return PineconeStore()


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("namespace", None)
    st.session_state.setdefault("cv_filename", None)
    st.session_state.setdefault("num_chunks", 0)
    st.session_state.setdefault("uploaded_hash", None)


def _ingest_cv(store: PineconeStore, file_bytes: bytes, filename: str) -> None:
    try:
        text = extract_text(file_bytes, filename)
    except UnsupportedFileTypeError as exc:
        st.error(str(exc))
        return

    if not text:
        st.error("Could not extract any text from the uploaded file.")
        return

    chunks = chunk_text(text)
    if not chunks:
        st.error("The CV produced no chunks after splitting.")
        return

    namespace = namespace_for(file_bytes)

    if store.namespace_exists(namespace):
        st.info(
            f"This CV is already indexed (namespace `{namespace}`). "
            "Skipping re-upload."
        )
    else:
        try:
            with st.spinner(f"Indexing {len(chunks)} chunks into Pinecone..."):
                store.upsert_chunks(namespace, chunks, source=filename)
        except PineconeUnauthorized:
            st.error(
                "Pinecone returned 401 Unauthorized. The index was likely deleted "
                "while the app was running. Click **Reconnect to Pinecone** in the "
                "sidebar, then re-upload the CV."
            )
            return

    st.session_state.namespace = namespace
    st.session_state.cv_filename = filename
    st.session_state.num_chunks = len(chunks)
    st.session_state.uploaded_hash = namespace
    st.session_state.messages = []
    st.success(f"CV ready. Ask questions about **{filename}**.")


def _render_sidebar(store: PineconeStore) -> str:
    with st.sidebar:
        st.header("CV RAG")
        st.caption("Upload a CV and ask questions grounded on its contents.")

        uploaded = st.file_uploader(
            "Upload CV",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            new_hash = namespace_for(file_bytes)
            if new_hash != st.session_state.get("uploaded_hash"):
                _ingest_cv(store, file_bytes, uploaded.name)

        st.divider()

        providers = available_providers()
        if not providers:
            st.error(
                "No LLM provider configured. Add OPENAI_API_KEY or GROQ_API_KEY "
                "to your .env file."
            )
            provider = PROVIDER_OPENAI
        else:
            options = [PROVIDER_OPENAI, PROVIDER_GROQ]
            default_index = 0 if PROVIDER_OPENAI in providers else 1
            provider = st.radio(
                "LLM provider",
                options=options,
                index=default_index,
                format_func=lambda p: {
                    PROVIDER_OPENAI: "OpenAI (gpt-4o-mini)",
                    PROVIDER_GROQ: "Groq (llama-3.3-70b)",
                }[p],
                help="If the selected provider's API key is missing, the app "
                "automatically falls back to the other.",
            )
            for p in options:
                if p not in providers:
                    st.caption(f"{p.upper()}_API_KEY not set — fallback only.")

        st.divider()

        st.subheader("Status")
        st.write(f"Index: `{store.index_name}`")
        if st.session_state.namespace:
            st.write(f"CV: **{st.session_state.cv_filename}**")
            st.write(f"Namespace: `{st.session_state.namespace}`")
            st.write(f"Chunks: {st.session_state.num_chunks}")
        else:
            st.info("Upload a CV to start.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        if st.button(
            "Reconnect to Pinecone",
            use_container_width=True,
            help="Use this if you deleted or recreated the Pinecone index while "
            "the app was running.",
        ):
            get_store.clear()
            st.session_state.namespace = None
            st.session_state.cv_filename = None
            st.session_state.num_chunks = 0
            st.session_state.uploaded_hash = None
            st.session_state.messages = []
            st.rerun()

        return provider


def _render_chunks(chunks: List[RetrievedChunk]) -> None:
    if not chunks:
        st.caption("No context retrieved.")
        return
    for i, c in enumerate(chunks, start=1):
        st.markdown(
            f"**[{i}]** `{c.id}` — score `{c.score:.3f}` "
            f"(chunk #{c.chunk_index} of {c.source})"
        )
        st.text(c.content)
        if i < len(chunks):
            st.markdown("---")


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            chunks = msg.get("chunks")
            if chunks:
                with st.expander(f"Retrieved context ({len(chunks)} chunks)"):
                    _render_chunks(chunks)


def _answer(store: PineconeStore, provider: str, question: str) -> None:
    namespace = st.session_state.namespace
    if not namespace:
        st.warning("Upload a CV before asking questions.")
        return

    prior = st.session_state.messages[:-1]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching CV..."):
                search_q = build_search_query(question, prior)
                chunks = store.search(namespace, search_q, top_k=5, rerank=True)
        except PineconeUnauthorized:
            st.error(
                "Pinecone returned 401 Unauthorized. The index may have been "
                "deleted. Click **Reconnect to Pinecone** in the sidebar."
            )
            return

        user_prompt = build_user_prompt(question, chunks, prior_messages=prior)

        try:
            with st.spinner("Generating answer..."):
                response = generate(SYSTEM_PROMPT, user_prompt, provider=provider)
        except LLMError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"LLM call failed: {exc}")
            return

        if response.fallback_used and response.fallback_reason:
            st.warning(response.fallback_reason)

        st.markdown(response.text)
        st.caption(f"Answered with {response.provider} ({response.model})")
        with st.expander(f"Retrieved context ({len(chunks)} chunks)"):
            _render_chunks(chunks)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.text, "chunks": chunks}
        )


def main() -> None:
    _init_state()

    st.title("CV RAG Chatbot")
    st.caption(
        "Retrieval-Augmented Generation over a single CV — "
        "Streamlit + Pinecone (llama-text-embed-v2) + OpenAI/Groq."
    )

    if not os.getenv("PINECONE_API_KEY"):
        st.error("PINECONE_API_KEY is missing in .env. Add it and reload.")
        st.stop()

    try:
        store = get_store()
    except Exception as exc:
        st.error(f"Failed to initialize Pinecone: {exc}")
        st.stop()

    provider = _render_sidebar(store)

    _render_history()

    question = st.chat_input(
        "Ask a question about the CV..."
        if st.session_state.namespace
        else "Upload a CV in the sidebar first..."
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        _answer(store, provider, question)


if __name__ == "__main__":
    main()
