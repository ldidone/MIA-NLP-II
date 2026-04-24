"""Streamlit chat UI for CV question answering with multi-agent RAG."""

from __future__ import annotations

import os
import re
from typing import List

import streamlit as st
from dotenv import load_dotenv

from rag.chunker import chunk_text
from rag.llm import (
    PROVIDER_GROQ,
    PROVIDER_OPENAI,
    available_providers,
)
from rag.loader import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError, extract_text
from rag.multi_agent import AgentRegistry, run_graph
from rag.pinecone_store import (
    PineconeStore,
    PineconeUnauthorized,
    RetrievedChunk,
    namespace_for,
)


load_dotenv()

st.set_page_config(page_title="CV RAG Chatbot", page_icon=None, layout="wide")


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_store() -> PineconeStore:
    return PineconeStore()


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("agent_registry", AgentRegistry())


def _person_prefix(person_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", person_name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_") or "person"
    return f"cv_{slug}"


def _ingest_cv(
    store: PineconeStore,
    registry: AgentRegistry,
    file_bytes: bytes,
    filename: str,
    person_name: str,
    aliases_csv: str,
    set_as_default: bool,
) -> None:
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

    namespace = namespace_for(file_bytes, prefix=_person_prefix(person_name))

    if store.namespace_exists(namespace):
        st.info(
            f"{person_name}'s CV is already indexed (namespace `{namespace}`). "
            "Skipping re-upload and refreshing the agent mapping."
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

    aliases = AgentRegistry.aliases_from_csv(aliases_csv)
    spec = registry.register(
        person_name=person_name,
        namespace=namespace,
        source_filename=filename,
        aliases=aliases,
        set_as_default=set_as_default,
    )
    st.session_state.messages = []
    st.success(
        f"CV ready for **{spec.display_name}**. "
        f"Namespace: `{spec.namespace}`. Ask questions in chat."
    )


def _render_sidebar(store: PineconeStore) -> str:
    registry: AgentRegistry = st.session_state.agent_registry

    with st.sidebar:
        st.header("Multi-CV RAG")
        st.caption(
            "Upload one CV per person. Queries are routed to the right CV agent."
        )

        person_name = st.text_input(
            "Person name",
            placeholder="e.g., Juan Perez",
            help="Name used by the router to detect mentions in questions.",
        )
        aliases_csv = st.text_input(
            "Aliases (optional, comma-separated)",
            placeholder="e.g., Juan, Juancito",
        )
        set_as_default = st.checkbox(
            "Set as default (student CV)",
            value=(registry.default_agent() is None),
            help=(
                "When a question mentions no person, the default CV agent is used."
            ),
        )

        uploaded = st.file_uploader(
            "Upload CV file",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=False,
        )
        if st.button("Index / update CV", use_container_width=True):
            if uploaded is None:
                st.warning("Upload a CV file first.")
            elif not person_name.strip():
                st.warning("Provide the person's name before indexing.")
            else:
                _ingest_cv(
                    store=store,
                    registry=registry,
                    file_bytes=uploaded.getvalue(),
                    filename=uploaded.name,
                    person_name=person_name.strip(),
                    aliases_csv=aliases_csv,
                    set_as_default=set_as_default,
                )

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
        agents = registry.list_agents()
        if not agents:
            st.info("Upload at least one CV to start.")
        else:
            st.write(f"Registered agents: **{len(agents)}**")
            default_id = registry.default_agent_id()
            options = {a.display_name: a.person_id for a in agents}
            name_list = list(options.keys())
            default_index = 0
            if default_id:
                for idx, name in enumerate(name_list):
                    if options[name] == default_id:
                        default_index = idx
                        break
            selected_name = st.selectbox(
                "Default agent",
                options=name_list,
                index=default_index,
            )
            selected_id = options[selected_name]
            if selected_id != default_id:
                registry.set_default_agent(selected_id)
                st.rerun()

            for agent in agents:
                is_default = agent.person_id == registry.default_agent_id()
                label = f"**{agent.display_name}**"
                if is_default:
                    label += " (default)"
                st.markdown(label)
                st.caption(
                    f"Namespace: `{agent.namespace}` | Source: `{agent.source_filename}`"
                )

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
            agent_results = msg.get("agent_results") or []
            for result in agent_results:
                agent_name = result["agent_name"]
                chunks = result["chunks"]
                with st.expander(
                    f"{agent_name} context ({len(chunks)} chunks)"
                ):
                    _render_chunks(chunks)


def _answer(store: PineconeStore, provider: str, question: str) -> None:
    registry: AgentRegistry = st.session_state.agent_registry
    if not registry.list_agents():
        st.warning("Upload at least one CV before asking questions.")
        return

    prior = st.session_state.messages[:-1]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Routing and retrieving context..."):
                result = run_graph(
                    question=question,
                    provider=provider,
                    prior_messages=prior,
                    store=store,
                    registry=registry,
                )
        except PineconeUnauthorized:
            st.error(
                "Pinecone returned 401 Unauthorized. The index may have been "
                "deleted. Click **Reconnect to Pinecone** in the sidebar."
            )
            return

        response = result.response
        if response.model == "n/a":
            st.warning(response.text)
            return

        if response.fallback_used and response.fallback_reason:
            st.warning(response.fallback_reason)

        route = result.route
        routed_names = ", ".join(a.display_name for a in route.selected_agents)
        if routed_names:
            mode_label = "default route" if route.used_default else "explicit route"
            st.caption(f"Route ({mode_label}): {routed_names}")
            st.caption(route.reason)

        st.markdown(response.text)
        st.caption(f"Answered with {response.provider} ({response.model})")

        serialized_agent_results = []
        for agent_result in result.agent_results:
            with st.expander(
                f"{agent_result.agent.display_name} context ({len(agent_result.chunks)} chunks)"
            ):
                _render_chunks(agent_result.chunks)
            serialized_agent_results.append(
                {
                    "agent_name": agent_result.agent.display_name,
                    "chunks": agent_result.chunks,
                }
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "agent_results": serialized_agent_results,
            }
        )


def main() -> None:
    _init_state()

    st.title("Multi-Agent CV RAG Chatbot")
    st.caption(
        "One CV agent per person, routed by query mentions, using "
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
        "Ask about one person or compare multiple people..."
        if st.session_state.agent_registry.list_agents()
        else "Upload at least one CV in the sidebar first..."
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        _answer(store, provider, question)


if __name__ == "__main__":
    main()
