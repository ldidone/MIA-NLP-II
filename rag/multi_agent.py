"""Multi-agent orchestration for CV-specific RAG answering using LangGraph."""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from .llm import LLMResponse, generate
from .pinecone_store import PineconeStore, RetrievedChunk
from .prompts import build_search_query, build_user_prompt


# ---------------------------------------------------------------------------
# Domain data classes (unchanged)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip combining accents for robust name matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "person"


def _split_aliases(raw_aliases: str) -> List[str]:
    return [part.strip() for part in raw_aliases.split(",") if part.strip()]


@dataclass(frozen=True)
class AgentSpec:
    """Configuration for one CV agent."""

    person_id: str
    display_name: str
    aliases: List[str]
    namespace: str
    source_filename: str
    metadata_filter: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    """Router result containing the selected agents."""

    selected_agents: List[AgentSpec]
    used_default: bool
    reason: str


@dataclass(frozen=True)
class AgentResult:
    """Retrieval output for one selected CV agent."""

    agent: AgentSpec
    search_query: str
    chunks: List[RetrievedChunk]


@dataclass(frozen=True)
class OrchestrationResult:
    """Final response and retrieval traces."""

    response: LLMResponse
    route: RouteDecision
    agent_results: List[AgentResult]


# ---------------------------------------------------------------------------
# AgentRegistry & CVRouter (unchanged domain logic)
# ---------------------------------------------------------------------------

class AgentRegistry:
    """In-memory registry mapping people/aliases to their CV agents."""

    def __init__(self) -> None:
        self._agents_by_id: Dict[str, AgentSpec] = {}
        self._default_agent_id: Optional[str] = None

    def register(
        self,
        person_name: str,
        namespace: str,
        source_filename: str,
        aliases: Optional[Sequence[str]] = None,
        set_as_default: bool = False,
    ) -> AgentSpec:
        clean_name = person_name.strip()
        if not clean_name:
            raise ValueError("person_name cannot be empty.")

        provided_aliases = [a.strip() for a in (aliases or []) if a and a.strip()]

        # Auto-expand a multi-word name into first-name and last-name aliases
        # so "Lucas" matches "Lucas Didone" without explicit alias configuration.
        name_parts = clean_name.split()
        auto_parts: List[str] = []
        if len(name_parts) > 1:
            auto_parts.append(name_parts[0])   # first name
            auto_parts.append(name_parts[-1])  # last name

        all_aliases: List[str] = []
        for alias in [clean_name, *auto_parts, *provided_aliases]:
            if alias.lower() not in {x.lower() for x in all_aliases}:
                all_aliases.append(alias)

        person_id = _slugify(clean_name)
        spec = AgentSpec(
            person_id=person_id,
            display_name=clean_name,
            aliases=all_aliases,
            namespace=namespace,
            source_filename=source_filename,
            metadata_filter={"person_id": person_id},
        )
        self._agents_by_id[person_id] = spec

        if set_as_default or self._default_agent_id is None:
            self._default_agent_id = person_id

        return spec

    def list_agents(self) -> List[AgentSpec]:
        return sorted(self._agents_by_id.values(), key=lambda a: a.display_name.lower())

    def default_agent(self) -> Optional[AgentSpec]:
        if not self._default_agent_id:
            return None
        return self._agents_by_id.get(self._default_agent_id)

    def default_agent_id(self) -> Optional[str]:
        return self._default_agent_id

    def set_default_agent(self, person_id: str) -> None:
        if person_id not in self._agents_by_id:
            raise ValueError(f"Unknown person_id: {person_id}")
        self._default_agent_id = person_id

    def find_agents_in_query(self, query: str) -> List[AgentSpec]:
        text = _normalize(query)
        matched: List[AgentSpec] = []

        for agent in self._agents_by_id.values():
            for alias in agent.aliases:
                norm_alias = _normalize(alias)
                if len(norm_alias) < 2:
                    continue
                escaped = re.escape(norm_alias)
                pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
                if re.search(pattern, text):
                    matched.append(agent)
                    break

        matched.sort(key=lambda a: a.display_name.lower())
        return matched

    @classmethod
    def aliases_from_csv(cls, raw_aliases: str) -> List[str]:
        return _split_aliases(raw_aliases)


class CVRouter:
    """Select one or more CV agents based on the question text."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def route(self, query: str) -> RouteDecision:
        matched = self.registry.find_agents_in_query(query)

        if len(matched) == 1:
            return RouteDecision(
                selected_agents=matched,
                used_default=False,
                reason=f"Detected person mention: {matched[0].display_name}",
            )

        if len(matched) > 1:
            names = ", ".join(a.display_name for a in matched)
            return RouteDecision(
                selected_agents=matched,
                used_default=False,
                reason=f"Detected multiple people: {names}",
            )

        # No name recognised — fall back to the default agent.
        default_agent = self.registry.default_agent()
        if default_agent is None:
            return RouteDecision(
                selected_agents=[],
                used_default=True,
                reason="No registered CV agents are available.",
            )
        return RouteDecision(
            selected_agents=[default_agent],
            used_default=True,
            reason=f"No person mention found; using default CV agent ({default_agent.display_name}).",
        )


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    question: str
    provider: str
    prior_messages: List[dict]
    store: Any
    registry: Any
    selected_agents: List[AgentSpec]
    route_decision: Optional[RouteDecision]
    agent_results: Annotated[List[AgentResult], operator.add]
    response: Optional[LLMResponse]


class RetrieveInput(TypedDict):
    """Per-branch state sent to retrieve_cv via Send.

    Only carries the fields retrieve_cv actually needs so the fan-out
    branches don't conflict on the reducer-managed agent_results channel.
    """
    question: str
    prior_messages: List[dict]
    store: Any
    agent_spec: AgentSpec


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------



def route_query(state: GraphState) -> dict:
    """Node: detect which person(s) are mentioned and decide routing."""
    registry: AgentRegistry = state["registry"]
    router = CVRouter(registry)
    decision = router.route(state["question"])
    return {
        "route_decision": decision,
        "selected_agents": decision.selected_agents,
    }


def no_agents_response(state: GraphState) -> dict:
    """Node: early-exit when no CV agents are registered."""
    return {
        "response": LLMResponse(
            text="No CV agents are registered yet. Upload at least one CV to continue.",
            provider=state["provider"],
            model="n/a",
        ),
    }


def retrieve_cv(state: RetrieveInput) -> dict:
    """Node: retrieve top-k chunks from one CV agent's namespace."""
    spec: AgentSpec = state["agent_spec"]
    store: PineconeStore = state["store"]
    question = state["question"]
    prior = state["prior_messages"]

    search_q = build_search_query(question, prior or [])
    try:
        chunks = store.search(
            namespace=spec.namespace,
            question=search_q,
            top_k=5,
            rerank=True,
        )
    except Exception:
        chunks = []
    result = AgentResult(agent=spec, search_query=search_q, chunks=chunks)
    return {"agent_results": [result]}


def collect_results(state: GraphState) -> dict:
    """Fan-in node: no-op that waits for all retrieve_cv branches to finish."""
    return {}


def _filter_prior_for_agent(
    prior_messages: List[dict], agent_name: str
) -> List[dict]:
    """Return only message pairs whose assistant turn used `agent_name`.

    Each (user, assistant) pair is kept only when the assistant message's
    agent_results contain the requested agent. This prevents context from
    exchanges about other candidates leaking into the current answer.
    """
    if not prior_messages:
        return []
    result: List[dict] = []
    i = 0
    while i < len(prior_messages):
        msg = prior_messages[i]
        if msg.get("role") == "user" and i + 1 < len(prior_messages):
            nxt = prior_messages[i + 1]
            if nxt.get("role") == "assistant":
                used = {r.get("agent_name") for r in (nxt.get("agent_results") or [])}
                if agent_name in used:
                    result.append(msg)
                    result.append(nxt)
                i += 2
                continue
        i += 1
    return result


_SYNTHESIZER_SYSTEM = (
    "You are a CV comparison assistant. Use ONLY the retrieved CV context from "
    "the listed people. Do not invent facts. If context is insufficient for any "
    "part of the question, explicitly say the CV does not contain that information. "
    "For multi-person answers, address each person separately before concluding. "
    "Answer in the same language as the user's question."
)


def generate_response(state: GraphState) -> dict:
    """Node: produce the final LLM answer from retrieved context."""
    question = state["question"]
    provider = state["provider"]
    prior = state["prior_messages"] or []
    agent_results: List[AgentResult] = state.get("agent_results") or []
    selected = state.get("selected_agents") or []

    if len(selected) == 1:
        spec = selected[0]
        name = spec.display_name
        chunks = agent_results[0].chunks if agent_results else []

        # Keep only history turns that were about this specific candidate.
        filtered_prior = _filter_prior_for_agent(prior, name)

        system_prompt = (
            f"You are a helpful assistant answering questions exclusively about "
            f"{name}'s CV. Ground every fact in the provided CV context. "
            "Do not use or reference information about any other person. "
            "If the answer is not in the CV context, say the CV does not contain it. "
            "Be concise, factual, and cite context numbers like [1], [2] when useful. "
            "Answer in the same language as the question."
        )
        user_prompt = build_user_prompt(question, chunks, prior_messages=filtered_prior)
        resp = generate(system_prompt, user_prompt, provider=provider)

    else:
        # Multi-person: fan-in all retrieved contexts and synthesise together.
        context = _format_multi_context(agent_results)
        history_block = _format_history_block(prior)
        user_prompt = (
            "Use the multi-person CV context below to answer the question.\n\n"
            f"{history_block}"
            "=== MULTI-CV CONTEXT ===\n"
            f"{context}\n"
            "=== END CONTEXT ===\n\n"
            f"Question: {question}\n\n"
            "Rules:\n"
            "- Ground all claims in the context above.\n"
            "- Address each person separately.\n"
            "- If information is missing for someone, say their CV does not contain it.\n"
        )
        resp = generate(_SYNTHESIZER_SYSTEM, user_prompt, provider=provider)

    return {"response": resp}


# ---------------------------------------------------------------------------
# Prompt helpers (extracted from the old ResponseSynthesizer)
# ---------------------------------------------------------------------------

def _format_multi_context(agent_results: List[AgentResult]) -> str:
    if not agent_results:
        return "(no context retrieved)"
    sections: List[str] = []
    for result in agent_results:
        person = result.agent.display_name
        if not result.chunks:
            sections.append(f"### {person}\n(no context retrieved for this CV)")
            continue
        chunk_lines: List[str] = []
        for idx, chunk in enumerate(result.chunks, start=1):
            chunk_lines.append(
                f"[{person}#{idx}] (source: {chunk.source}, chunk #{chunk.chunk_index}, "
                f"score={chunk.score:.3f})\n{chunk.content}"
            )
        sections.append(f"### {person}\n\n" + "\n\n".join(chunk_lines))
    return "\n\n".join(sections)


def _format_history_block(prior_messages: Optional[List[dict]]) -> str:
    if not prior_messages:
        return ""
    recent = []
    for message in prior_messages[-8:]:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Assistant"
            recent.append(f"{label}: {content}")
    if not recent:
        return ""
    return "=== PRIOR MESSAGES ===\n" + "\n\n".join(recent) + "\n\n"


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _route_after_decision(state: GraphState) -> Any:
    """Conditional edge from route_query.

    Returns "no_agents_response" when no agents matched, or a list of
    Send objects that fan-out one retrieve_cv call per selected agent.
    """
    selected = state.get("selected_agents") or []
    if not selected:
        return "no_agents_response"

    return [
        Send("retrieve_cv", {
            "question": state["question"],
            "prior_messages": state["prior_messages"],
            "store": state["store"],
            "agent_spec": spec,
        })
        for spec in selected
    ]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the multi-agent CV RAG graph."""
    builder = StateGraph(GraphState)

    builder.add_node("route_query", route_query)
    builder.add_node("no_agents_response", no_agents_response)
    builder.add_node("retrieve_cv", retrieve_cv)
    builder.add_node("collect_results", collect_results)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "route_query")

    builder.add_conditional_edges(
        "route_query",
        _route_after_decision,
        ["no_agents_response", "retrieve_cv"],
    )

    builder.add_edge("retrieve_cv", "collect_results")
    builder.add_edge("collect_results", "generate_response")
    builder.add_edge("generate_response", END)
    builder.add_edge("no_agents_response", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------

_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_graph(
    question: str,
    provider: str,
    prior_messages: List[dict],
    store: PineconeStore,
    registry: AgentRegistry,
) -> OrchestrationResult:
    """Run the LangGraph multi-agent pipeline and return an OrchestrationResult."""
    graph = _get_graph()

    initial_state: GraphState = {
        "question": question,
        "provider": provider,
        "prior_messages": prior_messages or [],
        "store": store,
        "registry": registry,
        "selected_agents": [],
        "route_decision": None,
        "agent_results": [],
        "response": None,
    }

    final_state = graph.invoke(initial_state)

    return OrchestrationResult(
        response=final_state["response"],
        route=final_state["route_decision"],
        agent_results=final_state["agent_results"],
    )
