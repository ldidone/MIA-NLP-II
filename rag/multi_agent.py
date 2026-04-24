"""Multi-agent orchestration for CV-specific RAG answering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .llm import LLMResponse, generate
from .pinecone_store import PineconeStore, RetrievedChunk
from .prompts import SYSTEM_PROMPT, build_search_query, build_user_prompt


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
        all_aliases: List[str] = []
        for alias in [clean_name, *provided_aliases]:
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
        text = query.lower()
        matched: List[AgentSpec] = []

        for agent in self._agents_by_id.values():
            for alias in agent.aliases:
                escaped = re.escape(alias.lower())
                pattern = rf"\b{escaped}\b"
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
        if matched:
            if len(matched) == 1:
                reason = f"Detected person mention: {matched[0].display_name}"
            else:
                names = ", ".join(a.display_name for a in matched)
                reason = f"Detected multiple people: {names}"
            return RouteDecision(selected_agents=matched, used_default=False, reason=reason)

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


class CVAgent:
    """Retrieval and answer generation for a single CV/person."""

    def __init__(self, store: PineconeStore, spec: AgentSpec) -> None:
        self.store = store
        self.spec = spec

    def retrieve(
        self,
        question: str,
        prior_messages: Optional[List[dict]] = None,
        top_k: int = 5,
    ) -> AgentResult:
        search_q = build_search_query(question, prior_messages or [])
        chunks = self.store.search(
            namespace=self.spec.namespace,
            question=search_q,
            top_k=top_k,
            rerank=True,
        )
        return AgentResult(agent=self.spec, search_query=search_q, chunks=chunks)

    def answer(
        self,
        question: str,
        provider: str,
        prior_messages: Optional[List[dict]] = None,
        top_k: int = 5,
    ) -> OrchestrationResult:
        retrieval = self.retrieve(question, prior_messages=prior_messages, top_k=top_k)
        scoped_question = f"Question about {self.spec.display_name}: {question}"
        user_prompt = build_user_prompt(
            scoped_question,
            retrieval.chunks,
            prior_messages=prior_messages or [],
        )
        response = generate(SYSTEM_PROMPT, user_prompt, provider=provider)
        route = RouteDecision(
            selected_agents=[self.spec],
            used_default=False,
            reason=f"Answered with single CV agent ({self.spec.display_name}).",
        )
        return OrchestrationResult(
            response=response,
            route=route,
            agent_results=[retrieval],
        )


class ResponseSynthesizer:
    """Combine multi-agent retrieval context into one grounded answer."""

    _SYSTEM = (
        "You are a CV comparison assistant. Use ONLY the retrieved CV context from "
        "the listed people. Do not invent facts. If context is insufficient for any "
        "part of the question, explicitly say the CVs do not contain that information. "
        "For comparisons, mention each person separately before concluding. "
        "Answer in the same language as the user's question."
    )

    @staticmethod
    def _format_multi_context(agent_results: List[AgentResult]) -> str:
        if not agent_results:
            return "(no context retrieved)"

        sections: List[str] = []
        for result in agent_results:
            person = result.agent.display_name
            if not result.chunks:
                sections.append(
                    f"### {person}\n(no context retrieved for this CV)"
                )
                continue

            chunk_lines: List[str] = []
            for idx, chunk in enumerate(result.chunks, start=1):
                chunk_lines.append(
                    f"[{person}#{idx}] (source: {chunk.source}, chunk #{chunk.chunk_index}, "
                    f"score={chunk.score:.3f})\n{chunk.content}"
                )
            sections.append(f"### {person}\n\n" + "\n\n".join(chunk_lines))
        return "\n\n".join(sections)

    def synthesize(
        self,
        question: str,
        agent_results: List[AgentResult],
        provider: str,
        prior_messages: Optional[List[dict]] = None,
    ) -> LLMResponse:
        history_block = ""
        if prior_messages:
            recent = []
            for message in prior_messages[-8:]:
                role = message.get("role")
                content = (message.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    label = "User" if role == "user" else "Assistant"
                    recent.append(f"{label}: {content}")
            if recent:
                history_block = "=== PRIOR MESSAGES ===\n" + "\n\n".join(recent) + "\n\n"

        context = self._format_multi_context(agent_results)
        user_prompt = (
            "Use the multi-person CV context to answer the question.\n\n"
            f"{history_block}"
            "=== MULTI-CV CONTEXT ===\n"
            f"{context}\n"
            "=== END CONTEXT ===\n\n"
            f"Question: {question}\n\n"
            "Rules:\n"
            "- Ground all claims in the context.\n"
            "- If missing, say the CV(s) do not contain that information.\n"
            "- For comparisons, make the basis of comparison explicit.\n"
        )
        return generate(self._SYSTEM, user_prompt, provider=provider)


class MultiAgentOrchestrator:
    """Router + per-agent retrieval + final synthesis pipeline."""

    def __init__(
        self,
        store: PineconeStore,
        registry: AgentRegistry,
        synthesizer: Optional[ResponseSynthesizer] = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.router = CVRouter(registry)
        self.synthesizer = synthesizer or ResponseSynthesizer()

    def answer(
        self,
        question: str,
        provider: str,
        prior_messages: Optional[List[dict]] = None,
    ) -> OrchestrationResult:
        route = self.router.route(question)
        if not route.selected_agents:
            empty_response = LLMResponse(
                text="No CV agents are registered yet. Upload at least one CV to continue.",
                provider=provider,
                model="n/a",
            )
            return OrchestrationResult(
                response=empty_response,
                route=route,
                agent_results=[],
            )

        if len(route.selected_agents) == 1:
            single_agent = CVAgent(self.store, route.selected_agents[0])
            result = single_agent.answer(
                question=question,
                provider=provider,
                prior_messages=prior_messages,
            )
            return OrchestrationResult(
                response=result.response,
                route=route,
                agent_results=result.agent_results,
            )

        agent_results: List[AgentResult] = []
        for spec in route.selected_agents:
            agent = CVAgent(self.store, spec)
            agent_results.append(agent.retrieve(question, prior_messages=prior_messages))

        final_response = self.synthesizer.synthesize(
            question=question,
            agent_results=agent_results,
            provider=provider,
            prior_messages=prior_messages,
        )
        return OrchestrationResult(
            response=final_response,
            route=route,
            agent_results=agent_results,
        )
