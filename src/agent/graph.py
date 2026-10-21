from __future__ import annotations

from typing import Any

from src.agent.nodes.explain import explain_node
from src.agent.nodes.filter_constraints import filter_constraints_node
from src.agent.nodes.parse_intent import parse_intent_node
from src.agent.nodes.rank import rank_node
from src.agent.nodes.retrieve import retrieve_node
from src.agent.state import AgentState
from src.services.llm_client import LLMClient
from src.services.recommendation_service import RecommendationService


class RecommendationAgent:
    def __init__(
        self,
        recommendation_service: RecommendationService,
        llm: LLMClient | None,
        config: dict[str, Any],
    ) -> None:
        self.service = recommendation_service
        self.llm = llm
        self.config = config
        self.compiled_graph = self._build_langgraph() if config.get("use_langgraph", True) else None

    def _parse(self, state: AgentState) -> dict[str, object]:
        return parse_intent_node(state, self.llm)

    def _retrieve(self, state: AgentState) -> dict[str, object]:
        return retrieve_node(state, self.service, int(self.config.get("retrieval_top_k", 300)))

    def _rank(self, state: AgentState) -> dict[str, object]:
        return rank_node(state, self.service)

    def _filter(self, state: AgentState) -> dict[str, object]:
        return filter_constraints_node(
            state,
            final_top_k=int(state.get("top_k", self.config.get("final_top_k", 10))),
            strict_category=bool(self.config.get("strict_category", True)),
            strict_max_price=bool(self.config.get("strict_max_price", True)),
        )

    def _explain(self, state: AgentState) -> dict[str, object]:
        return explain_node(state, self.llm)

    def _build_langgraph(self) -> Any | None:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None
        graph = StateGraph(AgentState)
        graph.add_node("parse_intent", self._parse)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rank", self._rank)
        graph.add_node("filter_constraints", self._filter)
        graph.add_node("explain", self._explain)
        graph.add_edge(START, "parse_intent")
        graph.add_edge("parse_intent", "retrieve")
        graph.add_edge("retrieve", "rank")
        graph.add_edge("rank", "filter_constraints")
        graph.add_edge("filter_constraints", "explain")
        graph.add_edge("explain", END)
        return graph.compile()

    def run(self, user_id: str, query: str, top_k: int | None = None) -> AgentState:
        initial: AgentState = {
            "user_id": user_id,
            "query": query,
            "top_k": int(top_k or self.config.get("final_top_k", 10)),
        }
        if self.compiled_graph is not None:
            return self.compiled_graph.invoke(initial)
        state = dict(initial)
        for node in (self._parse, self._retrieve, self._rank, self._filter, self._explain):
            state.update(node(state))
        return state  # type: ignore[return-value]
