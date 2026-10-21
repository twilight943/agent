from __future__ import annotations

from src.agent.state import AgentState
from src.services.recommendation_service import RecommendationService


def retrieve_node(
    state: AgentState,
    recommendation_service: RecommendationService,
    top_k: int,
) -> dict[str, object]:
    internal_user_id, candidates = recommendation_service.retrieve(
        state.get("user_id"), state.get("query", ""), top_k
    )
    return {"internal_user_id": internal_user_id, "candidates": candidates}
