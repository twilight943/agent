from __future__ import annotations

from src.agent.state import AgentState
from src.services.recommendation_service import RecommendationService


def rank_node(state: AgentState, recommendation_service: RecommendationService) -> dict[str, object]:
    ranked = recommendation_service.rank_candidates(
        state.get("internal_user_id"), state.get("candidates", [])
    )
    return {"ranked_items": recommendation_service.enrich(ranked)}
