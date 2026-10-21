from src.services.recommendation_service import RecommendationService


def recommendation_tool(
    service: RecommendationService, user_id: str, query: str, top_k: int = 300
) -> list[dict[str, object]]:
    _, candidates = service.retrieve(user_id, query, top_k)
    return candidates
