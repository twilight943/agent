from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeRecommendationService:
    def recommend(self, user_id, query, top_k):
        return [{"item_id": 7, "title": "demo", "price": 99.0, "score": 0.8}]


class FakeAgent:
    def run(self, user_id, query, top_k):
        return {
            "response": "推荐结果",
            "category": None,
            "max_price": 100.0,
            "preferences": [],
            "negative_preferences": [],
            "final_items": [{"item_id": 7, "title": "demo", "price": 99.0, "score": 0.8, "reason": "预算内"}],
        }


class FakeContainer:
    item_service = None
    recommendation_service = FakeRecommendationService()
    agent = FakeAgent()

    def health(self):
        return {
            "status": "ok",
            "environment": "test",
            "resources": {"items": True, "recommendation": True, "agent": True, "llm": False},
            "details": [],
        }


def test_health_recommend_and_chat() -> None:
    with TestClient(create_app(container=FakeContainer())) as client:
        assert client.get("/health").json()["status"] == "ok"
        recommendation = client.post("/recommend", json={"user_id": "u1", "query": "demo", "top_k": 1})
        assert recommendation.status_code == 200
        assert recommendation.json()["items"][0]["item_id"] == 7
        chat = client.post("/chat", json={"user_id": "u1", "message": "100元以下", "top_k": 1})
        assert chat.status_code == 200
        assert chat.json()["parsed_intent"]["max_price"] == 100.0
