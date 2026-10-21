from src.agent.graph import RecommendationAgent
from src.agent.nodes.filter_constraints import filter_constraints_node
from src.services.llm_client import LLMClient


class FakeRecommendationService:
    def retrieve(self, user_id, query, top_k):
        return 1, [
            {"item_id": 1, "retrieval_score": 0.9, "title": "expensive", "price": 1200.0},
            {"item_id": 2, "retrieval_score": 0.8, "title": "within budget", "price": 699.0},
        ]

    def rank_candidates(self, user_id, candidates):
        return candidates

    def enrich(self, candidates):
        return candidates


def test_hard_price_constraint_is_code_enforced() -> None:
    result = filter_constraints_node(
        {"max_price": 1000.0, "ranked_items": [{"item_id": 1, "price": 1001}, {"item_id": 2, "price": 999}]},
        final_top_k=10,
    )
    assert [item["item_id"] for item in result["final_items"]] == [2]


def test_sequential_agent_completes_without_llm() -> None:
    llm = LLMClient({"enabled": False})
    agent = RecommendationAgent(
        FakeRecommendationService(),
        llm,
        {"use_langgraph": False, "retrieval_top_k": 20, "final_top_k": 5},
    )
    state = agent.run("external-user", "1000元以下，适合通勤", top_k=5)
    assert state["max_price"] == 1000.0
    assert [item["item_id"] for item in state["final_items"]] == [2]
    assert "within budget" in state["response"]
