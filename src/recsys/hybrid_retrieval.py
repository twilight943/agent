from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _normalize(results: list[dict[str, Any]]) -> dict[int, float]:
    if not results:
        return {}
    scores = [float(result["retrieval_score"]) for result in results]
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return {int(result["item_id"]): 1.0 for result in results}
    return {
        int(result["item_id"]): (float(result["retrieval_score"]) - low) / (high - low)
        for result in results
    }


class HybridRetriever:
    def __init__(
        self,
        collaborative: Any | None = None,
        semantic: Any | None = None,
        collaborative_weight: float = 0.65,
        semantic_weight: float = 0.35,
    ) -> None:
        self.collaborative = collaborative
        self.semantic = semantic
        total = float(collaborative_weight) + float(semantic_weight)
        if total <= 0:
            raise ValueError("At least one retrieval weight must be positive")
        self.collaborative_weight = float(collaborative_weight) / total
        self.semantic_weight = float(semantic_weight) / total

    def hybrid_retrieve(self, user_id: int | None, query: str, top_k: int = 300) -> list[dict[str, float | int]]:
        collaborative_results = (
            self.collaborative.retrieve(user_id, top_k) if self.collaborative is not None and user_id else []
        )
        semantic_results = (
            self.semantic.semantic_retrieve(query, top_k) if self.semantic is not None and query.strip() else []
        )
        collaborative_scores = _normalize(collaborative_results)
        semantic_scores = _normalize(semantic_results)
        item_ids = collaborative_scores.keys() | semantic_scores.keys()
        combined = [
            {
                "item_id": item_id,
                "retrieval_score": (
                    self.collaborative_weight * collaborative_scores.get(item_id, 0.0)
                    + self.semantic_weight * semantic_scores.get(item_id, 0.0)
                ),
                "collaborative_score": collaborative_scores.get(item_id, 0.0),
                "semantic_score": semantic_scores.get(item_id, 0.0),
            }
            for item_id in item_ids
        ]
        return sorted(combined, key=lambda value: float(value["retrieval_score"]), reverse=True)[:top_k]
