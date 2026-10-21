from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_id: str
    internal_user_id: int | None
    query: str
    top_k: int
    category: str | None
    max_price: float | None
    preferences: list[str]
    negative_preferences: list[str]
    candidates: list[dict[str, Any]]
    ranked_items: list[dict[str, Any]]
    final_items: list[dict[str, Any]]
    response: str
