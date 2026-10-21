from __future__ import annotations

from typing import Any

from src.agent.state import AgentState


def _matches_category(actual: Any, requested: str) -> bool:
    if actual is None:
        return False
    actual_text, requested_text = str(actual).casefold().strip(), requested.casefold().strip()
    return actual_text == requested_text or requested_text in actual_text


def filter_constraints_node(
    state: AgentState,
    final_top_k: int,
    strict_category: bool = True,
    strict_max_price: bool = True,
) -> dict[str, object]:
    category, max_price = state.get("category"), state.get("max_price")
    filtered: list[dict[str, Any]] = []
    for item in state.get("ranked_items", []):
        if category and strict_category and not _matches_category(item.get("category"), category):
            continue
        if max_price is not None and strict_max_price:
            price = item.get("price")
            if price is None or float(price) > float(max_price):
                continue
        filtered.append(item)
        if len(filtered) >= final_top_k:
            break
    return {"final_items": filtered}
