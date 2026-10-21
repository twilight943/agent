from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.agent.prompts import EXPLANATION_SYSTEM_PROMPT
from src.agent.state import AgentState
from src.common.logger import get_logger
from src.services.llm_client import LLMClient


LOGGER = get_logger(__name__)


class ItemReason(BaseModel):
    item_id: int
    reason: str


class Explanation(BaseModel):
    reasons: list[ItemReason] = Field(default_factory=list)


def _default_reason(item: dict[str, Any], state: AgentState) -> str:
    reasons = []
    if state.get("category") and item.get("category"):
        reasons.append(f"类别符合“{state['category']}”")
    if state.get("max_price") is not None and item.get("price") is not None:
        reasons.append(f"价格 {item['price']} 未超过预算")
    return "，".join(reasons) or "综合召回与排序得分较高"


def explain_node(state: AgentState, llm: LLMClient | None) -> dict[str, object]:
    items = [dict(item) for item in state.get("final_items", [])]
    reasons = {int(item["item_id"]): _default_reason(item, state) for item in items}
    if items and llm is not None and llm.configured:
        safe_items = [
            {key: item.get(key) for key in ("item_id", "title", "category", "price", "description")}
            for item in items
        ]
        try:
            result = llm.structured_output(
                [
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"query": state.get("query", ""), "items": safe_items},
                            ensure_ascii=False,
                        ),
                    },
                ],
                Explanation,
            )
            allowed = set(reasons)
            for value in result.reasons:
                if value.item_id in allowed and value.reason.strip():
                    reasons[value.item_id] = value.reason.strip()
        except Exception as error:
            LOGGER.warning("Explanation LLM failed; using deterministic reasons: %s", error)
    for item in items:
        item["reason"] = reasons[int(item["item_id"])]
    if items:
        response = "\n".join(
            f"{index}. {item.get('title') or '商品 ' + str(item['item_id'])}：{item['reason']}"
            for index, item in enumerate(items, start=1)
        )
    else:
        response = "没有找到满足当前条件的商品，请尝试放宽类别或价格限制。"
    return {"final_items": items, "response": response}
