from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.agent.prompts import INTENT_SYSTEM_PROMPT
from src.agent.state import AgentState
from src.common.logger import get_logger
from src.services.llm_client import LLMClient


LOGGER = get_logger(__name__)


class ParsedIntent(BaseModel):
    category: str | None = None
    max_price: float | None = None
    preferences: list[str] = Field(default_factory=list)
    negative_preferences: list[str] = Field(default_factory=list)


def _fallback_parse(query: str) -> ParsedIntent:
    price_patterns = [
        r"(?:[¥￥]\s*)?(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下|之内|不超过)",
        r"(?:预算|最高价|max(?:imum)? price)\D{0,8}(\d+(?:\.\d+)?)",
        r"(?:under|below)\s*[¥￥$]?\s*(\d+(?:\.\d+)?)",
    ]
    max_price = None
    for pattern in price_patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            max_price = float(match.group(1))
            break
    return ParsedIntent(max_price=max_price, preferences=[query] if query.strip() else [])


def parse_intent_node(state: AgentState, llm: LLMClient | None) -> dict[str, object]:
    query = state.get("query", "")
    parsed = _fallback_parse(query)
    if llm is not None and llm.configured:
        try:
            parsed = llm.structured_output(
                [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                ParsedIntent,
            )
        except Exception as error:
            LOGGER.warning("Intent LLM failed; using deterministic parser: %s", error)
    return {
        "category": parsed.category,
        "max_price": parsed.max_price,
        "preferences": parsed.preferences,
        "negative_preferences": parsed.negative_preferences,
    }
