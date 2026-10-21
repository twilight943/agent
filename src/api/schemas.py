from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str
    query: str = ""
    top_k: int = Field(default=10, ge=1, le=100)


class ChatRequest(BaseModel):
    user_id: str
    message: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendationItem(BaseModel):
    item_id: int
    original_item_id: str | int | None = None
    title: str = ""
    category: str | None = None
    price: float | None = None
    description: str | None = None
    score: float = 0.0
    reason: str = ""


class RecommendationResponse(BaseModel):
    user_id: str
    items: list[RecommendationItem]


class ChatResponse(BaseModel):
    user_id: str
    response: str
    items: list[RecommendationItem]
    parsed_intent: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    environment: str
    resources: dict[str, bool]
    details: list[str] = Field(default_factory=list)
