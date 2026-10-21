from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


router = APIRouter()


def _container(request: Request) -> Any:
    return request.app.state.container


def _item(value: dict[str, Any]) -> RecommendationItem:
    return RecommendationItem(
        item_id=int(value["item_id"]),
        original_item_id=value.get("original_item_id"),
        title=str(value.get("title") or ""),
        category=value.get("category"),
        price=value.get("price"),
        description=value.get("description"),
        score=float(value.get("score", value.get("retrieval_score", 0.0))),
        reason=str(value.get("reason") or "综合召回与排序得分较高"),
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(**_container(request).health())


@router.get("/items/{item_id}", response_model=RecommendationItem)
def get_item(item_id: int, request: Request) -> RecommendationItem:
    container = _container(request)
    if container.item_service is None:
        raise HTTPException(status_code=503, detail="Item metadata is not loaded")
    item = container.item_service.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return _item(item)


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
    service = _container(request).recommendation_service
    if service is None:
        raise HTTPException(status_code=503, detail="Recommendation resources are not loaded")
    try:
        values = service.recommend(payload.user_id, payload.query, payload.top_k)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Recommendation failed: {error}") from error
    return RecommendationResponse(user_id=payload.user_id, items=[_item(value) for value in values])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    agent = _container(request).agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent resources are not loaded")
    try:
        state = agent.run(payload.user_id, payload.message, payload.top_k)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Agent failed: {error}") from error
    return ChatResponse(
        user_id=payload.user_id,
        response=state.get("response", ""),
        items=[_item(value) for value in state.get("final_items", [])],
        parsed_intent={
            "category": state.get("category"),
            "max_price": state.get("max_price"),
            "preferences": state.get("preferences", []),
            "negative_preferences": state.get("negative_preferences", []),
        },
    )
