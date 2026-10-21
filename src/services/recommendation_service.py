from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.common.config import data_path
from src.common.utils import load_json
from src.data.features import build_user_history_map
from src.embedding.item_encoder import TextEncoder
from src.embedding.query_encoder import QueryEncoder
from src.recsys.hybrid_retrieval import HybridRetriever
from src.recsys.ranker import DINRanker, load_din
from src.recsys.retrieval import CollaborativeRetriever, DenseIndex, load_two_tower
from src.recsys.semantic_retrieval import SemanticRetriever
from src.recsys.train_retriever import _device
from src.services.item_service import ItemService


class ResourceUnavailableError(RuntimeError):
    pass


class RecommendationService:
    def __init__(
        self,
        item_service: ItemService,
        retriever: HybridRetriever,
        ranker: DINRanker | None = None,
        user_mapping: dict[str, int] | None = None,
        histories: dict[int, list[int]] | None = None,
    ) -> None:
        self.item_service = item_service
        self.retriever = retriever
        self.ranker = ranker
        self.user_mapping = user_mapping or {}
        self.histories = histories or {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RecommendationService":
        processed = data_path(config, config["data"].get("processed_dir", "processed"))
        items_path = processed / "items.parquet"
        if not items_path.exists():
            raise ResourceUnavailableError(f"Processed item metadata not found: {items_path}")
        item_service = ItemService.from_parquet(items_path)
        users_path = processed / "users.parquet"
        user_mapping: dict[str, int] = {}
        if users_path.exists():
            users = pd.read_parquet(users_path)
            user_mapping = {
                str(row.original_user_id): int(row.user_id)
                for row in users.itertuples(index=False)
            }
        interactions_path = processed / config["data"].get("history_file", "train.parquet")
        interactions = pd.read_parquet(interactions_path) if interactions_path.exists() else pd.DataFrame()
        histories = build_user_history_map(interactions) if not interactions.empty else {}

        settings = config["retriever"]
        artifact_root = Path(str(config["artifact_root"]))
        map_path = artifact_root / settings["item_id_map"]
        item_ids = [int(value) for value in load_json(map_path)] if map_path.exists() else []
        device = _device(str(config.get("device", "cpu")))
        collaborative = None
        checkpoint_path = artifact_root / settings["checkpoint"]
        collaborative_embeddings = artifact_root / settings["collaborative_embeddings"]
        collaborative_index = artifact_root / settings["collaborative_index"]
        if checkpoint_path.exists() and item_ids and (collaborative_index.exists() or collaborative_embeddings.exists()):
            model = load_two_tower(checkpoint_path, device)
            index = DenseIndex.load(collaborative_index, collaborative_embeddings, item_ids)
            collaborative = CollaborativeRetriever(model, index, device)

        semantic = None
        semantic_embeddings = artifact_root / settings["semantic_embeddings"]
        semantic_index = artifact_root / settings["semantic_index"]
        if item_ids and (semantic_index.exists() or semantic_embeddings.exists()):
            index = DenseIndex.load(semantic_index, semantic_embeddings, item_ids)
            semantic = SemanticRetriever(QueryEncoder(TextEncoder(config["embedding"])), index)
        if collaborative is None and semantic is None:
            raise ResourceUnavailableError(
                f"No usable retrieval index under {artifact_root}; build indexes or configure artifact_root"
            )
        retriever = HybridRetriever(
            collaborative,
            semantic,
            collaborative_weight=float(settings.get("collaborative_weight", 0.65)),
            semantic_weight=float(settings.get("semantic_weight", 0.35)),
        )

        ranker = None
        rank_settings = config["ranker"]
        rank_checkpoint = artifact_root / rank_settings["checkpoint"]
        if rank_checkpoint.exists():
            rank_model = load_din(rank_checkpoint, device)
            ranker = DINRanker(
                rank_model, histories,
                max_history_length=int(rank_settings.get("max_history_length", 50)), device=device,
            )
        return cls(item_service, retriever, ranker, user_mapping, histories)

    def resolve_user_id(self, user_id: str | int | None) -> int | None:
        if user_id is None:
            return None
        text = str(user_id)
        if text in self.user_mapping:
            return self.user_mapping[text]
        try:
            numeric = int(text)
        except ValueError:
            return None
        return numeric if numeric > 0 else None

    def retrieve(self, user_id: str | int | None, query: str, top_k: int) -> tuple[int | None, list[dict[str, Any]]]:
        internal_user_id = self.resolve_user_id(user_id)
        candidates = self.retriever.hybrid_retrieve(internal_user_id, query, top_k)
        seen = set(self.histories.get(internal_user_id, [])) if internal_user_id else set()
        return internal_user_id, [value for value in candidates if int(value["item_id"]) not in seen]

    def rank_candidates(self, internal_user_id: int | None, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.ranker is not None:
            return self.ranker.rank(internal_user_id, candidates)
        ranked = []
        for candidate in candidates:
            value = dict(candidate)
            value["score"] = float(value.get("retrieval_score", 0.0))
            ranked.append(value)
        return ranked

    def enrich(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            item = self.item_service.get_item(int(candidate["item_id"]))
            if item is not None:
                enriched.append({**item, **candidate})
        return enriched

    def recommend(self, user_id: str | int | None, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        internal_user_id, candidates = self.retrieve(user_id, query, max(top_k * 5, top_k))
        ranked = self.rank_candidates(internal_user_id, candidates)
        return self.enrich(ranked[:top_k])
