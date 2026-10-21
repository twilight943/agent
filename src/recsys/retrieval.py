from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.recsys.models.two_tower import TwoTower


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)


class DenseIndex:
    """Cosine-similarity index with FAISS and exact NumPy backends."""

    def __init__(self, item_ids: list[int], embeddings: np.ndarray | None = None, index: Any = None) -> None:
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.embeddings = normalize_rows(embeddings) if embeddings is not None else None
        self.index = index
        if self.embeddings is None and self.index is None:
            raise ValueError("DenseIndex needs embeddings or a FAISS index")

    @classmethod
    def build(cls, embeddings: np.ndarray, item_ids: list[int]) -> "DenseIndex":
        normalized = normalize_rows(embeddings)
        if len(normalized) != len(item_ids):
            raise ValueError("Embedding count and item ID count differ")
        index = None
        try:
            import faiss

            index = faiss.IndexFlatIP(normalized.shape[1])
            index.add(normalized)
        except ImportError:
            pass
        return cls(item_ids=item_ids, embeddings=normalized, index=index)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        embeddings_path: str | Path,
        item_ids: list[int],
    ) -> "DenseIndex":
        index_path, embeddings_path = Path(index_path), Path(embeddings_path)
        index = None
        if index_path.exists():
            try:
                import faiss

                index = faiss.read_index(str(index_path))
            except ImportError:
                index = None
        embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        return cls(item_ids=item_ids, embeddings=embeddings, index=index)

    def save(self, index_path: str | Path, embeddings_path: str | Path) -> None:
        index_path, embeddings_path = Path(index_path), Path(embeddings_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(embeddings_path, self.embeddings)
        if self.index is not None:
            import faiss

            faiss.write_index(self.index, str(index_path))

    def search(self, query: np.ndarray, top_k: int) -> list[dict[str, float | int]]:
        if top_k <= 0:
            return []
        query = normalize_rows(query)
        limit = min(int(top_k), len(self.item_ids))
        if self.index is not None:
            scores, positions = self.index.search(query, limit)
            score_values, position_values = scores[0], positions[0]
        elif self.embeddings is not None:
            all_scores = self.embeddings @ query[0]
            position_values = np.argsort(-all_scores)[:limit]
            score_values = all_scores[position_values]
        else:
            return []
        return [
            {"item_id": int(self.item_ids[position]), "retrieval_score": float(score)}
            for score, position in zip(score_values, position_values)
            if position >= 0
        ]


def load_two_tower(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> TwoTower:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TwoTower(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


class CollaborativeRetriever:
    def __init__(self, model: TwoTower, index: DenseIndex, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.index = index
        self.device = torch.device(device)

    @torch.inference_mode()
    def retrieve(self, user_id: int, top_k: int = 300) -> list[dict[str, float | int]]:
        if user_id <= 0 or user_id > self.model.num_users:
            return []
        user = torch.tensor([user_id], dtype=torch.long, device=self.device)
        query = self.model.encode_user(user).cpu().numpy()
        return self.index.search(query, top_k)
