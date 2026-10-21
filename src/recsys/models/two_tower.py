from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _tower(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dimensions = [input_dim, *hidden_dims]
    if not hidden_dims or hidden_dims[-1] != output_dim:
        dimensions.append(output_dim)
    layers: list[nn.Module] = []
    for source, target in zip(dimensions[:-1], dimensions[1:]):
        layers.extend([nn.Linear(source, target), nn.ReLU(), nn.LayerNorm(target)])
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.user_embedding = nn.Embedding(self.num_users + 1, self.embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx=0)
        self.user_tower = _tower(self.embedding_dim, self.hidden_dims, self.embedding_dim)
        self.item_tower = _tower(self.embedding_dim, self.hidden_dims, self.embedding_dim)

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(self.user_embedding(user_ids)), dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(self.item_embedding(item_ids)), dim=-1)

    def forward(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        users = self.encode_user(user_ids)
        positives = self.encode_item(positive_item_ids)
        positive_scores = (users * positives).sum(dim=-1)
        result = {"positive_scores": positive_scores}
        if negative_item_ids is not None:
            negatives = self.encode_item(negative_item_ids)
            result["negative_scores"] = (users.unsqueeze(1) * negatives).sum(dim=-1)
        return result

    def loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        scores = self(user_ids, positive_item_ids, negative_item_ids)
        margins = scores["positive_scores"].unsqueeze(1) - scores["negative_scores"]
        return F.softplus(-margins).mean()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": list(self.hidden_dims),
        }
