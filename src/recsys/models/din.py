from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _mlp(dimensions: Sequence[int], dropout: float = 0.1) -> nn.Sequential:
    layers: list[nn.Module] = []
    for source, target in zip(dimensions[:-2], dimensions[1:-1]):
        layers.extend([nn.Linear(source, target), nn.ReLU(), nn.Dropout(dropout)])
    layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
    return nn.Sequential(*layers)


class DIN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        attention_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.attention_dim = int(attention_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.dropout = float(dropout)
        self.user_embedding = nn.Embedding(self.num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.num_items + 1, embedding_dim, padding_idx=0)
        self.attention = _mlp([embedding_dim * 4, attention_dim, 1], dropout=dropout)
        self.output = _mlp(
            [embedding_dim * 3, *self.hidden_dims, 1], dropout=dropout
        )

    def _attend(self, target: torch.Tensor, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = target.unsqueeze(1).expand_as(history)
        features = torch.cat([query, history, query - history, query * history], dim=-1)
        logits = self.attention(features).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(-1) * history).sum(dim=1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user = self.user_embedding(user_ids)
        target = self.item_embedding(item_ids)
        history = self.item_embedding(history_item_ids)
        attended = self._attend(target, history, history_item_ids.ne(0))
        return self.output(torch.cat([user, target, attended], dim=-1)).squeeze(-1)

    @torch.inference_mode()
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        probabilities = torch.sigmoid(self(user_ids, item_ids, history_item_ids))
        if was_training:
            self.train()
        return probabilities

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "attention_dim": self.attention_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
        }
