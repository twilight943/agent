from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.features import pad_history
from src.recsys.models.din import DIN


def load_din(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> DIN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DIN(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


class DINRanker:
    def __init__(
        self,
        model: DIN,
        histories: dict[int, list[int]],
        max_history_length: int = 50,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.histories = histories
        self.max_history_length = int(max_history_length)
        self.device = torch.device(device)

    def rank(self, user_id: int | None, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates or user_id is None or user_id <= 0 or user_id > self.model.num_users:
            return sorted(candidates, key=lambda value: float(value.get("retrieval_score", 0.0)), reverse=True)
        history, _ = pad_history(self.histories.get(user_id, []), self.max_history_length)
        count = len(candidates)
        users = torch.full((count,), user_id, dtype=torch.long, device=self.device)
        item_ids = torch.tensor([int(value["item_id"]) for value in candidates], dtype=torch.long, device=self.device)
        histories = torch.tensor([history] * count, dtype=torch.long, device=self.device)
        scores = self.model.predict(users, item_ids, histories).cpu().tolist()
        ranked = []
        for candidate, score in zip(candidates, scores):
            value = dict(candidate)
            value["ranking_score"] = float(score)
            value["score"] = float(score)
            ranked.append(value)
        return sorted(ranked, key=lambda value: float(value["ranking_score"]), reverse=True)
