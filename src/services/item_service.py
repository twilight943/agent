from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class ItemService:
    def __init__(self, items: pd.DataFrame) -> None:
        if "item_id" not in items:
            raise ValueError("items data must contain item_id")
        self.items = items.copy()
        self.items["item_id"] = self.items["item_id"].astype(int)
        self._by_id = {
            int(row["item_id"]): {key: _json_value(value) for key, value in row.items()}
            for row in self.items.to_dict(orient="records")
        }

    @classmethod
    def from_parquet(cls, path: str | Path) -> "ItemService":
        return cls(pd.read_parquet(path))

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        value = self._by_id.get(int(item_id))
        return dict(value) if value is not None else None

    def get_items(self, item_ids: Iterable[int]) -> list[dict[str, Any]]:
        return [item for item_id in item_ids if (item := self.get_item(int(item_id))) is not None]

    def __len__(self) -> int:
        return len(self._by_id)
