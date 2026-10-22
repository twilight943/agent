from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd


def _json_value(value):
    # NumPy 数组：转换为 Python 类型并递归处理
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())

    # NumPy 标量
    if isinstance(value, np.generic):
        return _json_value(value.item())

    # 列表或元组
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]

    # 字典
    if isinstance(value, dict):
        return {
            str(k): _json_value(v)
            for k, v in value.items()
        }

    # 空值
    if value is None or value is pd.NA:
        return None

    # NaN / Infinity
    if isinstance(value, float) and not math.isfinite(value):
        return None

    # 时间类型
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


# def _json_value(value: Any) -> Any:
#     if value is None or (isinstance(value, float) and math.isnan(value)):
#         return None
#     if hasattr(value, "item"):
#         return value.item()
#     return value


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
