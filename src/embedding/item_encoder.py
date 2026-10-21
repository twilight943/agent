from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
import pandas as pd


class TextEncoder:
    """Embedding adapter for a local SentenceTransformer or OpenAI-compatible API."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.provider = str(config.get("provider", "sentence_transformers"))
        self.normalize = bool(config.get("normalize", True))
        self._model: Any = None

    def _local_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError("Install sentence-transformers to use the local embedding provider") from error
            self._model = SentenceTransformer(str(self.config["model_name_or_path"]))
            if self.config.get("max_length") is not None:
                self._model.max_seq_length = int(self.config["max_length"])
            if self.config.get("use_fp16", False) and str(self._model.device).startswith("cuda"):
                self._model.half()
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.provider == "sentence_transformers":
            values = self._local_model().encode(
                list(texts),
                batch_size=int(self.config.get("batch_size", 128)),
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(values, dtype=np.float32)
        if self.provider in {"openai", "http"}:
            url = str(self.config["base_url"]).rstrip("/") + "/embeddings"
            headers = {"Authorization": f"Bearer {self.config.get('api_key', 'EMPTY')}"}
            batch_size = int(self.config.get("batch_size", 128))
            vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                response = httpx.post(
                    url,
                    headers=headers,
                    json={"model": self.config["model_name_or_path"], "input": list(texts[start:start + batch_size])},
                    timeout=float(self.config.get("timeout_seconds", 120)),
                )
                response.raise_for_status()
                ordered = sorted(response.json()["data"], key=lambda value: value.get("index", 0))
                vectors.extend(value["embedding"] for value in ordered)
            values = np.asarray(vectors, dtype=np.float32)
            if self.normalize:
                values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
            return values
        raise ValueError(f"Unsupported embedding provider: {self.provider}")


class ItemEncoder:
    def __init__(self, text_encoder: TextEncoder) -> None:
        self.text_encoder = text_encoder

    @staticmethod
    def item_text(item: dict[str, Any] | pd.Series) -> str:
        fields = [
            f"标题: {item.get('title', '')}",
            f"类别: {item.get('category', '')}",
            f"特征: {item.get('features', '')}",
            f"描述: {item.get('description', '')}",
        ]
        return "\n".join(part for part in fields if part.split(":", 1)[1].strip())

    def encode_items(self, items: pd.DataFrame) -> np.ndarray:
        return self.text_encoder.encode([self.item_text(row) for _, row in items.iterrows()])
