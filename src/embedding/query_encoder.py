from __future__ import annotations

import numpy as np

from src.embedding.item_encoder import TextEncoder


class QueryEncoder:
    def __init__(self, text_encoder: TextEncoder) -> None:
        self.text_encoder = text_encoder

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Semantic query must not be empty")
        return self.text_encoder.encode([query])[0]
