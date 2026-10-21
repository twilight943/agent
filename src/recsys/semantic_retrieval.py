from __future__ import annotations

from src.embedding.query_encoder import QueryEncoder
from src.recsys.retrieval import DenseIndex


class SemanticRetriever:
    def __init__(self, query_encoder: QueryEncoder, index: DenseIndex) -> None:
        self.query_encoder = query_encoder
        self.index = index

    def semantic_retrieve(self, query: str, top_k: int = 200) -> list[dict[str, float | int]]:
        if not query.strip():
            return []
        embedding = self.query_encoder.encode_query(query)
        return self.index.search(embedding, top_k)
