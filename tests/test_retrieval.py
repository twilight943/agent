import numpy as np
import torch

from src.recsys.hybrid_retrieval import HybridRetriever
from src.recsys.models.two_tower import TwoTower
from src.recsys.retrieval import DenseIndex


class FakeCollaborative:
    def retrieve(self, user_id: int, top_k: int):
        return [
            {"item_id": 1, "retrieval_score": 0.9},
            {"item_id": 2, "retrieval_score": 0.2},
        ]


class FakeSemantic:
    def semantic_retrieve(self, query: str, top_k: int):
        return [
            {"item_id": 2, "retrieval_score": 0.8},
            {"item_id": 3, "retrieval_score": 0.1},
        ]


def test_two_tower_shapes_and_dense_search() -> None:
    model = TwoTower(num_users=3, num_items=4, embedding_dim=8, hidden_dims=[16, 8])
    users = model.encode_user(torch.tensor([1, 2]))
    items = model.encode_item(torch.tensor([1, 3]))
    assert users.shape == items.shape == (2, 8)
    output = model(torch.tensor([1, 2]), torch.tensor([1, 3]), torch.tensor([[2, 3], [1, 4]]))
    assert output["negative_scores"].shape == (2, 2)

    index = DenseIndex.build(np.eye(3, dtype=np.float32), [10, 11, 12])
    assert index.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), 1)[0]["item_id"] == 11


def test_hybrid_retrieval_deduplicates_candidates() -> None:
    retriever = HybridRetriever(FakeCollaborative(), FakeSemantic(), 0.5, 0.5)
    results = retriever.hybrid_retrieve(1, "light headphones", 10)
    assert {value["item_id"] for value in results} == {1, 2, 3}
    assert len(results) == 3
