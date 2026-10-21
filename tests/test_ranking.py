import torch

from src.recsys.metrics import hit_rate_at_k, mrr, ndcg_at_k, recall_at_k
from src.recsys.models.din import DIN


def test_din_forward_and_predict_are_finite() -> None:
    model = DIN(
        num_users=3,
        num_items=5,
        embedding_dim=8,
        attention_dim=8,
        hidden_dims=[16, 8],
        dropout=0.0,
    )
    logits = model(
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.tensor([[0, 1, 2], [0, 0, 0]]),
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    probabilities = model.predict(
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.tensor([[0, 1, 2], [0, 0, 0]]),
    )
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_metrics() -> None:
    recommended, relevant = [1, 2, 3], {2, 4}
    assert recall_at_k(recommended, relevant, 3) == 0.5
    assert hit_rate_at_k(recommended, relevant, 1) == 0.0
    assert ndcg_at_k(recommended, relevant, 3) > 0
    assert mrr(recommended, relevant) == 0.5
