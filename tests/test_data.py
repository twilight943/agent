import pandas as pd

from src.data.dataset import RankingDataset, RetrievalDataset
from src.data.features import attach_user_histories
from src.data.preprocess import _text_value, iterative_k_core
from src.data.split import assert_no_temporal_leakage, split_by_user_time


def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "item_id": [1, 2, 3, 2, 3, 4],
            "timestamp": [1, 2, 3, 1, 3, 4],
            "label": [1.0] * 6,
        }
    )


def test_temporal_split_has_no_future_leakage() -> None:
    train, valid, test = split_by_user_time(interactions())
    assert len(train) == len(valid) == len(test) == 2
    assert_no_temporal_leakage(train, valid, test)
    assert train.groupby("user_id")["timestamp"].max().to_dict() == {1: 1, 2: 1}


def test_iterative_k_core_and_datasets() -> None:
    frame = interactions()
    filtered = iterative_k_core(frame, min_user_interactions=2, min_item_interactions=2)
    assert set(filtered.item_id) == {2, 3}
    retrieval = RetrievalDataset(frame, num_items=5, negative_samples=2)
    sample = retrieval[0]
    assert sample["negative_item_ids"].shape == (2,)
    assert all(value not in {1, 2, 3} for value in sample["negative_item_ids"].tolist())

    featured = attach_user_histories(frame, max_history_length=2)
    ranking = RankingDataset(featured, max_history_length=2)
    assert ranking[2]["history_item_ids"].tolist() == [1, 2]


def test_amazon_metadata_lists_become_embedding_text() -> None:
    assert _text_value(["Skin Care", "Face"]) == "Skin Care > Face"
    assert _text_value([]) == ""
