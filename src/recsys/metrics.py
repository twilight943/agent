from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(set(recommended[:k]) & relevant_set) / len(relevant_set)


def hit_rate_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    return float(bool(set(recommended[:k]) & set(relevant)))


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item_id in enumerate(recommended[:k])
        if item_id in relevant_set
    )
    ideal_length = min(k, len(relevant_set))
    ideal = sum(1.0 / math.log2(position + 2) for position in range(ideal_length))
    return dcg / ideal if ideal else 0.0


def mrr(recommended: Sequence[int], relevant: Iterable[int]) -> float:
    relevant_set = set(relevant)
    for position, item_id in enumerate(recommended, start=1):
        if item_id in relevant_set:
            return 1.0 / position
    return 0.0
