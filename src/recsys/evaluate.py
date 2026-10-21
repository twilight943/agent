from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import data_path, load_config
from src.common.logger import configure_logging, get_logger
from src.common.utils import save_json
from src.recsys.metrics import hit_rate_at_k, mrr, ndcg_at_k, recall_at_k
from src.services.recommendation_service import RecommendationService


LOGGER = get_logger(__name__)


def evaluate(config: dict[str, Any]) -> dict[str, float]:
    processed = data_path(config, config["data"].get("processed_dir", "processed"))
    test = pd.read_parquet(processed / "test.parquet")
    service = RecommendationService.from_config(config)
    relevant: dict[int, set[int]] = defaultdict(set)
    for row in test.itertuples(index=False):
        relevant[int(row.user_id)].add(int(row.item_id))
    totals = {
        "recall@20": 0.0,
        "recall@50": 0.0,
        "recall@100": 0.0,
        "ndcg@10": 0.0,
        "hitrate@10": 0.0,
        "mrr": 0.0,
    }
    for user_id, item_ids in relevant.items():
        recommendations = service.recommend(user_id, "", top_k=100)
        predicted = [int(value["item_id"]) for value in recommendations]
        totals["recall@20"] += recall_at_k(predicted, item_ids, 20)
        totals["recall@50"] += recall_at_k(predicted, item_ids, 50)
        totals["recall@100"] += recall_at_k(predicted, item_ids, 100)
        totals["ndcg@10"] += ndcg_at_k(predicted, item_ids, 10)
        totals["hitrate@10"] += hit_rate_at_k(predicted, item_ids, 10)
        totals["mrr"] += mrr(predicted, item_ids)
    count = max(len(relevant), 1)
    metrics = {name: value / count for name, value in totals.items()}
    output = Path(str(config["artifact_root"])) / "metrics" / "evaluation.json"
    save_json({"users": len(relevant), **metrics}, output)
    LOGGER.info("evaluation users=%d metrics=%s", len(relevant), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AgentRec offline")
    parser.add_argument("--env", choices=["local", "server"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    configure_logging()
    evaluate(load_config(args.env, args.config))


if __name__ == "__main__":
    main()
