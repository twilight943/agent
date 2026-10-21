from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.common.config import data_path, load_config
from src.common.logger import configure_logging, get_logger
from src.common.utils import ensure_dir, save_json
from src.embedding.item_encoder import ItemEncoder, TextEncoder
from src.recsys.retrieval import DenseIndex, load_two_tower
from src.recsys.train_retriever import _device


LOGGER = get_logger(__name__)


def build_indexes(config: dict[str, Any]) -> dict[str, Path]:
    settings = config["retriever"]
    artifact_root = Path(str(config["artifact_root"]))
    processed = data_path(config, config["data"].get("processed_dir", "processed"))
    items = pd.read_parquet(processed / "items.parquet").sort_values("item_id")
    item_ids = items["item_id"].astype(int).tolist()
    device = _device(str(config.get("device", "cpu")))
    model = load_two_tower(artifact_root / settings["checkpoint"], device)
    batch_size = int(config.get("embedding", {}).get("batch_size", 256))
    collaborative_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(item_ids), batch_size):
            batch = torch.tensor(item_ids[offset : offset + batch_size], dtype=torch.long, device=device)
            collaborative_parts.append(model.encode_item(batch).cpu().numpy())
    collaborative_embeddings = np.concatenate(collaborative_parts, axis=0)
    semantic_embeddings = ItemEncoder(TextEncoder(config["embedding"])).encode_items(items)

    collaborative_index = DenseIndex.build(collaborative_embeddings, item_ids)
    semantic_index = DenseIndex.build(semantic_embeddings, item_ids)
    paths = {
        "collaborative_index": artifact_root / settings["collaborative_index"],
        "collaborative_embeddings": artifact_root / settings["collaborative_embeddings"],
        "semantic_index": artifact_root / settings["semantic_index"],
        "semantic_embeddings": artifact_root / settings["semantic_embeddings"],
        "item_id_map": artifact_root / settings["item_id_map"],
    }
    collaborative_index.save(paths["collaborative_index"], paths["collaborative_embeddings"])
    semantic_index.save(paths["semantic_index"], paths["semantic_embeddings"])
    ensure_dir(paths["item_id_map"].parent)
    save_json(item_ids, paths["item_id_map"])
    LOGGER.info("Built collaborative and semantic indexes for %d items", len(item_ids))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AgentRec FAISS indexes")
    parser.add_argument("--env", choices=["local", "server"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    configure_logging()
    build_indexes(load_config(args.env, args.config))


if __name__ == "__main__":
    main()
