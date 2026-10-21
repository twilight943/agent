from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import data_path, load_config
from src.common.logger import configure_logging, get_logger
from src.common.utils import ensure_dir, save_checkpoint, save_json, set_seed
from src.data.dataset import RankingDataset, ranking_samples_with_negatives
from src.data.features import attach_user_histories
from src.recsys.models.din import DIN
from src.recsys.train_retriever import _device


LOGGER = get_logger(__name__)


def _prepare_split(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    max_history: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_frame.copy()
    valid = valid_frame.copy()
    train["__split"] = "train"
    valid["__split"] = "valid"
    combined = attach_user_histories(pd.concat([train, valid], ignore_index=True), max_history)
    return (
        combined[combined["__split"] == "train"].drop(columns="__split"),
        combined[combined["__split"] == "valid"].drop(columns="__split"),
    )


def _mean_loss(model: DIN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    values: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["user_id"].to(device),
                batch["item_id"].to(device),
                batch["history_item_ids"].to(device),
            )
            values.append(float(criterion(logits, batch["label"].to(device)).cpu()))
    return sum(values) / max(len(values), 1)


def train(config: dict[str, Any]) -> Path:
    set_seed(int(config.get("seed", 42)))
    processed = data_path(config, config["data"].get("processed_dir", "processed"))
    train_frame = pd.read_parquet(processed / "train.parquet")
    valid_frame = pd.read_parquet(processed / "valid.parquet")
    users = pd.read_parquet(processed / "users.parquet")
    items = pd.read_parquet(processed / "items.parquet")
    settings = config["ranker"]
    max_history = int(settings["max_history_length"])
    train_frame, valid_frame = _prepare_split(train_frame, valid_frame, max_history)
    num_items = int(items["item_id"].max())
    train_samples = ranking_samples_with_negatives(
        train_frame, num_items, int(settings.get("negative_samples", 4)), int(config.get("seed", 42))
    )
    valid_samples = ranking_samples_with_negatives(
        valid_frame, num_items, 1, int(config.get("seed", 42)) + 1
    )
    train_loader = DataLoader(
        RankingDataset(train_samples, max_history),
        batch_size=int(settings["batch_size"]), shuffle=True,
        num_workers=int(settings.get("num_workers", 0)),
    )
    valid_loader = DataLoader(
        RankingDataset(valid_samples, max_history),
        batch_size=int(settings["batch_size"]), shuffle=False,
        num_workers=int(settings.get("num_workers", 0)),
    )
    model = DIN(
        num_users=int(users["user_id"].max()), num_items=num_items,
        embedding_dim=int(settings["embedding_dim"]),
        attention_dim=int(settings["attention_dim"]), hidden_dims=settings["hidden_dims"],
    )
    device = _device(str(config.get("device", "cpu")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    patience_left = int(settings.get("patience", 3))
    artifact_root = Path(str(config["artifact_root"]))
    checkpoint = artifact_root / settings["checkpoint"]
    ensure_dir(checkpoint.parent)
    metrics: list[dict[str, float | int]] = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["user_id"].to(device), batch["item_id"].to(device),
                batch["history_item_ids"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss = _mean_loss(model, valid_loader, device) if len(valid_samples) else sum(train_losses) / max(len(train_losses), 1)
        train_loss = sum(train_losses) / max(len(train_losses), 1)
        metrics.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        LOGGER.info("ranker epoch=%d train_loss=%.6f validation_loss=%.6f", epoch, train_loss, validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience_left = int(settings.get("patience", 3))
            save_checkpoint(
                {"state_dict": model.state_dict(), "model": model.checkpoint_metadata(), "epoch": epoch},
                checkpoint,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info("Early stopping ranker at epoch %d", epoch)
                break
    save_json({"epochs": metrics, "best_validation_loss": best_loss}, artifact_root / "metrics" / "ranker.json")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DIN ranker")
    parser.add_argument("--env", choices=["local", "server"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    configure_logging()
    train(load_config(args.env, args.config))


if __name__ == "__main__":
    main()
