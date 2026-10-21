from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.config import data_path, load_config
from src.common.logger import configure_logging, get_logger
from src.common.utils import ensure_dir, save_checkpoint, save_json, set_seed
from src.data.dataset import RetrievalDataset
from src.recsys.models.two_tower import TwoTower


LOGGER = get_logger(__name__)


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _mean_loss(model: TwoTower, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            loss = model.loss(
                batch["user_id"].to(device),
                batch["positive_item_id"].to(device),
                batch["negative_item_ids"].to(device),
            )
            values.append(float(loss.cpu()))
    return sum(values) / max(len(values), 1)


def train(config: dict[str, Any]) -> Path:
    set_seed(int(config.get("seed", 42)))
    processed = data_path(config, config["data"].get("processed_dir", "processed"))
    train_frame = pd.read_parquet(processed / "train.parquet")
    valid_frame = pd.read_parquet(processed / "valid.parquet")
    users = pd.read_parquet(processed / "users.parquet")
    items = pd.read_parquet(processed / "items.parquet")
    settings = config["retriever"]
    dataset = RetrievalDataset(
        train_frame,
        num_items=int(items["item_id"].max()),
        negative_samples=int(settings["negative_samples"]),
        seed=int(config.get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings.get("num_workers", 0)),
    )
    valid_loader = None
    if not valid_frame.empty:
        valid_dataset = RetrievalDataset(
            valid_frame,
            num_items=int(items["item_id"].max()),
            negative_samples=int(settings["negative_samples"]),
            seed=int(config.get("seed", 42)) + 1,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=False,
            num_workers=int(settings.get("num_workers", 0)),
        )
    model = TwoTower(
        num_users=int(users["user_id"].max()),
        num_items=int(items["item_id"].max()),
        embedding_dim=int(settings["embedding_dim"]),
        hidden_dims=settings["hidden_dims"],
    )
    device = _device(str(config.get("device", "cpu")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    best_loss = float("inf")
    patience_left = int(settings.get("patience", 3))
    artifact_root = Path(str(config["artifact_root"]))
    checkpoint = artifact_root / settings["checkpoint"]
    metrics: list[dict[str, float | int]] = []
    ensure_dir(checkpoint.parent)

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(
                batch["user_id"].to(device),
                batch["positive_item_id"].to(device),
                batch["negative_item_ids"].to(device),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = sum(losses) / max(len(losses), 1)
        validation_loss = _mean_loss(model, valid_loader, device) if valid_loader is not None else epoch_loss
        metrics.append({"epoch": epoch, "train_loss": epoch_loss, "validation_loss": validation_loss})
        LOGGER.info(
            "retriever epoch=%d train_loss=%.6f validation_loss=%.6f",
            epoch, epoch_loss, validation_loss,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience_left = int(settings.get("patience", 3))
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "model": model.checkpoint_metadata(),
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "validation_loss": validation_loss,
                },
                checkpoint,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info("Early stopping retriever at epoch %d", epoch)
                break
    save_json(
        {"epochs": metrics, "validation_rows": len(valid_frame), "best_loss": best_loss},
        artifact_root / "metrics" / "retriever.json",
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Two-Tower retriever")
    parser.add_argument("--env", choices=["local", "server"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    configure_logging()
    train(load_config(args.env, args.config))


if __name__ == "__main__":
    main()
