from __future__ import annotations

import json
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def save_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".pt") as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, destination)


@contextmanager
def timed() -> Iterator[Callable[[], float]]:
    started = perf_counter()
    yield lambda: perf_counter() - started
