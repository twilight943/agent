from __future__ import annotations

import logging
import logging.config
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    handlers: dict[str, dict[str, object]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    }
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": str(path),
            "maxBytes": 10_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
        }
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                }
            },
            "handlers": handlers,
            "root": {"level": level.upper(), "handlers": list(handlers)},
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
