from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logger(name: str = "clip") -> logging.Logger:
    """Create a simple stream logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dict as pretty JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
