"""Ingestion pipeline for loading the Omnia Retail dataset into ChromaDB."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from omnia_support_models import Dataset

if TYPE_CHECKING:
    from pathlib import Path


def load_dataset(path: Path) -> Dataset:
    """Read and validate the dataset JSON file."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return Dataset.model_validate(raw)
