"""Shared evaluation infrastructure: hashing, dataset sync, config helpers."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import yaml
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset, get_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.datasets import EvaluationDataset


def content_hash(raw: str) -> str:
    """Compute a SHA-256 hash of the gold set file contents."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sync_dataset(
    records: list[dict[str, Any]],
    hash_value: str,
    experiment_id: str,
    dataset_name_prefix: str = "omnia-support-gold-set",
) -> EvaluationDataset:
    """Sync evaluation records into an MLflow dataset, skipping if unchanged."""
    short_hash = hash_value[:12]
    dataset_name = f"{dataset_name_prefix}-{short_hash}"

    try:
        dataset = get_dataset(name=dataset_name)
    except MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
    else:
        print(f"  MLflow dataset '{dataset_name}' is up to date")
        return dataset

    print(f"  Creating MLflow dataset '{dataset_name}'...")
    dataset = create_dataset(
        name=dataset_name,
        experiment_id=experiment_id,
    )

    dataset.merge_records(records)
    print(f"  Synced {len(records)} records to MLflow dataset")
    return dataset


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict if no path given."""
    if config_path is None:
        return {}
    if not config_path.exists():
        raise SystemExit(f"Error: config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def resolve_config(
    args: Any,
    defaults: dict[str, Any],
    yaml_config: dict[str, Any],
) -> None:
    """Apply defaults → YAML → CLI precedence to parsed args (in place)."""
    for key, default in defaults.items():
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            continue
        yaml_val = yaml_config.get(key)
        setattr(args, key, yaml_val if yaml_val is not None else default)


def resolve_prompt_uri(name: str, version: int | None, alias: str) -> str:
    """Build an MLflow prompt URI from name + version or alias."""
    if version is not None:
        return f"prompts:/{name}/{version}"
    return f"prompts:/{name}@{alias}"
