"""Dataset manifest schema for reproducible installs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_MANIFEST_SCHEMA = "viewtrust.datasets.manifest"
DATASET_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    url: str
    target_subdir: str
    sha256: str | None = None
    archive_type: str | None = None
    optional: bool = False


@dataclass(frozen=True)
class DatasetManifest:
    schema_name: str
    schema_version: int
    datasets: tuple[DatasetEntry, ...]


def _entry_from_dict(data: dict[str, Any]) -> DatasetEntry:
    required = ("name", "url", "target_subdir")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"dataset entry missing required keys: {missing}")

    return DatasetEntry(
        name=str(data["name"]),
        url=str(data["url"]),
        target_subdir=str(data["target_subdir"]),
        sha256=str(data["sha256"]) if data.get("sha256") else None,
        archive_type=str(data["archive_type"]) if data.get("archive_type") else None,
        optional=bool(data.get("optional", False)),
    )


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Load a JSON dataset manifest."""

    data = json.loads(path.read_text(encoding="utf-8"))
    schema_name = data.get("schema_name")
    schema_version = data.get("schema_version")
    if schema_name != DATASET_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected dataset manifest schema: {schema_name}")
    if schema_version != DATASET_MANIFEST_VERSION:
        raise ValueError(f"unexpected dataset manifest version: {schema_version}")

    datasets = data.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("dataset manifest requires a datasets list")

    return DatasetManifest(
        schema_name=schema_name,
        schema_version=schema_version,
        datasets=tuple(_entry_from_dict(entry) for entry in datasets),
    )
