"""Observation-only JSON writer for Priority 0 logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from viewtrust.logging.schema import (
    EVENT_SCHEMA,
    RUN_METADATA_SCHEMA,
    SCHEMA_VERSION,
    validate_event,
)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


class Priority0Logger:
    """Write observation-only metadata and events.

    This class records facts supplied by the caller. It does not compute trust
    scores, change training behavior, or import GPU-specific packages.
    """

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.events_path = run_dir / "events.jsonl"
        self.metadata_path = run_dir / "metadata.json"
        self.config_snapshot_path = run_dir / "config_snapshot.json"
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def write_metadata(self, metadata: dict[str, Any]) -> Path:
        document = {
            "schema_name": RUN_METADATA_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at_utc": utc_now_iso(),
            "metadata": metadata,
        }
        self.metadata_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.metadata_path

    def write_config_snapshot(self, config: dict[str, Any]) -> Path:
        document = {
            "schema_name": "viewtrust.priority0.config_snapshot",
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at_utc": utc_now_iso(),
            "config": config,
        }
        self.config_snapshot_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.write_event("config_snapshot", {"path": str(self.config_snapshot_path)})
        return self.config_snapshot_path

    def write_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "schema_name": EVENT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "event_type": event_type,
            "created_at_utc": utc_now_iso(),
            "payload": payload,
        }
        validate_event(event)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def write_run_start(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.write_event("run_start", payload or {})

    def write_run_end(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.write_event("run_end", payload or {})
