"""Priority 0 observation-only log schemas."""

from __future__ import annotations

SCHEMA_VERSION = 1
RUN_METADATA_SCHEMA = "viewtrust.priority0.run_metadata"
EVENT_SCHEMA = "viewtrust.priority0.event"

ALLOWED_EVENT_TYPES = {
    "run_start",
    "run_end",
    "config_snapshot",
    "summary_snapshot",
    "table_snapshot",
    "command_start",
    "command_end",
    "iteration_observation",
    "view_observation",
    "gpu_memory_observation",
    "timing_observation",
    "mock_observation",
}

REQUIRED_EVENT_FIELDS = {
    "schema_name",
    "schema_version",
    "run_id",
    "event_type",
    "created_at_utc",
    "payload",
}


def validate_event(event: dict[str, object]) -> None:
    """Validate the stable fields shared by all Priority 0 events."""

    missing = REQUIRED_EVENT_FIELDS.difference(event)
    if missing:
        raise ValueError(f"event missing required fields: {sorted(missing)}")

    if event["schema_name"] != EVENT_SCHEMA:
        raise ValueError(f"unexpected event schema: {event['schema_name']}")

    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema version: {event['schema_version']}")

    if event["event_type"] not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"unsupported event_type: {event['event_type']}")

    if not isinstance(event["payload"], dict):
        raise ValueError("event payload must be a dictionary")
