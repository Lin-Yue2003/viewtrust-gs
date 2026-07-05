#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for Priority 0 observation logging."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def main() -> int:
    if "gsplat" in sys.modules:
        raise SystemExit("Priority 0 logging smoke test must not import gsplat")

    project_root = _bootstrap_project_imports()

    from viewtrust.configs.loader import load_simple_yaml
    from viewtrust.logging.schema import EVENT_SCHEMA, SCHEMA_VERSION
    from viewtrust.logging.writer import Priority0Logger
    from viewtrust.utils.paths import ensure_child_path, resolve_relative_path

    config_path = project_root / "configs" / "default.yaml"
    config = load_simple_yaml(config_path)

    output_root = resolve_relative_path(
        project_root,
        str(config.get("paths", {}).get("output_root", "./outputs")),
    )
    expected_output_root = (project_root / "outputs").resolve()
    if output_root != expected_output_root:
        raise ValueError("local smoke test writes only under ./outputs")

    run_id = "priority0-mock-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "priority0_logging_smoke_test" / run_id
    ensure_child_path(output_root, run_dir)

    logger = Priority0Logger(run_dir=run_dir, run_id=run_id)
    logger.write_metadata(
        {
            "stage": "priority_0_observation",
            "mode": "local_cpu_mock",
            "training_behavior_modified": False,
        }
    )
    logger.write_run_start({"source": "priority0_logging_smoke_test"})
    logger.write_config_snapshot(config)
    logger.write_event(
        "iteration_observation",
        {
            "iteration": 0,
            "scene": "mock_scene",
            "condition": "local_cpu",
            "notes": "observation-only placeholder; no training code executed",
        },
    )
    logger.write_event(
        "view_observation",
        {
            "iteration": 0,
            "view_id": "mock_view_000",
            "observed": True,
        },
    )
    logger.write_run_end({"status": "ok"})

    for required_path in (
        logger.metadata_path,
        logger.config_snapshot_path,
        logger.events_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    events = [
        json.loads(line)
        for line in logger.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(events) != 5:
        raise ValueError(f"expected 5 events, got {len(events)}")

    for event in events:
        if event["schema_name"] != EVENT_SCHEMA:
            raise ValueError(f"unexpected event schema: {event['schema_name']}")
        if event["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unexpected schema version: {event['schema_version']}")

    print(f"priority0 run_dir: {run_dir}")
    print("priority0 logging smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
