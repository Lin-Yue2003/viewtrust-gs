#!/usr/bin/env python3
"""LOCAL-SAFE smoke test for extensible measurement artifacts."""

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
        raise SystemExit("measurement format smoke test must not import gsplat")

    project_root = _bootstrap_project_imports()

    from viewtrust.analysis.statistics import summarize_table
    from viewtrust.configs.loader import load_simple_yaml
    from viewtrust.logging.writer import Priority0Logger
    from viewtrust.utils.paths import ensure_child_path, resolve_relative_path

    config = load_simple_yaml(project_root / "configs" / "default.yaml")
    output_root = resolve_relative_path(
        project_root,
        str(config.get("paths", {}).get("output_root", "./outputs")),
    )
    if output_root != (project_root / "outputs").resolve():
        raise ValueError("measurement smoke test writes only under ./outputs")

    run_id = "measurement-format-mock-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "measurement_format_smoke_test" / run_id
    ensure_child_path(output_root, run_dir)

    logger = Priority0Logger(run_dir=run_dir, run_id=run_id)
    logger.write_metadata(
        {
            "stage": "priority_0_observation",
            "measurement_format_version": 1,
            "training_behavior_modified": False,
        }
    )
    logger.write_config_snapshot(config)
    logger.write_run_start({"mode": "local_cpu_mock"})

    iteration_rows = [
        {
            "iteration": 0,
            "elapsed_ms": 12.5,
            "gpu_memory_allocated_mb": 0.0,
            "visible_gaussians": 120,
            "notes": "mock row",
        },
        {
            "iteration": 1,
            "elapsed_ms": 13.0,
            "gpu_memory_allocated_mb": 0.0,
            "visible_gaussians": 122,
            "notes": "mock row",
        },
    ]
    view_rows = [
        {
            "iteration": 0,
            "view_id": "mock_view_000",
            "camera_uid": "mock_camera_000",
            "width": 64,
            "height": 64,
        },
        {
            "iteration": 1,
            "view_id": "mock_view_001",
            "camera_uid": "mock_camera_001",
            "width": 64,
            "height": 64,
        },
    ]

    for row in iteration_rows:
        logger.write_event("iteration_observation", row)
    for row in view_rows:
        logger.write_event("view_observation", row)

    logger.write_table(
        "iteration_observations",
        iteration_rows,
        [
            "iteration",
            "elapsed_ms",
            "gpu_memory_allocated_mb",
            "visible_gaussians",
            "notes",
        ],
    )
    logger.write_table(
        "view_observations",
        view_rows,
        ["iteration", "view_id", "camera_uid", "width", "height"],
    )
    stats = {
        "iteration_observations": summarize_table(
            iteration_rows,
            ["elapsed_ms", "gpu_memory_allocated_mb", "visible_gaussians"],
        )
    }
    logger.write_stats(stats)
    logger.write_summary(
        {
            "status": "ok",
            "tables": ["iteration_observations", "view_observations"],
            "event_count_expected": 9,
            "observation_only": True,
        }
    )
    logger.write_run_end({"status": "ok"})

    required_paths = [
        logger.metadata_path,
        logger.config_snapshot_path,
        logger.events_path,
        logger.summary_path,
        logger.stats_path,
        logger.tables_dir / "iteration_observations.csv",
        logger.tables_dir / "view_observations.csv",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    stats_doc = json.loads(logger.stats_path.read_text(encoding="utf-8"))
    elapsed_mean = stats_doc["stats"]["iteration_observations"]["elapsed_ms"]["mean"]
    if elapsed_mean != 12.75:
        raise ValueError(f"unexpected elapsed_ms mean: {elapsed_mean}")

    print(f"measurement format run_dir: {run_dir}")
    print("measurement format smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
