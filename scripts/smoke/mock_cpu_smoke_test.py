#!/usr/bin/env python3
"""LOCAL-SAFE CPU-only smoke test for the ViewTrust-GS scaffold."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    if "gsplat" in sys.modules:
        raise SystemExit("mock smoke test must not import gsplat")

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from viewtrust.configs.loader import load_simple_yaml
    from viewtrust.utils.paths import resolve_relative_path

    config_path = project_root / "configs" / "default.yaml"
    config = load_simple_yaml(config_path)

    paths = config.get("paths", {})
    data_root = resolve_relative_path(project_root, str(paths.get("data_root", "./data")))
    output_root = resolve_relative_path(project_root, str(paths.get("output_root", "./outputs")))
    third_party_root = resolve_relative_path(project_root, str(paths.get("third_party_root", "./third_party")))

    if output_root != (project_root / "outputs").resolve():
        raise ValueError("mock smoke test writes only under ./outputs")

    run_id = "mock-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "mock_smoke_test" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    logger_schema_placeholder = {
        "schema_name": "viewtrust.priority0.metadata",
        "schema_version": 1,
        "required_fields": ["run_id", "stage", "paths", "created_at_utc"],
    }
    for key in ("schema_name", "schema_version", "required_fields"):
        if key not in logger_schema_placeholder:
            raise ValueError(f"logger schema placeholder missing {key}")

    metadata = {
        "run_id": run_id,
        "stage": config.get("project", {}).get("stage", "priority_0_observation"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "project_root": str(project_root),
            "data_root": str(data_root),
            "output_root": str(output_root),
            "third_party_root": str(third_party_root),
        },
        "logger_schema_placeholder": logger_schema_placeholder,
    }

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"config loaded: {config_path}")
    print(f"run_id: {run_id}")
    print(f"metadata written: {run_dir / 'metadata.json'}")
    print("mock smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
