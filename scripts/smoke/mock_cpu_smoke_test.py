#!/usr/bin/env python3
"""LOCAL-SAFE CPU-only smoke test for the ViewTrust-GS scaffold."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected a key/value pair")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value:
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))

    return root


def resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"mock smoke test requires relative config paths, got {raw_path}")
    return (project_root / path).resolve()


def main() -> int:
    if "gsplat" in sys.modules:
        raise SystemExit("mock smoke test must not import gsplat")

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "default.yaml"
    config = load_simple_yaml(config_path)

    paths = config.get("paths", {})
    data_root = resolve_project_path(project_root, str(paths.get("data_root", "./data")))
    output_root = resolve_project_path(project_root, str(paths.get("output_root", "./outputs")))
    third_party_root = resolve_project_path(project_root, str(paths.get("third_party_root", "./third_party")))

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
