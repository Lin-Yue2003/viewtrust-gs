#!/usr/bin/env python3
"""Inspect a generated natural corruption dataset condition."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _frames(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    frames = payload.get("frames", [])
    return frames if isinstance(frames, list) else []


def _extensionless(frames: list[dict[str, Any]]) -> bool:
    return all(Path(str(frame.get("file_path", ""))).suffix == "" for frame in frames)


def _all_images_exist(condition_path: Path, frames: list[dict[str, Any]]) -> bool:
    for frame in frames:
        file_path = Path(str(frame.get("file_path", "")))
        image_path = condition_path / file_path.with_suffix(".png")
        if not image_path.exists():
            return False
    return True


def inspect_condition(data_root: Path, scene: str, condition: str) -> dict[str, Any]:
    condition_path = (data_root / "viewtrust-mini" / "nerf_synthetic" / scene / condition).resolve()
    train_path = condition_path / "transforms_train.json"
    test_path = condition_path / "transforms_test.json"
    target_path = condition_path / "transforms_target.json"
    manifest_json = condition_path / "corruption_manifest.json"
    manifest_csv = condition_path / "corruption_manifest.csv"
    summary_json = condition_path / "corruption_summary.json"
    preview_grid = condition_path / "preview" / "preview_grid.png"
    train_frames = _frames(train_path)
    test_frames = _frames(test_path)
    target_frames = _frames(target_path)
    all_frames = [*train_frames, *test_frames, *target_frames]
    summary = _load_json(summary_json)
    manifest = _load_json(manifest_json)
    csv_rows: list[dict[str, str]] = []
    if manifest_csv.exists():
        with manifest_csv.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
    corrupted_image_count = int(summary.get("corrupted_image_count", 0) or 0)
    test_target_uncorrupted = all(
        row.get("was_corrupted") != "true"
        for row in csv_rows
        if row.get("split") in {"test", "target"}
    )
    selected_names = set(summary.get("selected_train_views", []))
    train_names = {Path(str(frame.get("file_path", ""))).name for frame in train_frames}
    selected_valid = selected_names.issubset(train_names)
    preview_required = int(summary.get("preview_count", 0) or 0) > 0
    extensionless = _extensionless(all_frames)
    all_exist = _all_images_exist(condition_path, all_frames)
    warnings: list[str] = []
    checks = {
        "condition_exists": condition_path.exists(),
        "has_images_dir": (condition_path / "images").is_dir(),
        "has_transforms_train": train_path.exists(),
        "has_transforms_test": test_path.exists(),
        "has_transforms_target": target_path.exists(),
        "has_manifest_json": manifest_json.exists(),
        "has_manifest_csv": manifest_csv.exists(),
        "has_summary_json": summary_json.exists(),
        "has_preview_grid": (not preview_required) or preview_grid.exists(),
        "transforms_extensionless_file_path": extensionless,
        "all_transform_images_exist": all_exist,
        "test_target_uncorrupted": test_target_uncorrupted,
        "selected_train_views_valid": selected_valid,
    }
    for name, ok in checks.items():
        if not ok:
            warnings.append(name)
    valid = all(checks.values())
    return {
        "scene": scene,
        "condition": condition,
        "condition_path": str(condition_path),
        **checks,
        "train_view_count": len(train_frames),
        "test_view_count": len(test_frames),
        "target_view_count": len(target_frames),
        "total_view_count": len(all_frames),
        "corrupted_image_count": corrupted_image_count,
        "manifest_corruption_count": len(manifest.get("corruptions", [])),
        "valid": valid,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("VIEWTRUST_DATA_ROOT", "./data"))
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--require-valid", action="store_true")
    parser.add_argument("--require-corrupted-count", type=int)
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    report = inspect_condition(data_root, args.scene, args.condition)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_valid and not report["valid"]:
        return 1
    if (
        args.require_corrupted_count is not None
        and report["corrupted_image_count"] != args.require_corrupted_count
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
