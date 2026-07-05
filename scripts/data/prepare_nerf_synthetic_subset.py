#!/usr/bin/env python3
"""Prepare a minimal NeRF Synthetic clean subset for ViewTrust-GS."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def parse_args() -> argparse.Namespace:
    data_root = os.environ.get("VIEWTRUST_DATA_ROOT")
    default_raw_scene_root = (
        str(Path(data_root) / "raw" / "nerf_synthetic" / "chair")
        if data_root
        else "data/raw/nerf_synthetic/chair"
    )
    default_output_root = (
        str(Path(data_root) / "viewtrust-mini" / "nerf_synthetic" / "chair")
        if data_root
        else "data/viewtrust-mini/nerf_synthetic/chair"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scene-root", default=default_raw_scene_root)
    parser.add_argument("--output-root", default=default_output_root)
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--max-train-views", type=int, default=20)
    parser.add_argument("--max-test-views", type=int, default=5)
    parser.add_argument("--max-target-views", type=int, default=3)
    parser.add_argument("--max-image-width", type=int, default=400)
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.datasets.nerf_synthetic import (
        plan_summary,
        prepare_nerf_synthetic_subset,
    )

    args = parse_args()
    max_image_width = args.max_image_width if args.max_image_width > 0 else None
    plan = prepare_nerf_synthetic_subset(
        raw_scene_root=_resolve_path(project_root, args.raw_scene_root),
        output_root=_resolve_path(project_root, args.output_root),
        scene=args.scene,
        condition=args.condition,
        max_train_views=args.max_train_views,
        max_test_views=args.max_test_views,
        max_target_views=args.max_target_views,
        max_image_width=max_image_width,
        copy_mode=args.copy_mode,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    print(json.dumps(plan_summary(plan, dry_run=args.dry_run), indent=2, sort_keys=True))
    print("nerf synthetic subset dry-run ok" if args.dry_run else "nerf synthetic subset prepare ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
