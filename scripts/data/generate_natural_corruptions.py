#!/usr/bin/env python3
"""Generate one natural corruption condition from a clean mini dataset."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("VIEWTRUST_DATA_ROOT", "./data"))
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--source-condition", default="clean")
    parser.add_argument("--output-condition", required=True)
    parser.add_argument(
        "--corruption-type",
        choices=("occluder", "blur", "exposure", "color_shift", "noise", "mixed"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--num-corrupt-train-views", type=int)
    parser.add_argument("--corrupt-train-fraction", type=float)
    parser.add_argument("--corrupt-view-names", nargs="+")
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.datasets.natural_corruptions import generate_natural_corruption_condition

    args = parse_args()
    try:
        summary = generate_natural_corruption_condition(
            data_root=_resolve_path(project_root, args.data_root),
            scene=args.scene,
            source_condition=args.source_condition,
            output_condition=args.output_condition,
            corruption_type=args.corruption_type,
            seed=args.seed,
            num_corrupt_train_views=args.num_corrupt_train_views,
            corrupt_train_fraction=args.corrupt_train_fraction,
            corrupt_view_names=args.corrupt_view_names,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            preview_count=args.preview_count,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("natural corruption dry-run ok" if args.dry_run else "natural corruption generation ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
