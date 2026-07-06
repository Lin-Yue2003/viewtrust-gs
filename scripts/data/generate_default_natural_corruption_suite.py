#!/usr/bin/env python3
"""Generate the default PR10 natural corruption condition suite."""

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
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--num-corrupt-train-views", type=int, default=4)
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.datasets.natural_corruptions import (
        DEFAULT_CONDITIONS,
        generate_natural_corruption_condition,
    )

    args = parse_args()
    data_root = _resolve_path(project_root, args.data_root)
    summaries = []
    try:
        for output_condition, corruption_type in DEFAULT_CONDITIONS.items():
            summaries.append(
                generate_natural_corruption_condition(
                    data_root=data_root,
                    scene=args.scene,
                    source_condition=args.source_condition,
                    output_condition=output_condition,
                    corruption_type=corruption_type,
                    seed=args.seed,
                    num_corrupt_train_views=args.num_corrupt_train_views,
                    corrupt_train_fraction=None,
                    corrupt_view_names=None,
                    copy_mode=args.copy_mode,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    preview_count=8,
                )
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"conditions": summaries, "mode": "dry-run" if args.dry_run else "write"}, indent=2, sort_keys=True))
    print("default natural corruption suite dry-run ok" if args.dry_run else "default natural corruption suite generation ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
