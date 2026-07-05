#!/usr/bin/env python3
"""Render clean train/test/target views for a completed baseline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_project_imports() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PR6 clean view evaluation images with official Gaussian Splatting."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--third-party-root", required=True, type=Path)
    parser.add_argument("--trainer", default="gaussian-splatting")
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--iteration", type=int, default=500)
    parser.add_argument("--splits", nargs="+", default=["train", "test", "target"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    _bootstrap_project_imports()

    from viewtrust.evaluation.gaussian_splatting_render import (
        ViewRenderConfig,
        render_clean_views,
    )

    args = parse_args()
    try:
        summary = render_clean_views(
            ViewRenderConfig(
                run_dir=args.run_dir,
                data_root=args.data_root,
                third_party_root=args.third_party_root,
                scene=args.scene,
                condition=args.condition,
                iteration=args.iteration,
                gpu=args.gpu,
                sample_interval_s=args.sample_interval_s,
                splits=tuple(args.splits),
                trainer=args.trainer,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
