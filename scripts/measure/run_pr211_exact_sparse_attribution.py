#!/usr/bin/env python3
"""Run PR21.1 exact sparse pixel-to-Gaussian attribution replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pr200-dir", required=True, type=Path)
    parser.add_argument("--pr210-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--split", default="train")
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-pixels-per-view", type=int, default=128)
    parser.add_argument("--max-contributors-per-pixel", type=int, default=16)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.gsplat_sparse_attribution import build_pr211_exact_sparse_attribution

    try:
        summary, exit_code = build_pr211_exact_sparse_attribution(
            run_dir=_resolve_path(project_root, args.run_dir),
            pr200_dir=_resolve_path(project_root, args.pr200_dir),
            pr210_dir=_resolve_path(project_root, args.pr210_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            iteration=args.iteration,
            split=args.split,
            views=args.views,
            output_dir=_resolve_path(project_root, args.output_dir),
            device=args.device,
            top_pixels_per_view=args.top_pixels_per_view,
            max_contributors_per_pixel=args.max_contributors_per_pixel,
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
