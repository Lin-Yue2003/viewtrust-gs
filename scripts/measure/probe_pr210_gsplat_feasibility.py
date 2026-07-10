#!/usr/bin/env python3
"""Probe PR21.0 gsplat feasibility for an official 3DGS run directory."""

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
    parser.add_argument("--scene", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--split", default="train")
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--compare-official-renders", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--image-width", type=int, default=400)
    parser.add_argument("--image-height", type=int, default=400)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.gsplat_feasibility import run_gsplat_feasibility_probe

    try:
        summary, exit_code = run_gsplat_feasibility_probe(
            run_dir=_resolve_path(project_root, args.run_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            iteration=args.iteration,
            split=args.split,
            views=args.views,
            output_dir=_resolve_path(project_root, args.output_dir),
            device=args.device,
            max_views=args.max_views,
            allow_missing=args.allow_missing,
            strict=args.strict,
            compare_official_renders=args.compare_official_renders,
            metadata_only=args.metadata_only,
            skip_render=args.skip_render,
            image_width=args.image_width,
            image_height=args.image_height,
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
