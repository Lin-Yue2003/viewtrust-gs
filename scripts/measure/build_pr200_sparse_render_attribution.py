#!/usr/bin/env python3
"""Build PR20.0 sparse residual-weighted Gaussian attribution logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr193-dir", required=True, type=Path)
    parser.add_argument("--pr195-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--top-pixels", type=int, default=512)
    parser.add_argument("--top-gaussians-per-pixel", type=int, default=16)
    parser.add_argument("--residual-metric", choices=("l1", "l2", "abs_rgb", "alpha_weighted_l1"), default="l1")
    parser.add_argument(
        "--artifact-mask-mode",
        choices=("top_residual", "corruption_mask_if_available", "full_image_sampled", "external_mask_csv"),
        default="top_residual",
    )
    parser.add_argument("--external-mask-csv", type=Path)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.sparse_render_attribution import build_sparse_render_attribution

    try:
        summary, exit_code = build_sparse_render_attribution(
            pr193_dir=_resolve_path(project_root, args.pr193_dir),
            pr195_dir=_resolve_path(project_root, args.pr195_dir),
            run_dir=_resolve_path(project_root, args.run_dir),
            data_root=_resolve_path(project_root, args.data_root),
            output_dir=_resolve_path(project_root, args.output_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            views=args.views,
            max_views=args.max_views,
            top_pixels=args.top_pixels,
            top_gaussians_per_pixel=args.top_gaussians_per_pixel,
            residual_metric=args.residual_metric,
            artifact_mask_mode=args.artifact_mask_mode,
            external_mask_csv=_resolve_path(project_root, args.external_mask_csv),
            downsample=args.downsample,
            write_markdown=args.write_markdown,
            strict=args.strict,
            allow_missing=args.allow_missing,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
