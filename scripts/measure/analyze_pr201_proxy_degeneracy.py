#!/usr/bin/env python3
"""Analyze PR20.0 proxy candidate-pool degeneracy."""

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
    parser.add_argument("--pr200-dir", action="append", required=True, type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--condition")
    parser.add_argument("--subset-name")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--weight-uniformity-tol", type=float, default=1e-9)
    parser.add_argument("--reuse-threshold", type=float, default=0.95)
    parser.add_argument("--degenerate-jaccard-threshold", type=float, default=0.95)
    parser.add_argument("--max-report-rows", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.proxy_degeneracy_diagnosis import PR201Config, analyze_pr201_proxy_degeneracy

    try:
        summary, exit_code = analyze_pr201_proxy_degeneracy(
            pr200_dirs=[_resolve_path(project_root, path) for path in args.pr200_dir],
            output_dir=_resolve_path(project_root, args.output_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            config=PR201Config(
                top_k=args.top_k,
                weight_uniformity_tol=args.weight_uniformity_tol,
                reuse_threshold=args.reuse_threshold,
                degenerate_jaccard_threshold=args.degenerate_jaccard_threshold,
                max_report_rows=args.max_report_rows,
            ),
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
