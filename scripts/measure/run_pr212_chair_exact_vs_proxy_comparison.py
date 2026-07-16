#!/usr/bin/env python3
"""Run PR21.2 chair exact-vs-proxy contributor-ID comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path | str | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr200-dir", required=True, type=Path)
    parser.add_argument("--pr211-dir", required=True, type=Path)
    parser.add_argument("--pr201-dir")
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", default="corrupt_occluder")
    parser.add_argument("--subset-name", default="seed_20260710")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--direct-views", nargs="*", default=["train_004", "train_009", "train_012", "train_017"])
    parser.add_argument("--collateral-views", nargs="*", default=["train_014"])
    parser.add_argument("--control-views", nargs="*", default=["train_013"])
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.pr212_exact_vs_proxy import build_pr212_chair_exact_vs_proxy_comparison

    pr201_dir = None if args.pr201_dir in (None, "") else _resolve_path(project_root, args.pr201_dir)
    try:
        summary, exit_code = build_pr212_chair_exact_vs_proxy_comparison(
            pr200_dir=_resolve_path(project_root, args.pr200_dir),
            pr211_dir=_resolve_path(project_root, args.pr211_dir),
            pr201_dir=pr201_dir,
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            output_dir=_resolve_path(project_root, args.output_dir),
            direct_views=args.direct_views,
            collateral_views=args.collateral_views,
            control_views=args.control_views,
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
