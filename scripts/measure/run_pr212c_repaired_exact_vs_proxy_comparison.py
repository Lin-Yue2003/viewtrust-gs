#!/usr/bin/env python3
"""Run PR21.2c repaired chair exact-vs-proxy comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr200-chair-dir", required=True, type=Path)
    parser.add_argument("--pr211-chair-dir", required=True, type=Path)
    parser.add_argument("--pr212-chair-dir", required=True, type=Path)
    parser.add_argument("--pr212a-chair-dir", required=True, type=Path)
    parser.add_argument("--pr212b-chair-dir", required=True, type=Path)
    parser.add_argument("--pr213-chair-dir", required=True, type=Path)
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", default="corrupt_occluder")
    parser.add_argument("--subset-name", default="seed_20260710")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.pr212c_repaired_comparison import build_pr212c_repaired_exact_vs_proxy_comparison

    try:
        summary, exit_code = build_pr212c_repaired_exact_vs_proxy_comparison(
            pr200_chair_dir=_resolve_path(project_root, args.pr200_chair_dir),
            pr211_chair_dir=_resolve_path(project_root, args.pr211_chair_dir),
            pr212_chair_dir=_resolve_path(project_root, args.pr212_chair_dir),
            pr212a_chair_dir=_resolve_path(project_root, args.pr212a_chair_dir),
            pr212b_chair_dir=_resolve_path(project_root, args.pr212b_chair_dir),
            pr213_chair_dir=_resolve_path(project_root, args.pr213_chair_dir),
            output_dir=_resolve_path(project_root, args.output_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
