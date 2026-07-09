#!/usr/bin/env python3
"""Bind PR17/PR18 view groups onto PR19.2 exact Gaussian logs."""

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
    parser.add_argument("--exact-log-dir", required=True, type=Path)
    parser.add_argument("--pr17-dir", required=True, type=Path)
    parser.add_argument("--pr18-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--copy-pr19-ready-bundle", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.exact_view_group_binding import bind_exact_view_groups

    try:
        summary, exit_code = bind_exact_view_groups(
            exact_log_dir=_resolve_path(project_root, args.exact_log_dir),
            pr17_dir=_resolve_path(project_root, args.pr17_dir),
            pr18_dir=_resolve_path(project_root, args.pr18_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            output_dir=_resolve_path(project_root, args.output_dir),
            top_k=args.top_k,
            write_markdown=args.write_markdown,
            copy_pr19_ready_bundle=args.copy_pr19_ready_bundle,
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
