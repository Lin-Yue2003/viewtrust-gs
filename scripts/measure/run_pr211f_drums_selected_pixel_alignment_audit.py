#!/usr/bin/env python3
"""Run PR21.1f drums selected-pixel source alignment audit."""

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
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pr200-dir", required=True, type=Path)
    parser.add_argument("--pr211-dir", required=True, type=Path)
    parser.add_argument("--pr210-dir", type=Path)
    parser.add_argument("--scene", default="drums")
    parser.add_argument("--condition", default="corrupt_occluder")
    parser.add_argument("--subset-name", default="seed_20260710")
    parser.add_argument("--views", nargs="*", default=["train_004", "train_009", "train_012", "train_017", "train_007", "train_013"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="")
    parser.add_argument("--top-pixels-per-view", type=int, default=128)
    parser.add_argument("--max-contributors-per-pixel", type=int, default=16)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.pr211f_alignment import build_pr211f_drums_alignment_audit

    del args.device
    del args.max_contributors_per_pixel
    try:
        summary, exit_code = build_pr211f_drums_alignment_audit(
            run_dir=_resolve_path(project_root, args.run_dir),
            pr200_dir=_resolve_path(project_root, args.pr200_dir),
            pr211_dir=_resolve_path(project_root, args.pr211_dir),
            pr210_dir=_resolve_path(project_root, args.pr210_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            views=args.views,
            output_dir=_resolve_path(project_root, args.output_dir),
            top_pixels_per_view=args.top_pixels_per_view,
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
