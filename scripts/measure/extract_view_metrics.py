#!/usr/bin/env python3
"""Extract PR6 per-view clean metrics from rendered images."""

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
        description="Extract ViewTrust-GS PR6 per-view clean metrics."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scene", default="chair")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--iteration", type=int, default=500)
    parser.add_argument("--splits", nargs="+", default=["train", "test", "target"])
    parser.add_argument("--require-renders", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    _bootstrap_project_imports()

    from viewtrust.evaluation.view_metrics import ViewMetricsConfig, extract_view_metrics

    args = parse_args()
    try:
        summary = extract_view_metrics(
            ViewMetricsConfig(
                run_dir=args.run_dir,
                scene=args.scene,
                condition=args.condition,
                iteration=args.iteration,
                require_renders=args.require_renders,
                splits=tuple(args.splits),
            )
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
