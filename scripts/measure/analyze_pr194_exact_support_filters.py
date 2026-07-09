#!/usr/bin/env python3
"""Analyze stricter exact Gaussian support modes for PR19.4."""

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
    parser.add_argument("--pr193-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument(
        "--support-modes",
        nargs="+",
        default=["broad", "birth", "prune", "high_event", "dominant_source", "low_entropy", "suspicious_alive"],
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--event-percentile", type=float, default=95.0)
    parser.add_argument("--dominant-source-threshold", type=float, default=0.5)
    parser.add_argument("--low-entropy-threshold", type=float, default=0.35)
    parser.add_argument("--min-event-count", type=int, default=3)
    parser.add_argument("--alive-only", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    from viewtrust.analysis.exact_support_filtering import analyze_exact_support_filters

    try:
        summary, exit_code = analyze_exact_support_filters(
            pr193_dir=_resolve_path(project_root, args.pr193_dir),
            output_dir=_resolve_path(project_root, args.output_dir),
            scene=args.scene,
            condition=args.condition,
            subset_name=args.subset_name,
            support_modes=args.support_modes,
            top_k=args.top_k,
            event_percentile=args.event_percentile,
            dominant_source_threshold=args.dominant_source_threshold,
            low_entropy_threshold=args.low_entropy_threshold,
            min_event_count=args.min_event_count,
            alive_only=args.alive_only,
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
