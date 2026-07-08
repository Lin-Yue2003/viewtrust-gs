#!/usr/bin/env python3
"""Build one PR13 offline signal output for use by PR14 aggregation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-view-influence-dir", required=True, type=Path)
    parser.add_argument("--corrupt-view-influence-dir", required=True, type=Path)
    parser.add_argument("--view-influence-comparison-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    _bootstrap_project_imports()
    from scripts.measure.build_offline_viewtrust_signals import build_offline_viewtrust_signals

    args = parse_args()
    output_dir = args.output_root / f"offline_viewtrust_{args.condition}_pr14_input"
    summary = build_offline_viewtrust_signals(
        clean_view_influence_dir=args.clean_view_influence_dir,
        corrupt_view_influence_dir=args.corrupt_view_influence_dir,
        view_influence_comparison_dir=args.view_influence_comparison_dir,
        output_dir=output_dir,
        signal_config=None,
        top_k=args.top_k,
        write_markdown=args.write_markdown,
        quiet=False,
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
