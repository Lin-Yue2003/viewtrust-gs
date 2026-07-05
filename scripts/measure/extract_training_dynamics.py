#!/usr/bin/env python3
"""Extract post-hoc training dynamics from an observed baseline run."""

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
        description="Extract ViewTrust-GS PR5 training dynamics from an observed run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument(
        "--allow-missing-tensorboard",
        action="store_true",
        default=True,
        help="Allow extraction to succeed when TensorBoard scalar data is absent.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the compact JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    _bootstrap_project_imports()

    from viewtrust.training.dynamics import (
        TrainingDynamicsExtractionConfig,
        extract_training_dynamics,
    )

    args = parse_args()
    try:
        summary = extract_training_dynamics(
            TrainingDynamicsExtractionConfig(
                run_dir=args.run_dir,
                require_success=args.require_success,
                allow_missing_tensorboard=args.allow_missing_tensorboard,
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
