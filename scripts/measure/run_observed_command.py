#!/usr/bin/env python3
"""Run any command with Priority 0 observation-only measurement logging."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("VIEWTRUST_OUTPUT_ROOT", "./outputs"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--label", default="observed-command")
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to observe. Put -- before the command.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()

    from viewtrust.configs.loader import load_simple_yaml
    from viewtrust.logging.writer import Priority0Logger
    from viewtrust.measurement.command_runner import run_observed_command
    from viewtrust.utils.paths import ensure_child_path, resolve_relative_path

    args = parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command. Use: run_observed_command.py -- <command> [args...]")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_simple_yaml(config_path)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = resolve_relative_path(project_root, args.output_root)
    run_id = args.run_id or "observed-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / args.label / run_id
    ensure_child_path(output_root, run_dir)

    logger = Priority0Logger(run_dir=run_dir, run_id=run_id)
    result = run_observed_command(
        command=command,
        logger=logger,
        config=config,
        sample_interval_s=args.sample_interval_s,
        label=args.label,
    )

    print(f"observed command run_dir: {result.run_dir}")
    print(f"returncode: {result.returncode}")
    print(f"elapsed_s: {result.elapsed_s:.6f}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
