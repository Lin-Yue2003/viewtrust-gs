#!/usr/bin/env python3
"""Inspect PR7 training event outputs for a baseline run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, len(list(csv.reader(handle))) - 1)


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_training_events(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "training_events_summary.json"
    training_events_csv = run_dir / "tables" / "training_events.csv"
    densification_events_csv = run_dir / "tables" / "densification_events.csv"
    gaussian_count_csv = run_dir / "tables" / "gaussian_count_timeseries.csv"
    missing = [
        str(path.relative_to(run_dir))
        for path in (
            summary_path,
            training_events_csv,
            densification_events_csv,
            gaussian_count_csv,
        )
        if not path.exists()
    ]
    summary = _json_file(summary_path) if summary_path.exists() else {}
    return {
        "run_dir": str(run_dir),
        "missing_required_paths": missing,
        "has_training_events_summary": summary_path.exists(),
        "has_training_events_csv": training_events_csv.exists(),
        "has_densification_events_csv": densification_events_csv.exists(),
        "has_gaussian_count_timeseries_csv": gaussian_count_csv.exists(),
        "training_event_rows": summary.get(
            "training_event_rows",
            _csv_row_count(training_events_csv),
        ),
        "densification_event_rows": summary.get(
            "densification_event_rows",
            _csv_row_count(densification_events_csv),
        ),
        "densification_trigger_count": summary.get("densification_trigger_count"),
        "initial_gaussian_count": summary.get("initial_gaussian_count"),
        "final_gaussian_count": summary.get("final_gaussian_count"),
        "observation_only": summary.get("observation_only"),
        "warnings": summary.get("warnings", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--require-events", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_training_events(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_events and report["missing_required_paths"]:
        missing = ", ".join(report["missing_required_paths"])
        print(
            "ERROR: training event outputs are required but missing: "
            f"{missing}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
