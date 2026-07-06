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


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(number)


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_event_sanity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "invalid_training_event_rows": None,
            "max_visible_gaussian_count": None,
            "max_visibility_ratio": None,
            "max_gaussian_count": None,
            "max_radii_nonzero_count": None,
        }

    invalid_rows = 0
    max_visible: int | None = None
    max_ratio: float | None = None
    max_gaussian: int | None = None
    max_radii_nonzero: int | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gaussian_count = _int_or_none(row.get("gaussian_count"))
            visible_count = _int_or_none(row.get("visible_gaussian_count"))
            visibility_ratio = _float_or_none(row.get("visibility_ratio"))
            radii_nonzero_count = _int_or_none(row.get("radii_nonzero_count"))
            row_invalid = row.get("status") == "invalid"
            if gaussian_count is not None:
                max_gaussian = (
                    gaussian_count
                    if max_gaussian is None
                    else max(max_gaussian, gaussian_count)
                )
                if gaussian_count < 0:
                    row_invalid = True
            if visible_count is not None:
                max_visible = (
                    visible_count
                    if max_visible is None
                    else max(max_visible, visible_count)
                )
                if visible_count < 0:
                    row_invalid = True
            if visibility_ratio is not None:
                max_ratio = (
                    visibility_ratio
                    if max_ratio is None
                    else max(max_ratio, visibility_ratio)
                )
                if visibility_ratio < 0 or visibility_ratio > 1:
                    row_invalid = True
            if radii_nonzero_count is not None:
                max_radii_nonzero = (
                    radii_nonzero_count
                    if max_radii_nonzero is None
                    else max(max_radii_nonzero, radii_nonzero_count)
                )
                if radii_nonzero_count < 0:
                    row_invalid = True
            if (
                gaussian_count is not None
                and visible_count is not None
                and visible_count > gaussian_count
            ):
                row_invalid = True
            if (
                gaussian_count is not None
                and radii_nonzero_count is not None
                and radii_nonzero_count > gaussian_count
            ):
                row_invalid = True
            if row_invalid:
                invalid_rows += 1

    return {
        "invalid_training_event_rows": invalid_rows,
        "max_visible_gaussian_count": max_visible,
        "max_visibility_ratio": max_ratio,
        "max_gaussian_count": max_gaussian,
        "max_radii_nonzero_count": max_radii_nonzero,
    }


def _densification_event_sanity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"invalid_densification_event_rows": None}
    invalid_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            before = _int_or_none(row.get("gaussian_count_before"))
            after = _int_or_none(row.get("gaussian_count_after"))
            delta = _int_or_none(row.get("gaussian_count_delta"))
            row_invalid = row.get("status") == "invalid"
            if after is not None and after < 0:
                row_invalid = True
            if before is not None and before < 0:
                row_invalid = True
            if before is not None and after is not None and delta != after - before:
                row_invalid = True
            if row_invalid:
                invalid_rows += 1
    return {
        "invalid_densification_event_rows": invalid_rows,
    }


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
    sanity = _training_event_sanity(training_events_csv)
    densification_sanity = _densification_event_sanity(densification_events_csv)
    summary_invalid_rows = _int_or_none(summary.get("invalid_training_event_rows"))
    sanity_invalid_rows = sanity["invalid_training_event_rows"]
    invalid_training_event_rows = (
        summary_invalid_rows
        if sanity_invalid_rows is None
        else max(summary_invalid_rows or 0, sanity_invalid_rows)
    )
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
        "requested_iterations": summary.get("requested_iterations"),
        "logged_iteration_count": summary.get("logged_iteration_count"),
        "invalid_training_event_rows": invalid_training_event_rows,
        "invalid_densification_event_rows": densification_sanity[
            "invalid_densification_event_rows"
        ],
        "max_visible_gaussian_count": sanity["max_visible_gaussian_count"],
        "max_visibility_ratio": sanity["max_visibility_ratio"],
        "max_gaussian_count": sanity["max_gaussian_count"],
        "max_radii_nonzero_count": sanity["max_radii_nonzero_count"],
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
    if args.require_events and report["invalid_training_event_rows"]:
        print(
            "ERROR: training event outputs contain invalid scalar rows: "
            f"{report['invalid_training_event_rows']}",
            file=sys.stderr,
        )
        return 1
    if args.require_events and report["invalid_densification_event_rows"]:
        print(
            "ERROR: densification event outputs contain invalid scalar rows: "
            f"{report['invalid_densification_event_rows']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
