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
            "has_view_identity": False,
            "missing_view_identity_rows": None,
            "unique_training_view_count": None,
            "unique_sampled_view_count": None,
            "sampled_train_view_count": None,
            "sampled_test_view_count": None,
            "sampled_target_view_count": None,
            "sampled_unknown_view_count": None,
            "unexpected_non_train_sampled_view_count": None,
            "unexpected_non_train_sampled_views": [],
            "sampled_view_rows": None,
        }

    invalid_rows = 0
    max_visible: int | None = None
    max_ratio: float | None = None
    max_gaussian: int | None = None
    max_radii_nonzero: int | None = None
    sampled_view_rows = 0
    missing_view_identity_rows = 0
    training_view_names: set[str] = set()
    sampled_view_names: set[str] = set()
    sampled_train_views: set[str] = set()
    sampled_test_views: set[str] = set()
    sampled_target_views: set[str] = set()
    sampled_unknown_views: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_type = row.get("event_type")
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
            if event_type == "iteration_metrics":
                sampled_view_rows += 1
                view_name = str(row.get("view_name") or "")
                if not view_name:
                    missing_view_identity_rows += 1
                else:
                    view_split = str(row.get("view_split") or "")
                    sampled_view_names.add(view_name)
                    if view_split == "train" or view_name.startswith("train_"):
                        training_view_names.add(view_name)
                        sampled_train_views.add(view_name)
                    elif view_split == "test" or view_name.startswith("test_"):
                        sampled_test_views.add(view_name)
                    elif view_split == "target" or view_name.startswith("target_"):
                        sampled_target_views.add(view_name)
                    else:
                        sampled_unknown_views.add(view_name)
            if row_invalid:
                invalid_rows += 1

    return {
        "invalid_training_event_rows": invalid_rows,
        "max_visible_gaussian_count": max_visible,
        "max_visibility_ratio": max_ratio,
        "max_gaussian_count": max_gaussian,
        "max_radii_nonzero_count": max_radii_nonzero,
        "has_view_identity": sampled_view_rows > 0 and missing_view_identity_rows == 0,
        "missing_view_identity_rows": missing_view_identity_rows,
        "unique_training_view_count": len(training_view_names),
        "unique_sampled_view_count": len(sampled_view_names),
        "sampled_train_view_count": len(sampled_train_views),
        "sampled_test_view_count": len(sampled_test_views),
        "sampled_target_view_count": len(sampled_target_views),
        "sampled_unknown_view_count": len(sampled_unknown_views),
        "unexpected_non_train_sampled_view_count": len(sampled_test_views | sampled_target_views),
        "unexpected_non_train_sampled_views": sorted(sampled_test_views | sampled_target_views),
        "sampled_view_rows": sampled_view_rows,
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
    warnings = list(summary.get("warnings", []))
    if sanity["unexpected_non_train_sampled_view_count"]:
        warnings.append(
            "Training iteration metrics include non-train views. Use --eval during official 3DGS training to keep test cameras held out."
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
        "has_view_identity": sanity["has_view_identity"],
        "missing_view_identity_rows": sanity["missing_view_identity_rows"],
        "unique_training_view_count": sanity["unique_training_view_count"],
        "unique_sampled_view_count": sanity["unique_sampled_view_count"],
        "sampled_train_view_count": sanity["sampled_train_view_count"],
        "sampled_test_view_count": sanity["sampled_test_view_count"],
        "sampled_target_view_count": sanity["sampled_target_view_count"],
        "sampled_unknown_view_count": sanity["sampled_unknown_view_count"],
        "unexpected_non_train_sampled_view_count": sanity[
            "unexpected_non_train_sampled_view_count"
        ],
        "unexpected_non_train_sampled_views": sanity["unexpected_non_train_sampled_views"],
        "sampled_view_rows": sanity["sampled_view_rows"],
        "densification_trigger_count": summary.get("densification_trigger_count"),
        "initial_gaussian_count": summary.get("initial_gaussian_count"),
        "final_gaussian_count": summary.get("final_gaussian_count"),
        "observation_only": summary.get("observation_only"),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--require-events", action="store_true")
    parser.add_argument("--require-view-identity", action="store_true")
    parser.add_argument("--require-train-only-sampling", action="store_true")
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
    if args.require_view_identity and report["missing_view_identity_rows"]:
        print(
            "ERROR: training event outputs are missing sampled view identity rows: "
            f"{report['missing_view_identity_rows']}",
            file=sys.stderr,
        )
        return 1
    if (
        args.require_train_only_sampling
        and report["unexpected_non_train_sampled_view_count"]
    ):
        print(
            "ERROR: training iteration metrics include non-train sampled views: "
            f"{report['unexpected_non_train_sampled_views']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
