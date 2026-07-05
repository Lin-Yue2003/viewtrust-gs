#!/usr/bin/env python3
"""Inspect PR8 Gaussian lifecycle outputs for a baseline run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return value in (True, "true", "True", "1", 1)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_final_rows(rows: list[dict[str, str]], summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    known_count = _int_or_none(summary.get("known_gaussian_count"))
    final_count = _int_or_none(summary.get("final_gaussian_count"))
    alive_count = _int_or_none(summary.get("alive_final_count"))
    dead_count = _int_or_none(summary.get("dead_final_count"))

    if known_count is not None and len(rows) != known_count:
        errors.append("gaussian_lifecycle_final.csv row count != known_gaussian_count")
    if final_count is not None and alive_count is not None and alive_count != final_count:
        errors.append("alive_final_count != final_gaussian_count")
    if (
        known_count is not None
        and alive_count is not None
        and dead_count is not None
        and alive_count + dead_count != known_count
    ):
        errors.append("alive_final_count + dead_final_count != known_gaussian_count")

    gaussian_ids: set[int] = set()
    final_indices: set[int] = set()
    for row in rows:
        gaussian_id = _int_or_none(row.get("gaussian_id"))
        if gaussian_id is None:
            errors.append("missing gaussian_id")
        elif gaussian_id in gaussian_ids:
            errors.append(f"duplicate gaussian_id: {gaussian_id}")
        else:
            gaussian_ids.add(gaussian_id)

        lifetime = _int_or_none(row.get("lifetime_iterations"))
        if lifetime is not None and lifetime < 0:
            errors.append(f"negative lifetime_iterations for gaussian_id={gaussian_id}")

        if _truthy(row.get("alive")):
            final_index = _int_or_none(row.get("final_index"))
            if final_index is None:
                errors.append(f"alive row missing final_index for gaussian_id={gaussian_id}")
            elif final_index in final_indices:
                errors.append(f"duplicate alive final_index: {final_index}")
            else:
                final_indices.add(final_index)
        elif _int_or_none(row.get("death_iteration")) is None:
            errors.append(f"dead row missing death_iteration for gaussian_id={gaussian_id}")

    return errors


def inspect_gaussian_lifecycle(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "gaussian_lifecycle_summary.json"
    events_csv = run_dir / "tables" / "gaussian_lifecycle_events.csv"
    final_csv = run_dir / "tables" / "gaussian_lifecycle_final.csv"
    required = (summary_path, events_csv, final_csv)
    missing = [str(path.relative_to(run_dir)) for path in required if not path.exists()]
    summary = _json_file(summary_path) if summary_path.exists() else {}
    final_rows = _csv_rows(final_csv)
    csv_errors = _validate_final_rows(final_rows, summary) if final_rows else []
    summary_violations = _int_or_none(summary.get("invariant_violations")) or 0
    invariant_violations = summary_violations + len(csv_errors)
    return {
        "run_dir": str(run_dir),
        "has_gaussian_lifecycle_summary": summary_path.exists(),
        "has_gaussian_lifecycle_events_csv": events_csv.exists(),
        "has_gaussian_lifecycle_final_csv": final_csv.exists(),
        "observation_only": summary.get("observation_only"),
        "enabled": summary.get("enabled"),
        "initial_gaussian_count": summary.get("initial_gaussian_count"),
        "final_gaussian_count": summary.get("final_gaussian_count"),
        "known_gaussian_count": summary.get("known_gaussian_count"),
        "alive_final_count": summary.get("alive_final_count"),
        "dead_final_count": summary.get("dead_final_count"),
        "birth_event_count": summary.get("birth_event_count"),
        "clone_birth_count": summary.get("clone_birth_count"),
        "split_birth_count": summary.get("split_birth_count"),
        "densification_birth_count": summary.get("densification_birth_count"),
        "prune_death_count": summary.get("prune_death_count"),
        "lifecycle_event_rows": summary.get("lifecycle_event_rows"),
        "final_lifecycle_rows": summary.get("final_lifecycle_rows", len(final_rows)),
        "invariant_violations": invariant_violations,
        "csv_invariant_errors": csv_errors,
        "missing_required_paths": missing,
        "warnings": summary.get("warnings", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--require-lifecycle", action="store_true")
    parser.add_argument("--require-no-invariant-violations", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_gaussian_lifecycle(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_lifecycle and report["missing_required_paths"]:
        missing = ", ".join(report["missing_required_paths"])
        print(
            f"ERROR: Gaussian lifecycle outputs are required but missing: {missing}",
            file=sys.stderr,
        )
        return 1
    if args.require_no_invariant_violations and report["invariant_violations"]:
        print(
            "ERROR: Gaussian lifecycle invariant violations detected: "
            f"{report['invariant_violations']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
