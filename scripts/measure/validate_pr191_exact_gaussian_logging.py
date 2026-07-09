#!/usr/bin/env python3
"""Validate PR19.1 exact Gaussian lifecycle logging outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_project_imports() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    return project_root


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR19.1 Exact Gaussian Lifecycle Logging Validation",
        "",
        "This validation is offline observation only. It is not a defense and does not modify training behavior.",
        "",
        "## Summary",
        f"- Identity consistency: `{summary.get('identity_consistency_passed')}`",
        f"- Parent/root consistency: `{summary.get('parent_child_consistency_passed')}`",
        f"- Row index used as stable ID: `{summary.get('uses_row_index_as_stable_id')}`",
        f"- Exact log files present: `{summary.get('exported_files_exist')}`",
        "",
        "Stable Gaussian IDs are monotonic sidecar identifiers. Tensor row indices are only diagnostic metadata.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_logs(*, exact_log_dir: Path, output_dir: Path, write_markdown: bool) -> tuple[dict[str, Any], int]:
    _bootstrap_project_imports()
    from viewtrust.analysis.exact_gaussian_lifecycle import (
        REQUIRED_EXACT_FILES,
        compute_exact_support_summary,
        compute_exact_view_group_overlap,
        load_json,
        validate_exact_gaussian_logs,
        write_artifact_manifest,
        write_csv_rows,
        write_json,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_exact_gaussian_logs(exact_log_dir)
    exact_summary = load_json(exact_log_dir / "exact_gaussian_logging_summary.json")
    missing_rows = [
        {
            "missing_path": str(exact_log_dir / name),
            "status": "missing",
            "details": f"required exact log output missing: {name}",
        }
        for name in REQUIRED_EXACT_FILES
        if not (exact_log_dir / name).is_file()
    ]
    identity_rows = [
        {
            "check_name": "identity_consistency_passed",
            "passed": validation["identity_consistency_passed"],
            "details": ";".join(validation["validation_errors"]),
        },
        {
            "check_name": "no_duplicate_alive_gaussian_ids",
            "passed": validation["no_duplicate_alive_gaussian_ids"],
            "details": "",
        },
        {
            "check_name": "row_count_matches_current_gaussian_count",
            "passed": validation["row_count_matches_current_gaussian_count"],
            "details": "",
        },
    ]
    parent_rows = [
        {
            "check_name": "parent_ids_exist_or_empty",
            "passed": validation["parent_ids_exist_or_empty"],
            "details": "",
        },
        {
            "check_name": "root_ids_exist",
            "passed": validation["root_ids_exist"],
            "details": "",
        },
        {
            "check_name": "prune_events_reference_existing_ids",
            "passed": validation["prune_events_reference_existing_ids"],
            "details": "",
        },
    ]
    summary = {
        "schema_name": "viewtrust.pr191.exact_gaussian_lifecycle_logging.validation_summary",
        "schema_version": 1,
        "observation_only": True,
        "training_intervention": False,
        "defense_enabled": False,
        "exact_log_dir": str(exact_log_dir),
        "identity_consistency_passed": validation["identity_consistency_passed"],
        "parent_child_consistency_passed": validation["parent_ids_exist_or_empty"] and validation["root_ids_exist"],
        "prune_consistency_passed": validation["prune_events_reference_existing_ids"],
        "uses_row_index_as_stable_id": exact_summary.get("uses_row_index_as_stable_id"),
        "stable_gaussian_ids_enabled": exact_summary.get("stable_gaussian_ids_enabled"),
        "exported_files_exist": validation["exported_files_exist"],
        "identity_row_count": validation["identity_row_count"],
        "event_row_count": validation["event_row_count"],
        "support_row_count": validation["support_row_count"],
        "validation_errors": validation["validation_errors"],
        "validation_warnings": validation["validation_warnings"],
    }
    write_json(output_dir / "pr191_exact_gaussian_logging_validation_summary.json", summary)
    write_csv_rows(
        output_dir / "pr191_identity_consistency.csv",
        identity_rows,
        ["check_name", "passed", "details"],
    )
    write_csv_rows(
        output_dir / "pr191_parent_child_consistency.csv",
        parent_rows,
        ["check_name", "passed", "details"],
    )
    support_rows = compute_exact_support_summary(exact_log_dir)
    support_fields = list(support_rows[0]) if support_rows else ["gaussian_id"]
    write_csv_rows(output_dir / "pr191_support_summary.csv", support_rows, support_fields)
    overlap_rows = compute_exact_view_group_overlap(exact_log_dir)
    if overlap_rows:
        write_csv_rows(output_dir / "pr191_view_group_overlap.csv", overlap_rows, list(overlap_rows[0]))
    write_csv_rows(
        output_dir / "pr191_missing_outputs.csv",
        missing_rows,
        ["missing_path", "status", "details"],
    )
    _write_report(output_dir / "pr191_report.md", summary)
    write_artifact_manifest(output_dir, exact_log_dir)
    return summary, 0 if summary["identity_consistency_passed"] and summary["exported_files_exist"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-log-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    project_root = _bootstrap_project_imports()
    args = parse_args()
    try:
        summary, exit_code = validate_logs(
            exact_log_dir=_resolve_path(project_root, args.exact_log_dir),
            output_dir=_resolve_path(project_root, args.output_dir),
            write_markdown=args.write_markdown,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
