#!/usr/bin/env python3
"""Build a consolidated Priority 0 observation report for one run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_NAME = "viewtrust.priority0_report.summary"
SCHEMA_VERSION = 1


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    payload = _json_file(path)
    return payload.get("summary", payload)


def _metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metadata.json"
    return _json_file(path) if path.exists() else {}


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _requested_iterations(metadata: dict[str, Any], training_events: dict[str, Any]) -> int | None:
    explicit = _int_or_none(metadata.get("iterations")) or _int_or_none(
        training_events.get("requested_iterations")
    )
    if explicit is not None:
        return explicit
    command = metadata.get("command")
    if isinstance(command, list):
        for index, token in enumerate(command[:-1]):
            if token == "--iterations":
                return _int_or_none(command[index + 1])
    return None


def _artifact_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".log":
        return "log"
    if suffix == ".ply":
        return "ply"
    return "file"


def _artifact_row(
    run_dir: Path,
    relative_path: str,
    *,
    required: bool,
    artifact_group: str,
    description: str,
) -> dict[str, Any]:
    path = run_dir / relative_path
    exists = path.exists()
    return {
        "relative_path": relative_path,
        "exists": str(exists).lower(),
        "file_type": _artifact_type(path) if exists else "",
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "required": str(required).lower(),
        "artifact_group": artifact_group,
        "description": description,
    }


def _artifact_rows(
    run_dir: Path,
    *,
    include_view_metrics: bool,
    include_training_events: bool,
    include_gaussian_lifecycle: bool,
) -> list[dict[str, Any]]:
    rows = [
        _artifact_row(run_dir, "metadata.json", required=True, artifact_group="run_metadata", description="Run metadata"),
        _artifact_row(run_dir, "summary.json", required=True, artifact_group="run_metadata", description="Observed command summary"),
        _artifact_row(run_dir, "stats.json", required=True, artifact_group="run_metadata", description="Observed command stats"),
        _artifact_row(run_dir, "config_snapshot.json", required=True, artifact_group="run_metadata", description="Configuration snapshot"),
        _artifact_row(run_dir, "stdout.log", required=True, artifact_group="logs", description="Trainer stdout"),
        _artifact_row(run_dir, "stderr.log", required=True, artifact_group="logs", description="Trainer stderr"),
        _artifact_row(run_dir, "trainer_output", required=True, artifact_group="training_output", description="Official trainer output directory"),
        _artifact_row(run_dir, "tables/command_summary.csv", required=True, artifact_group="tables", description="Command timing table"),
        _artifact_row(run_dir, "tables/gpu_memory_samples.csv", required=False, artifact_group="tables", description="GPU memory samples"),
    ]
    if include_training_events:
        rows.extend(
            [
                _artifact_row(run_dir, "training_events_summary.json", required=True, artifact_group="training_events", description="Training event summary"),
                _artifact_row(run_dir, "tables/training_events.csv", required=True, artifact_group="training_events", description="Training event rows"),
                _artifact_row(run_dir, "tables/densification_events.csv", required=True, artifact_group="training_events", description="Densification event rows"),
                _artifact_row(run_dir, "tables/gaussian_count_timeseries.csv", required=True, artifact_group="training_events", description="Gaussian count time series"),
            ]
        )
    if include_gaussian_lifecycle:
        rows.extend(
            [
                _artifact_row(run_dir, "gaussian_lifecycle_summary.json", required=True, artifact_group="gaussian_lifecycle", description="Gaussian lifecycle summary"),
                _artifact_row(run_dir, "tables/gaussian_lifecycle_events.csv", required=True, artifact_group="gaussian_lifecycle", description="Gaussian lifecycle event rows"),
                _artifact_row(run_dir, "tables/gaussian_lifecycle_final.csv", required=True, artifact_group="gaussian_lifecycle", description="Gaussian lifecycle final rows"),
            ]
        )
    if include_view_metrics:
        rows.extend(
            [
                _artifact_row(run_dir, "view_metrics_summary.json", required=True, artifact_group="view_metrics", description="View metrics summary"),
                _artifact_row(run_dir, "tables/view_metrics.csv", required=True, artifact_group="view_metrics", description="Per-view metrics"),
                _artifact_row(run_dir, "tables/view_render_artifacts.csv", required=True, artifact_group="view_metrics", description="Rendered image artifact table"),
            ]
        )
    return rows


def build_report(
    run_dir: Path,
    *,
    include_view_metrics: bool,
    include_training_events: bool,
    include_gaussian_lifecycle: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.resolve()
    metadata = _metadata(run_dir)
    command_summary = _summary(run_dir)
    training_events = _json_file(run_dir / "training_events_summary.json") if (run_dir / "training_events_summary.json").exists() else {}
    lifecycle = _json_file(run_dir / "gaussian_lifecycle_summary.json") if (run_dir / "gaussian_lifecycle_summary.json").exists() else {}
    view_metrics = _json_file(run_dir / "view_metrics_summary.json") if (run_dir / "view_metrics_summary.json").exists() else {}
    artifact_rows = _artifact_rows(
        run_dir,
        include_view_metrics=include_view_metrics,
        include_training_events=include_training_events,
        include_gaussian_lifecycle=include_gaussian_lifecycle,
    )
    missing_required = [
        row["relative_path"]
        for row in artifact_rows
        if row["required"] == "true" and row["exists"] != "true"
    ]

    invalid_event_rows = _int_or_none(training_events.get("invalid_training_event_rows"))
    lifecycle_violations = _int_or_none(lifecycle.get("invariant_violations"))
    dynamics_summary = (
        _json_file(run_dir / "training_dynamics_summary.json")
        if (run_dir / "training_dynamics_summary.json").exists()
        else {}
    )
    final_count = (
        _int_or_none(lifecycle.get("final_gaussian_count"))
        or _int_or_none(training_events.get("final_gaussian_count"))
        or _int_or_none(dynamics_summary.get("final_gaussian_count"))
    )
    priority0_complete = (
        not missing_required
        and command_summary.get("returncode") == 0
        and (not include_training_events or invalid_event_rows == 0)
        and (not include_gaussian_lifecycle or lifecycle_violations == 0)
    )
    warnings: list[str] = []
    if missing_required:
        warnings.append(f"Missing required artifacts: {', '.join(missing_required)}")
    if include_training_events and invalid_event_rows != 0:
        warnings.append("Training event invariant rows are not clean")
    if include_gaussian_lifecycle and lifecycle_violations != 0:
        warnings.append("Gaussian lifecycle invariant violations are present")

    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_id": metadata.get("run_id") or run_dir.name,
        "scene": metadata.get("scene") or training_events.get("scene") or lifecycle.get("scene"),
        "condition": metadata.get("condition")
        or training_events.get("condition")
        or lifecycle.get("condition"),
        "trainer": metadata.get("trainer") or training_events.get("trainer") or lifecycle.get("trainer"),
        "requested_iterations": _requested_iterations(metadata, training_events),
        "training_success": command_summary.get("returncode") == 0,
        "observation_only": command_summary.get("observation_only") is True,
        "has_view_metrics": bool(view_metrics),
        "has_training_events": bool(training_events),
        "has_gaussian_lifecycle": bool(lifecycle),
        "invalid_training_event_rows": invalid_event_rows,
        "gaussian_lifecycle_invariant_violations": lifecycle_violations,
        "final_gaussian_count": final_count,
        "known_gaussian_count": lifecycle.get("known_gaussian_count"),
        "alive_final_count": lifecycle.get("alive_final_count"),
        "dead_final_count": lifecycle.get("dead_final_count"),
        "view_count_total": view_metrics.get("view_count_total"),
        "view_count_by_split": view_metrics.get("view_count_by_split"),
        "priority0_complete": priority0_complete,
        "missing_required_artifacts": missing_required,
        "warnings": warnings,
    }
    return summary, artifact_rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "relative_path",
        "exists",
        "file_type",
        "size_bytes",
        "required",
        "artifact_group",
        "description",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    sections = [
        "# Priority 0 Report",
        "",
        "## Run Identity",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Run dir: `{summary.get('run_dir')}`",
        f"- Scene: `{summary.get('scene')}`",
        f"- Condition: `{summary.get('condition')}`",
        f"- Trainer: `{summary.get('trainer')}`",
        "",
        "## Environment Summary",
        "- See `metadata.json`, `config_snapshot.json`, and server environment docs.",
        "",
        "## Dataset Summary",
        f"- Scene/condition: `{summary.get('scene')}` / `{summary.get('condition')}`",
        "",
        "## Training Command Summary",
        f"- Requested iterations: `{summary.get('requested_iterations')}`",
        f"- Training success: `{summary.get('training_success')}`",
        "",
        "## Final Gaussian Count",
        f"- Final Gaussian count: `{summary.get('final_gaussian_count')}`",
        "",
        "## Training Dynamics Summary",
        "- See training dynamics artifacts when present.",
        "",
        "## View Metrics Summary",
        f"- Has view metrics: `{summary.get('has_view_metrics')}`",
        f"- View count total: `{summary.get('view_count_total')}`",
        "",
        "## Training Event Summary",
        f"- Has training events: `{summary.get('has_training_events')}`",
        f"- Invalid training event rows: `{summary.get('invalid_training_event_rows')}`",
        "",
        "## Densification Event Summary",
        "- See `tables/densification_events.csv`.",
        "",
        "## Gaussian Lifecycle Summary",
        f"- Has lifecycle: `{summary.get('has_gaussian_lifecycle')}`",
        f"- Known Gaussian count: `{summary.get('known_gaussian_count')}`",
        f"- Alive final count: `{summary.get('alive_final_count')}`",
        f"- Dead final count: `{summary.get('dead_final_count')}`",
        f"- Invariant violations: `{summary.get('gaussian_lifecycle_invariant_violations')}`",
        "",
        "## Observation-only Guarantees",
        "- PR9 adds measurement, comparison, validation, and reporting only.",
        "- It does not change loss, optimizer behavior, sampling, rendering, densification, pruning, or opacity reset decisions.",
        "",
        "## Known Limitations",
        "- This report does not prove bitwise determinism.",
        "- No trust score, defense, corruption, or poison condition is implemented.",
        "",
        "## Next PR Recommendations",
        "- PR10 natural corruption condition generation and clean-vs-corrupt observation comparison.",
        "",
        "## Warnings",
        *[f"- {warning}" for warning in summary.get("warnings", [])],
        "" if summary.get("warnings") else "- None",
        "",
    ]
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-view-metrics", action="store_true")
    parser.add_argument("--include-training-events", action="store_true")
    parser.add_argument("--include-gaussian-lifecycle", action="store_true")
    parser.add_argument("--require-priority0-complete", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, manifest_rows = build_report(
        args.run_dir,
        include_view_metrics=args.include_view_metrics,
        include_training_events=args.include_training_events,
        include_gaussian_lifecycle=args.include_gaussian_lifecycle,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "priority0_report_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(args.output_dir / "priority0_artifact_manifest.csv", manifest_rows)
    if args.write_markdown:
        (args.output_dir / "priority0_report.md").write_text(
            _markdown(summary),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_priority0_complete and not summary["priority0_complete"]:
        print("ERROR: Priority 0 report is incomplete.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
